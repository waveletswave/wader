"""Mutation operators: write a copy of a DHSVM setup with one known error injected.

Each operator names the rules that must catch it. The mutated stream network
keeps every line of the original except the fields that change (order and
outlet ID); routing orders are recomputed as 1 + max(order of upstream
segments), so the result still satisfies DHSVM's routing-order requirement
(channel.c:538-549) and DHSVM reads and runs it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.io import rewrite_network


class MutationError(Exception):
    pass


@dataclass(frozen=True)
class MutationInfo:
    name: str
    title: str
    targets: tuple[str, ...]
    description: str


@dataclass
class MutatedNetwork:
    info: MutationInfo
    text: str
    downstream: dict[int, int | None]
    order: dict[int, int]
    changes: list[str] = field(default_factory=list)


Operator = Callable[[DhsvmCase, dict], MutatedNetwork]
MUTATIONS: dict[str, tuple[MutationInfo, Operator]] = {}


def _operator(info: MutationInfo) -> Callable[[Operator], Operator]:
    def deco(fn: Operator) -> Operator:
        MUTATIONS[info.name] = (info, fn)
        return fn
    return deco


# ------------------------------------------------------------------ helpers
def routing_orders(down: dict[int, int | None]) -> dict[int, int]:
    """order = 1 + max(order of upstream segments); headwaters get 1."""
    children: dict[int, list[int]] = defaultdict(list)
    for s, d in down.items():
        if d is not None:
            children[d].append(s)
    order: dict[int, int] = {}
    for start in down:
        if start in order:
            continue
        stack, on_path = [(start, False)], set()
        while stack:
            node, done = stack.pop()
            if done:
                order[node] = 1 + max((order[c] for c in children[node]), default=0)
                on_path.discard(node)
                continue
            if node in order:
                continue
            if node in on_path:
                raise MutationError(f"links form a cycle through segment {node}")
            on_path.add(node)
            stack.append((node, True))
            stack.extend((c, False) for c in children[node] if c not in order)
    return order


def _tree(case: DhsvmCase) -> tuple[dict[int, int | None], dict[int, list[int]], int]:
    """Intended downstream map, children lists, and the outlet with the largest tree."""
    down = case.network.intended_downstream()
    children: dict[int, list[int]] = defaultdict(list)
    for s, d in down.items():
        if d is not None:
            children[d].append(s)
    outlets = [s for s, d in down.items() if d is None]
    if not outlets:
        raise MutationError("the network has no outlet")

    def size(root: int) -> int:
        n, todo = 0, [root]
        while todo:
            x = todo.pop()
            n += 1
            todo.extend(children[x])
        return n

    main = max(outlets, key=lambda o: (size(o), -o))
    return down, children, main


def _path_lengths(case: DhsvmCase, children: dict[int, list[int]], root: int) -> dict[int, float]:
    """Channel length from each segment of the tree down to ``root`` (both included)."""
    by_id = case.network.by_id()
    dist = {root: by_id[root].length}
    todo = deque([root])
    while todo:
        x = todo.popleft()
        for c in children[x]:
            dist[c] = dist[x] + by_id[c].length
            todo.append(c)
    return dist


def _finish(case: DhsvmCase, info: MutationInfo, down: dict[int, int | None],
            changes: list[str]) -> MutatedNetwork:
    order = routing_orders(down)
    text = rewrite_network(case.network, down, order)
    return MutatedNetwork(info, text, down, order, changes)


# ---------------------------------------------------------------- operators
@_operator(MutationInfo(
    name="reverse-network",
    title="Reverse the flow direction of the main stem",
    targets=("DHSVM-NET-004", "DHSVM-NET-006"),
    description=(
        "Treat the network as an undirected tree and re-root it at the headwater segment "
        "farthest (by channel length) from the outlet: every link on the path between the two "
        "flips direction and that headwater becomes the outlet (outlet ID 0). Orders are "
        "recomputed. Slopes, lengths, classes and the stream map are unchanged."
    ),
))
def reverse_network(case: DhsvmCase, params: dict) -> MutatedNetwork:
    down, children, main = _tree(case)
    dist = _path_lengths(case, children, main)
    leaves = [n for n in dist if not children[n] and n != main]
    if not leaves:
        raise MutationError("the outlet has no upstream segments")
    new_root = max(leaves, key=lambda n: (dist[n], -n))
    adj: dict[int, list[int]] = defaultdict(list)
    for n in dist:
        if down[n] is not None:
            adj[n].append(down[n])
            adj[down[n]].append(n)
    parent: dict[int, int | None] = {new_root: None}
    todo = deque([new_root])
    while todo:
        x = todo.popleft()
        for y in adj[x]:
            if y not in parent:
                parent[y] = x
                todo.append(y)
    new_down = dict(down)
    new_down.update(parent)
    flipped = [n for n in dist if new_down[n] != down[n]]
    changes = [f"segment {n}: drains to {down[n] or 'outside'} -> {new_down[n] or 'outside'}"
               for n in flipped]
    info = MUTATIONS["reverse-network"][0]
    out = _finish(case, info, new_down, changes)
    out.changes.insert(0, f"re-rooted at headwater segment {new_root} "
                          f"({dist[new_root] / 1000:.1f} km of channel from outlet {main}); "
                          f"{len(flipped)} links changed")
    return out


def _leaf_gaps(case: DhsvmCase, children: dict[int, list[int]],
               main: int) -> list[tuple[float, int]]:
    cells = case.cells
    if main not in cells:
        raise MutationError(f"outlet segment {main} has no stream-map cells")
    oc = cells[main]
    out = []
    for n in _path_lengths(case, children, main):
        if children[n] or n == main or n not in cells or n in children[main]:
            continue
        a = cells[n]
        gap = float(np.sqrt(((a[:, None, :] - oc[None, :, :]) ** 2).sum(-1).min()))
        out.append((gap, n))
    return sorted(out, key=lambda t: (-t[0], t[1]))


@_operator(MutationInfo(
    name="jump",
    title="Connect headwater segments to a distant outlet",
    targets=("DHSVM-NET-005",),
    description=(
        "Rewire the `count` (default 1) headwater segments farthest from the outlet segment "
        "(by distance between stream-map cells) so that they drain directly into the outlet. "
        "The outlet is the lowest segment, so no link rises; the links jump across the basin."
    ),
))
def jump(case: DhsvmCase, params: dict) -> MutatedNetwork:
    count = int(params.get("count", 1))
    down, children, main = _tree(case)
    picks = [(g, n) for g, n in _leaf_gaps(case, children, main) if g > 1.5][:count]
    if len(picks) < count:
        raise MutationError("not enough headwater segments away from the outlet")
    new_down = dict(down)
    changes = []
    for gap, n in picks:
        new_down[n] = main
        changes.append(f"segment {n}: drains to {down[n]} -> {main} ({gap:.1f} cells away)")
    return _finish(case, MUTATIONS["jump"][0], new_down, changes)


@_operator(MutationInfo(
    name="force-wire",
    title="Wire many segments directly to the outlet",
    targets=("DHSVM-NET-006", "DHSVM-NET-005"),
    description=(
        "Rewire the `count` (default 8) headwater segments farthest from the outlet so that "
        "they drain directly into it, as a generator does when it attaches every segment it "
        "could not connect to one sink."
    ),
))
def force_wire(case: DhsvmCase, params: dict) -> MutatedNetwork:
    count = int(params.get("count", 8))
    down, children, main = _tree(case)
    picks = [n for _, n in _leaf_gaps(case, children, main)][:count]
    if len(picks) < count:
        raise MutationError(f"only {len(picks)} headwater segments available")
    new_down = dict(down)
    changes = [f"segment {n}: drains to {down[n]} -> {main}" for n in picks]
    for n in picks:
        new_down[n] = main
    return _finish(case, MUTATIONS["force-wire"][0], new_down, changes)


@_operator(MutationInfo(
    name="negative-outlet",
    title="Write the basin outlet as -1",
    targets=("DHSVM-NET-013",),
    description=(
        "Write the outlet ID of every outlet segment as -1, the convention of some network "
        "generators. DHSVM reads it as 1."
    ),
))
def negative_outlet(case: DhsvmCase, params: dict) -> MutatedNetwork:
    down, _, _ = _tree(case)
    outlets = [s for s, d in down.items() if d is None]
    changes = [f"segment {s}: outlet written as -1" for s in outlets]
    keep_order = {s.id: s.order for s in case.network.segments}
    text = rewrite_network(case.network, down, keep_order, outlet_sentinel="-1")
    return MutatedNetwork(MUTATIONS["negative-outlet"][0], text, down, keep_order, changes)


# ------------------------------------------------------------------ writing
def mutate_case(case: DhsvmCase, name: str, out_dir: str | Path,
                params: dict | None = None) -> tuple[Path, MutatedNetwork]:
    """Write ``out_dir/<network file>`` and ``out_dir/INPUT.<name>``.

    The new INPUT differs from the original only in ``[ROUTING] Stream Network
    File``, which points to the mutated file by absolute path, so the mutated
    setup runs from the same working directory as the original.
    """
    if name not in MUTATIONS:
        raise MutationError(f"unknown mutation {name!r}; choose from {', '.join(MUTATIONS)}")
    _, fn = MUTATIONS[name]
    mutated = fn(case, params or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net_path = (out_dir / case.network.path.name).resolve()
    net_path.write_bytes(mutated.text.encode("latin-1"))
    config = case.config.with_value("ROUTING", "Stream Network File", str(net_path))
    input_path = out_dir / f"INPUT.{name}"
    config.write(input_path)
    return input_path, mutated
