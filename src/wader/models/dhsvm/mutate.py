"""Mutation operators: write a copy of a DHSVM setup with one known error injected.

Each operator names the rules that must catch it and says what DHSVM itself
does with the mutant (``MutationInfo.dhsvm``); ``tests/test_dhsvm_runs.py``
checks that statement by running DHSVM on the Chiwawa case.

A mutant replaces one or more of the files the INPUT names and keeps every
other byte: the topology operators rewrite only the order and outlet fields of
the stream network and recompute routing orders as 1 + max(order of upstream
segments), so the result still satisfies DHSVM's routing-order requirement
(channel.c:538-549) and DHSVM reads and runs it.

Two kinds of mutant do not replace a file the INPUT names. Some change the
text of an INPUT entry (the output directory). Others change a state file,
which DHSVM finds through a directory entry and a name it builds itself; they
bring their own state directory, with links to the files they leave alone.
"""

from __future__ import annotations

import shutil
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.io import replace_token, rewrite_network
from wader.models.dhsvm.maps import (
    STATE_DIRECTORY,
    MapSpec,
    expected_maps,
    state_files,
    state_stamp,
)

NETWORK = ("ROUTING", "Stream Network File")
STREAM_MAP = ("ROUTING", "Stream Map File")
CLASSES = ("ROUTING", "Stream Class File")
DEM = ("TERRAIN", "DEM File")
SOIL_DEPTH = ("SOILS", "Soil Depth File")
OUTPUT_DIRECTORY = ("OUTPUT", "Output Directory")

# DHSVM builds state file names in char[NAMESIZE + 1] (InitModelState.c:57; settings.h:94).
# The longest, "Interception.State.MM.DD.YYYY.HH.MM.SS.bin", has 42 characters.
STATE_PREFIX_MAX = 127 - 42


class MutationError(Exception):
    pass


@dataclass(frozen=True)
class MutationInfo:
    name: str
    title: str
    targets: tuple[str, ...]
    description: str
    dhsvm: str = "runs to the end without an error"  # what DHSVM does with the mutant


@dataclass
class Mutated:
    """A mutant: new contents for some of the files the INPUT names, new text for some
    INPUT entries, and files of its own."""

    info: MutationInfo
    files: dict[tuple[str, str], bytes]  # INPUT (section, key) -> new file content
    changes: list[str] = field(default_factory=list)
    downstream: dict[int, int | None] | None = None  # topology operators only
    order: dict[int, int] | None = None
    # INPUT (section, key) -> new text; "{out}" stands for the mutant's directory
    values: dict[tuple[str, str], str] = field(default_factory=dict)
    # files of the mutant's own, by path below its directory: new content, or an
    # existing file to link to
    tree: dict[str, bytes] = field(default_factory=dict)
    links: dict[str, Path] = field(default_factory=dict)


Operator = Callable[[DhsvmCase, dict], Mutated]
MUTATIONS: dict[str, tuple[MutationInfo, Operator]] = {}


def _operator(info: MutationInfo) -> Callable[[Operator], Operator]:
    def deco(fn: Operator) -> Operator:
        MUTATIONS[info.name] = (info, fn)
        return fn
    return deco


def _text(lines: list[str]) -> bytes:
    return "\n".join(lines).encode("latin-1")


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
    main = max(outlets, key=lambda o: (_size(children, o), -o))
    return down, children, main


def _size(children: dict[int, list[int]], root: int) -> int:
    """Number of segments in the tree that drains through ``root`` (itself included)."""
    n, todo, seen = 0, [root], {root}
    while todo:
        x = todo.pop()
        n += 1
        for c in children[x]:
            if c not in seen:
                seen.add(c)
                todo.append(c)
    return n


def _path_lengths(case: DhsvmCase, children: dict[int, list[int]], root: int) -> dict[int, float]:
    """Channel length from each segment of the tree down to ``root`` (both included)."""
    by_id = case.network.by_id()
    dist = {root: by_id[root].length}
    todo = deque([root])
    while todo:
        x = todo.popleft()
        for c in children[x]:
            if c not in dist:
                dist[c] = dist[x] + by_id[c].length
                todo.append(c)
    return dist


def _finish(case: DhsvmCase, info: MutationInfo, down: dict[int, int | None],
            changes: list[str]) -> Mutated:
    order = routing_orders(down)
    text = rewrite_network(case.network, down, order)
    return Mutated(info, {NETWORK: text.encode("latin-1")}, changes, down, order)


def _keep_orders(case: DhsvmCase, info: MutationInfo, down: dict[int, int | None],
                 changes: list[str], outlet_sentinel: str | None = None) -> Mutated:
    """Rewrite outlet fields only; routing orders stay as they are."""
    order = {s.id: s.order for s in case.network.segments}
    text = rewrite_network(case.network, down, order, outlet_sentinel=outlet_sentinel)
    return Mutated(info, {NETWORK: text.encode("latin-1")}, changes, down, order)


# ------------------------------------------------- topology (network links)
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
def reverse_network(case: DhsvmCase, params: dict) -> Mutated:
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
def jump(case: DhsvmCase, params: dict) -> Mutated:
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
def force_wire(case: DhsvmCase, params: dict) -> Mutated:
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
    dhsvm="runs to the end; the network 'Totals' outflow is 0",
))
def negative_outlet(case: DhsvmCase, params: dict) -> Mutated:
    down, _, _ = _tree(case)
    outlets = [s for s, d in down.items() if d is None]
    changes = [f"segment {s}: outlet written as -1" for s in outlets]
    return _keep_orders(case, MUTATIONS["negative-outlet"][0], down, changes,
                        outlet_sentinel="-1")


@_operator(MutationInfo(
    name="dangling-outlet",
    title="Point a segment at an outlet segment that does not exist",
    targets=("DHSVM-NET-003",),
    description=(
        "Rewrite the outlet ID of one headwater segment (the first in the file, or "
        "`segment`) to an ID that no segment has."
    ),
    dhsvm="stops while reading the network: 'cannot find outlet', exit 5",
))
def dangling_outlet(case: DhsvmCase, params: dict) -> Mutated:
    down, children, main = _tree(case)
    ids = set(down)
    if "segment" in params:
        seg = int(params["segment"])
        if seg not in ids:
            raise MutationError(f"no segment {seg}")
    else:
        heads = [s.id for s in case.network.segments if not children[s.id] and s.id != main]
        if not heads:
            raise MutationError("the outlet has no upstream segments")
        seg = heads[0]
    missing = max(ids) + 1
    if missing > 65535:  # stay below the unsigned short limit (channel.h:16)
        missing = next(i for i in range(1, 65536) if i not in ids)
    new_down = dict(down)
    new_down[seg] = missing
    changes = [f"segment {seg}: drains to {down[seg] or 'outside'} -> {missing}, "
               "which names no segment"]
    return _keep_orders(case, MUTATIONS["dangling-outlet"][0], new_down, changes)


@_operator(MutationInfo(
    name="cycle",
    title="Make a segment drain into one of its own upstream segments",
    targets=("DHSVM-NET-003",),
    description=(
        "Take the segment with the largest upstream tree that is not an outlet and rewrite "
        "its outlet ID to its own largest tributary. The two then form a cycle, and "
        "everything upstream of them never reaches the outlet. Routing orders are kept."
    ),
    dhsvm="runs to the end without an error; the water that enters the cycle is lost",
))
def cycle(case: DhsvmCase, params: dict) -> Mutated:
    down, children, main = _tree(case)
    candidates = [s for s in down if down[s] is not None and children[s]]
    if not candidates:
        raise MutationError("no segment with both an upstream and a downstream segment")
    seg = max(candidates, key=lambda s: (_size(children, s), -s))
    tributary = max(children[seg], key=lambda c: (_size(children, c), -c))
    new_down = dict(down)
    new_down[seg] = tributary
    changes = [f"segment {seg}: drains to {down[seg]} -> {tributary}, its own tributary "
               f"({_size(children, seg)} of {len(down)} segments now drain into the cycle)"]
    return _keep_orders(case, MUTATIONS["cycle"][0], new_down, changes)


# ------------------------------------------------- stream network records
@_operator(MutationInfo(
    name="zero-slope",
    title="Set the slope of one segment to 0",
    targets=("DHSVM-NET-001",),
    description=(
        "Rewrite the slope of one segment (the first in the file, or `segment`) as 0.00000, "
        "as a generator does that takes the slope from a flat DEM without a lower bound."
    ),
    dhsvm="stops while reading the network: 'channel slope (0.000000) invalid', exit 5",
))
def zero_slope(case: DhsvmCase, params: dict) -> Mutated:
    net = case.network
    if not net.segments:
        raise MutationError("the network has no segments")
    seg = net.by_id().get(int(params["segment"])) if "segment" in params else net.segments[0]
    if seg is None:
        raise MutationError(f"no segment {params['segment']}")
    lines = list(net.lines)
    lines[seg.lineno - 1] = replace_token(lines[seg.lineno - 1], 2, "0.00000")
    return Mutated(MUTATIONS["zero-slope"][0], {NETWORK: _text(lines)},
                   [f"segment {seg.id}: slope {seg.slope:g} -> 0"])


@_operator(MutationInfo(
    name="missing-class",
    title="Give one segment a channel class that is not in the class file",
    targets=("DHSVM-NET-010",),
    description=(
        "Rewrite the class of one segment (the first in the file, or `segment`) to an ID "
        "the class file does not define."
    ),
    dhsvm="stops while reading the network: 'channel class N not found', exit 5",
))
def missing_class(case: DhsvmCase, params: dict) -> Mutated:
    net = case.network
    if not net.segments:
        raise MutationError("the network has no segments")
    seg = net.by_id().get(int(params["segment"])) if "segment" in params else net.segments[0]
    if seg is None:
        raise MutationError(f"no segment {params['segment']}")
    known = {c.id for c in case.classes.classes}
    new = max(known | {s.cls for s in net.segments}) + 1
    lines = list(net.lines)
    lines[seg.lineno - 1] = replace_token(lines[seg.lineno - 1], 4, str(new))
    return Mutated(MUTATIONS["missing-class"][0], {NETWORK: _text(lines)},
                   [f"segment {seg.id}: class {seg.cls} -> {new}, which the class file "
                    "does not define"])


# PNNL's unedited class template: widths by contributing area, depths by slope
# (DHSVM-PNNL@e6f333d CreateStreamNetwork_PythonV/channelclass.py:35-124).
_TEMPLATE_WIDTHS = (0.5, 1.0, 2.0, 3.0, 4.0, 4.5)
_TEMPLATE_DEPTHS = (0.03, 0.05, 0.1)


@_operator(MutationInfo(
    name="placeholder-classes",
    title="Replace class widths and depths by PNNL's template placeholders",
    targets=("DHSVM-NET-010",),
    description=(
        "Rewrite the width and bank height of every channel class with the values of PNNL's "
        "class template (widths 0.5-4.5 m, depths 0.03-0.10 m), which the template says "
        "must be edited by hand. Manning's n is kept."
    ),
    dhsvm="runs to the end without an error; the hydrograph changes (shallow depths slow "
          "the routing)",
))
def placeholder_classes(case: DhsvmCase, params: dict) -> Mutated:
    table = case.classes
    if not table.classes:
        raise MutationError("the class file defines no classes")
    lines = list(table.lines)
    changes = []
    for k, c in enumerate(table.classes):
        width = _TEMPLATE_WIDTHS[k % 6]
        depth = _TEMPLATE_DEPTHS[(k // 6) % 3]
        line = replace_token(lines[c.lineno - 1], 2, f"{depth:g}")
        lines[c.lineno - 1] = replace_token(line, 1, f"{width:.1f}")
        changes.append(f"class {c.id}: width {c.width:g} -> {width:g} m, bank height "
                       f"{c.bank_height:g} -> {depth:g} m")
    return Mutated(MUTATIONS["placeholder-classes"][0], {CLASSES: _text(lines)}, changes)


# ------------------------------------------------------------- stream map
@_operator(MutationInfo(
    name="map-swap-row-col",
    title="Swap the column and row fields of the stream map",
    targets=("DHSVM-NET-007",),
    description=(
        "Write every stream-map record as 'row column ...' instead of 'column row ...'. On "
        "a grid that is not square some records then fall outside the grid."
    ),
    dhsvm="stops while reading the stream map: 'bad coordinates', exit 5",
))
def map_swap_row_col(case: DhsvmCase, params: dict) -> Mutated:
    sm, g = case.stream_map, case.grid
    outside = int(((sm.row >= g.ncols) | (sm.col >= g.nrows)).sum())
    if not outside:
        raise MutationError("after the swap every record is still inside the grid, so "
                            "DHSVM would not reject the map")
    lines = list(sm.lines)
    for lineno, col, row in zip(sm.lineno, sm.col, sm.row, strict=True):
        line = replace_token(lines[lineno - 1], 1, str(col))
        lines[lineno - 1] = replace_token(line, 0, str(row))
    return Mutated(MUTATIONS["map-swap-row-col"][0], {STREAM_MAP: _text(lines)},
                   [f"{len(sm)} records: column and row swapped; {outside} now lie outside "
                    f"the {g.nrows} x {g.ncols} grid"])


@_operator(MutationInfo(
    name="zero-length-record",
    title="Set the channel length of one stream-map record to 0",
    targets=("DHSVM-NET-007",),
    description=(
        "Rewrite the length of the first stream-map record that is alone in an in-basin "
        "cell as 0.00000, as can happen when a segment only touches the corner of a cell."
    ),
    dhsvm="runs to the end with exit status 0; streamflow downstream of the cell is NaN",
))
def zero_length_record(case: DhsvmCase, params: dict) -> Mutated:
    sm, g = case.stream_map, case.grid
    in_grid = (sm.row >= 0) & (sm.row < g.nrows) & (sm.col >= 0) & (sm.col < g.ncols)
    per_cell = Counter(zip(sm.row[in_grid].tolist(), sm.col[in_grid].tolist(), strict=True))
    for k in np.flatnonzero(in_grid):
        r, c = int(sm.row[k]), int(sm.col[k])
        if per_cell[(r, c)] == 1 and case.mask[r, c] and sm.length[k] > 0:
            break
    else:
        raise MutationError("no in-basin cell holds exactly one stream-map record")
    lines = list(sm.lines)
    lineno = int(sm.lineno[k])
    lines[lineno - 1] = replace_token(lines[lineno - 1], 3, "0.00000")
    return Mutated(MUTATIONS["zero-length-record"][0], {STREAM_MAP: _text(lines)},
                   [f"line {lineno}: segment {int(sm.segment[k])} in cell row {r}, column {c}: "
                    f"length {sm.length[k]:g} -> 0"])


# ------------------------------------------------------------ binary maps
@_operator(MutationInfo(
    name="truncate-map",
    title="Cut a binary map file short",
    targets=("DHSVM-MAP-001",),
    description=(
        "Keep only the first half of one binary map (the DEM, or the map whose label is "
        "given as `map`, e.g. 'soil depth'), as after an interrupted copy."
    ),
    dhsvm="stops when it reads the map: 'Error while reading file', exit 2",
))
def truncate_map(case: DhsvmCase, params: dict) -> Mutated:
    label = str(params.get("map", "DEM"))
    # only maps that an INPUT entry names directly (not the monthly shading files)
    specs = {s.label: s for s in expected_maps(case) if s.path == case.path(*s.entry)}
    if label not in specs:
        raise MutationError(f"no map {label!r}; choose from {', '.join(specs)}")
    spec = specs[label]
    data = spec.path.read_bytes()
    return Mutated(MUTATIONS["truncate-map"][0], {spec.entry: data[: len(data) // 2]},
                   [f"{label} ({spec.path.name}): {len(data)} -> {len(data) // 2} bytes"])


@_operator(MutationInfo(
    name="float64-dem",
    title="Write the DEM as 8-byte floats",
    targets=("DHSVM-MAP-001",),
    description=(
        "Write the DEM with 8-byte instead of 4-byte floats (numpy's default number type), "
        "which doubles the file. DHSVM reads the first half as 4-byte floats."
    ),
    dhsvm="reports no read error and computes with meaningless elevations; on Chiwawa it "
          "stops at the first time step with error 47 (table index out of bounds)",
))
def float64_dem(case: DhsvmCase, params: dict) -> Mutated:
    path = case.path(*DEM)
    dem = np.fromfile(path, dtype="<f4")
    data = dem.astype("<f8").tobytes()
    return Mutated(MUTATIONS["float64-dem"][0], {DEM: data},
                   [f"DEM ({path.name}): 4-byte -> 8-byte floats, {dem.nbytes} -> "
                    f"{len(data)} bytes"])


@_operator(MutationInfo(
    name="shallow-soil",
    title="Make the soil as shallow as the root zones in a few cells",
    targets=("DHSVM-MAP-003",),
    description=(
        "Set the soil depth of a few basin cells (9, or `cells`) to exactly the total root "
        "depth of their vegetation class, as a soil depth map made without looking at the "
        "vegetation table can."
    ),
    dhsvm="stops before it reads the network: it prints the first such cell ('Soil depth "
          "is ..., Root depth is ...') and exits with status 255",
))
def shallow_soil(case: DhsvmCase, params: dict) -> Mutated:
    n = int(params.get("cells", 9))
    g = case.grid
    path = case.path(*SOIL_DEPTH)
    depth = np.fromfile(path, dtype="<f4")
    if depth.size != g.nrows * g.ncols:
        raise MutationError(f"{path} does not hold {g.nrows} x {g.ncols} 4-byte floats")
    depth = depth.reshape(g.nrows, g.ncols)
    types, veg = case.veg_types, case.veg_type
    cells = np.argwhere(case.mask & (veg >= 1) & (veg <= len(types)))
    if n < 1 or len(cells) < n:
        raise MutationError(f"the basin has {len(cells)} cells with a vegetation class of "
                            f"the table; cannot change {n}")
    first = (len(cells) - n) // 2
    totals = np.array([0.0] + [t.total_root_depth for t in types], dtype="<f4")
    for row, col in cells[first:first + n]:
        depth[row, col] = totals[veg[row, col]]
    row, col = (int(v) for v in cells[first])
    return Mutated(MUTATIONS["shallow-soil"][0], {SOIL_DEPTH: depth.tobytes()},
                   [f"soil depth ({path.name}): {n} basin cells from row {row}, column {col} "
                    "set to the total root depth of their vegetation class "
                    f"({float(totals[veg[row, col]]):.6g} m there)"])


_STATE_KINDS = {"canopy interception state": "Interception", "snow state": "Snow",
                "soil state": "Soil"}


def _state_mutant(case: DhsvmCase, name: str, label: str, data: bytes,
                  change: str) -> Mutated:
    """A mutant whose state directory holds ``data`` as the ``label`` state and links to
    the other state files DHSVM reads at start-up."""
    stamp = state_stamp(case)
    tree: dict[str, bytes] = {}
    links: dict[str, Path] = {}
    for spec in state_files(case):
        new_name = f"state/{_STATE_KINDS[spec.label]}.State.{stamp}.bin"
        if spec.label == label:
            tree[new_name] = data
        elif spec.path.is_file():
            links[new_name] = spec.path
    for kind in ("Channel", "Hydrograph"):  # tables DHSVM reads from the same place
        table = case.start_path(f"{case.state_prefix}{kind}.State.{stamp}")
        if table.is_file():
            links[f"state/{kind}.State.{stamp}"] = table
    return Mutated(MUTATIONS[name][0], {}, [change], values={STATE_DIRECTORY: "{out}/state/"},
                   tree=tree, links=links)


def _state_spec(case: DhsvmCase, label: str) -> MapSpec:
    specs = {s.label: s for s in state_files(case)}
    if label not in specs:
        raise MutationError(f"no state {label!r}; choose from {', '.join(specs)}")
    if not specs[label].path.is_file():
        raise MutationError(f"the setup has no {specs[label].path}")
    return specs[label]


@_operator(MutationInfo(
    name="truncate-state",
    title="Cut a state file short",
    targets=("DHSVM-STA-001",),
    description=(
        "Keep only the first half of one gridded state file (the soil state, or the one "
        "whose label is given as `state`, e.g. 'snow state'), as after an interrupted copy."
    ),
    dhsvm="stops when it restores the model state: 'Error while reading file', exit 2",
))
def truncate_state(case: DhsvmCase, params: dict) -> Mutated:
    label = str(params.get("state", "soil state"))
    spec = _state_spec(case, label)
    data = spec.path.read_bytes()
    return _state_mutant(case, "truncate-state", label, data[: len(data) // 2],
                         f"{label} ({spec.path.name}): {len(data)} -> {len(data) // 2} bytes")


@_operator(MutationInfo(
    name="state-extra-layer",
    title="Write the soil state for one more soil layer",
    targets=("DHSVM-STA-001",),
    description=(
        "Write the soil state as a setup with one more soil layer would: one more moisture "
        "data set and one more temperature data set (copies of the last layer's), which "
        "makes the file two data sets longer."
    ),
    dhsvm="reports no error and runs to the end, but takes every variable after the "
          "moisture layers from the wrong data set; on Chiwawa soil temperatures become "
          "surface water and the first flows are more than 100 times larger",
))
def state_extra_layer(case: DhsvmCase, params: dict) -> Mutated:
    spec = _state_spec(case, "soil state")
    g = case.grid
    n_soil = (spec.layers - 4) // 2
    data = np.fromfile(spec.path, dtype="<f4")
    if data.size != spec.layers * g.nrows * g.ncols:
        raise MutationError(f"{spec.path} does not hold {spec.layers} data sets of "
                            f"{g.nrows} x {g.ncols} 4-byte floats")
    sets = data.reshape(spec.layers, g.nrows, g.ncols)
    moist, surface = sets[: n_soil + 1], sets[n_soil + 1: n_soil + 2]
    temps, rest = sets[n_soil + 2: 2 * n_soil + 2], sets[2 * n_soil + 2:]
    longer = np.concatenate([moist, moist[-1:], surface, temps, temps[-1:], rest])
    return _state_mutant(case, "state-extra-layer", "soil state", longer.tobytes(),
                         f"soil state ({spec.path.name}): {spec.layers} -> {len(longer)} data "
                         f"sets, as for {n_soil + 1} soil layers")


@_operator(MutationInfo(
    name="missing-output-dir",
    title="Name an output directory that does not exist",
    targets=("DHSVM-RUN-002",),
    description="Set Output Directory to no-such-directory/, which is not created.",
    dhsvm="stops when it opens its first output file, after reading the maps and the "
          "network: 'Cannot open file: no-such-directory/Aggregated.Values', exit 3",
))
def missing_output_dir(case: DhsvmCase, params: dict) -> Mutated:
    new = "no-such-directory/"
    if case.start_path(new).exists():
        raise MutationError(f"{case.start_path(new)} exists")
    return Mutated(MUTATIONS["missing-output-dir"][0], {},
                   [f"Output Directory: {case.output_prefix!r} -> {new!r}"],
                   values={OUTPUT_DIRECTORY: new})


@_operator(MutationInfo(
    name="output-no-slash",
    title="Drop the '/' at the end of the output directory",
    targets=("DHSVM-RUN-002",),
    description="Remove the '/' that ends Output Directory.",
    dhsvm="runs to the end, but writes its output next to the directory, under names that "
          "start with the directory's name (outputAggregated.Values for output)",
))
def output_no_slash(case: DhsvmCase, params: dict) -> Mutated:
    old = case.output_prefix
    new = old.rstrip("/")
    if new == old or not new:
        raise MutationError(f"Output Directory = {old!r} has no '/' at its end to remove")
    return Mutated(MUTATIONS["output-no-slash"][0], {},
                   [f"Output Directory: {old!r} -> {new!r}"], values={OUTPUT_DIRECTORY: new})


# ------------------------------------------------------------------ writing
def mutate_case(case: DhsvmCase, name: str, out_dir: str | Path,
                params: dict | None = None) -> tuple[Path, Mutated]:
    """Write the mutant's files and ``out_dir/INPUT.<name>``.

    Each replaced file keeps its name and goes to ``out_dir``, as do the files of the
    mutant's own. The new INPUT differs from the original only in the entries that name
    replaced files, which point to the new files by absolute path, and in the entries the
    mutant changes, so the mutated setup runs from the same working directory as the
    original.
    """
    if name not in MUTATIONS:
        raise MutationError(f"unknown mutation {name!r}; choose from {', '.join(MUTATIONS)}")
    _, fn = MUTATIONS[name]
    mutated = fn(case, params or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = case.config
    for (section, key), content in mutated.files.items():
        new_path = (out_dir / case.path(section, key).name).resolve()
        new_path.write_bytes(content)
        config = config.with_value(section, key, str(new_path))
    root = out_dir.resolve()
    for rel, content in mutated.tree.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(content)
    for rel, source in mutated.links.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        try:
            target.symlink_to(source.resolve())
        except OSError:  # no symbolic links here: copy instead
            shutil.copyfile(source, target)
    for (section, key), value in mutated.values.items():
        value = value.replace("{out}", str(root))
        if (section, key) == STATE_DIRECTORY and len(value) > STATE_PREFIX_MAX:
            raise MutationError(
                f"the mutant's state directory {value} has {len(value)} characters, but "
                f"DHSVM has room for {STATE_PREFIX_MAX} in front of a state file name "
                "(InitModelState.c:57); choose a shorter output directory")
        config = config.with_value(section, key, value)
    input_path = out_dir / f"INPUT.{name}"
    config.write(input_path)
    return input_path, mutated
