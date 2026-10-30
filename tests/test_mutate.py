"""Mutation operators produce networks DHSVM reads and runs, with one error injected."""

from __future__ import annotations

import pytest

from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.config import DhsvmConfig
from wader.models.dhsvm.io import read_stream_network
from wader.models.dhsvm.mutate import MUTATIONS, MutationError, mutate_case, routing_orders


def test_routing_orders():
    assert routing_orders({1: 3, 2: 3, 3: None}) == {1: 1, 2: 1, 3: 2}
    assert routing_orders({1: 2, 2: 3, 3: None, 4: 3}) == {1: 1, 2: 2, 3: 3, 4: 1}
    with pytest.raises(MutationError, match="cycle"):
        routing_orders({1: 2, 2: 1})


def assert_routable(down: dict, order: dict) -> None:
    """What DHSVM needs to route every segment (channel.c:538-549)."""
    assert sum(d is None for d in down.values()) >= 1
    assert all(order[s] < order[d] for s, d in down.items() if d is not None)
    assert sorted(set(order.values())) == list(range(1, max(order.values()) + 1))


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_mutations_stay_readable_and_routable(synthetic, tmp_path, name):
    path, work = synthetic
    case = DhsvmCase(path, work)
    params = {"count": 2} if name == "force-wire" else None
    new_input, m = mutate_case(case, name, tmp_path / "mut", params)
    net = read_stream_network(tmp_path / "mut" / "stream.network.dat")
    assert not [n for n in net.notes if "error" in n.message]
    assert [s.id for s in net.segments] == [s.id for s in case.network.segments]
    for old, new in zip(case.network.segments, net.segments, strict=True):
        assert (old.slope, old.length, old.cls, old.save) == (new.slope, new.length, new.cls,
                                                              new.save)
    assert_routable(net.intended_downstream(), {s.id: s.order for s in net.segments})
    # the new INPUT differs from the original only in the network file
    old_lines = case.config.text.splitlines()
    new_lines = DhsvmConfig.read(new_input).text.splitlines()
    changed = [b for a, b in zip(old_lines, new_lines, strict=True) if a != b]
    assert len(changed) == 1 and changed[0].startswith("Stream Network File")


def test_force_wire_needs_enough_headwaters(synthetic, tmp_path):
    path, work = synthetic
    with pytest.raises(MutationError):
        mutate_case(DhsvmCase(path, work), "force-wire", tmp_path / "mut")
