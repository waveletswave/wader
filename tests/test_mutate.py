"""Mutation operators inject one known error and leave everything else alone."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import NCOLS, NROWS

from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.config import DhsvmConfig
from wader.models.dhsvm.io import read_stream_classes, read_stream_map, read_stream_network
from wader.models.dhsvm.mutate import MUTATIONS, MutationError, mutate_case, routing_orders

# operators that only rewire links: DHSVM must still read and route the result
TOPOLOGY = ("reverse-network", "jump", "force-wire", "negative-outlet")
SYNTHETIC_PARAMS = {"force-wire": {"count": 2}}


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


@pytest.mark.parametrize("name", TOPOLOGY)
def test_topology_mutations_stay_readable_and_routable(synthetic, tmp_path, name):
    path, work = synthetic
    case = DhsvmCase(path, work)
    _, m = mutate_case(case, name, tmp_path / "mut", SYNTHETIC_PARAMS.get(name))
    net = read_stream_network(tmp_path / "mut" / "stream.network.dat")
    assert not [n for n in net.notes if n.kind == "error"]
    assert [s.id for s in net.segments] == [s.id for s in case.network.segments]
    for old, new in zip(case.network.segments, net.segments, strict=True):
        assert (old.slope, old.length, old.cls, old.save) == (new.slope, new.length, new.cls,
                                                              new.save)
    assert_routable(net.intended_downstream(), {s.id: s.order for s in net.segments})
    assert m.downstream == net.intended_downstream()


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_a_mutant_differs_only_in_the_files_it_names(synthetic, tmp_path, name):
    path, work = synthetic
    case = DhsvmCase(path, work)
    new_input, m = mutate_case(case, name, tmp_path / "mut", SYNTHETIC_PARAMS.get(name))
    assert m.files and m.changes and m.info.dhsvm
    mutant = DhsvmCase(new_input, work)
    for section, key in m.files:
        assert mutant.path(section, key).read_bytes() != case.path(section, key).read_bytes()
    # the new INPUT differs from the original only in the entries for those files
    old_lines = case.config.text.splitlines()
    new_lines = DhsvmConfig.read(new_input).text.splitlines()
    changed = [b for a, b in zip(old_lines, new_lines, strict=True) if a != b]
    assert sorted(line.split("=")[0].strip() for line in changed) == sorted(k for _, k in m.files)


def test_force_wire_needs_enough_headwaters(synthetic, tmp_path):
    path, work = synthetic
    with pytest.raises(MutationError):
        mutate_case(DhsvmCase(path, work), "force-wire", tmp_path / "mut")


def test_cycle_and_dangling_outlet_keep_the_orders(synthetic, tmp_path):
    path, work = synthetic
    case = DhsvmCase(path, work)
    orders = {s.id: s.order for s in case.network.segments}
    _, m = mutate_case(case, "cycle", tmp_path / "cycle")
    assert m.downstream == {1: 3, 2: 3, 3: 1, 4: 5, 5: None, 6: 5}  # 3 now feeds its tributary 1
    assert m.order == orders
    _, m = mutate_case(case, "dangling-outlet", tmp_path / "dangling")
    net = read_stream_network(tmp_path / "dangling" / "stream.network.dat")
    assert net.by_id()[1].outlet == 7 and 7 not in net.by_id()
    assert {s.id: s.order for s in net.segments} == orders


def test_record_mutations_touch_one_field(synthetic, tmp_path):
    path, work = synthetic
    case = DhsvmCase(path, work)
    mutate_case(case, "zero-slope", tmp_path / "slope")
    net = read_stream_network(tmp_path / "slope" / "stream.network.dat")
    assert [s.slope for s in net.segments] == [0.0, 0.05, 0.05, 0.05, 0.05, 0.05]
    mutate_case(case, "missing-class", tmp_path / "class")
    net = read_stream_network(tmp_path / "class" / "stream.network.dat")
    assert [s.cls for s in net.segments] == [2, 1, 1, 1, 1, 1]
    mutate_case(case, "zero-length-record", tmp_path / "length")
    sm = read_stream_map(tmp_path / "length" / "stream.map.dat")
    assert sm.length[0] == 0.0 and (sm.length[1:] == case.stream_map.length[1:]).all()
    assert (sm.row == case.stream_map.row).all() and (sm.col == case.stream_map.col).all()


def test_map_swap_row_col(synthetic, tmp_path):
    path, work = synthetic
    case = DhsvmCase(path, work)
    _, m = mutate_case(case, "map-swap-row-col", tmp_path / "mut")
    sm = read_stream_map(tmp_path / "mut" / "stream.map.dat")
    assert (sm.row == case.stream_map.col).all() and (sm.col == case.stream_map.row).all()
    assert (sm.col >= NCOLS).any() and NROWS != NCOLS
    assert "now lie outside" in m.changes[0]


def test_placeholder_classes_follow_the_pnnl_template(synthetic, tmp_path):
    path, work = synthetic
    (work / "input" / "stream.class.dat").write_text(
        "".join(f"{i} {i}.0 1.5 0.04 0.0\n" for i in range(1, 9)))
    mutate_case(DhsvmCase(path, work), "placeholder-classes", tmp_path / "mut")
    table = read_stream_classes(tmp_path / "mut" / "stream.class.dat")
    assert [c.width for c in table.classes] == [0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 0.5, 1.0]
    assert [c.bank_height for c in table.classes] == [0.03] * 6 + [0.05] * 2
    assert {c.friction for c in table.classes} == {0.04}


def test_binary_map_mutations(synthetic, tmp_path):
    path, work = synthetic
    case = DhsvmCase(path, work)
    size = NROWS * NCOLS * 4
    mutate_case(case, "truncate-map", tmp_path / "short")
    assert (tmp_path / "short" / "dem.bin").stat().st_size == size // 2
    mutate_case(case, "truncate-map", tmp_path / "short2", {"map": "soil depth"})
    assert (tmp_path / "short2" / "soild.bin").stat().st_size == size // 2
    with pytest.raises(MutationError, match="choose from"):
        mutate_case(case, "truncate-map", tmp_path / "short3", {"map": "nope"})
    mutate_case(case, "float64-dem", tmp_path / "f8")
    f8 = np.fromfile(tmp_path / "f8" / "dem.bin", dtype="<f8").reshape(NROWS, NCOLS)
    assert (f8 == case.dem).all()
