"""Mutation operators inject one known error and leave everything else alone."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import NCOLS, NROWS, STAMP

from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.config import DhsvmConfig
from wader.models.dhsvm.io import read_stream_classes, read_stream_map, read_stream_network
from wader.models.dhsvm.maps import state_files
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
def test_a_mutant_differs_only_in_what_it_names(synthetic, short_tmp, name):
    path, work = synthetic
    case = DhsvmCase(path, work)
    new_input, m = mutate_case(case, name, short_tmp / "mut", SYNTHETIC_PARAMS.get(name))
    assert (m.files or m.values) and m.changes and m.info.dhsvm
    mutant = DhsvmCase(new_input, work)
    for section, key in m.files:
        assert mutant.path(section, key).read_bytes() != case.path(section, key).read_bytes()
    for section, key in m.values:
        assert mutant.config.get(section, key) != case.config.get(section, key)
    # the new INPUT differs from the original only in the entries the mutant names
    old_lines = case.config.text.splitlines()
    new_lines = DhsvmConfig.read(new_input).text.splitlines()
    changed = [b for a, b in zip(old_lines, new_lines, strict=True) if a != b]
    assert sorted(line.split("=")[0].strip() for line in changed) == sorted(
        key for _, key in [*m.files, *m.values])


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


def test_shallow_soil_changes_the_given_number_of_basin_cells(synthetic, tmp_path):
    path, work = synthetic
    case = DhsvmCase(path, work)
    new_input, m = mutate_case(case, "shallow-soil", tmp_path / "mut", {"cells": 4})
    old = np.fromfile(case.path("SOILS", "Soil Depth File"), dtype="<f4")
    new = np.fromfile(DhsvmCase(new_input, work).path("SOILS", "Soil Depth File"), dtype="<f4")
    changed = np.flatnonzero(old != new)
    assert len(changed) == 4 and case.mask.ravel()[changed].all()
    # exactly the single-precision sum that DHSVM computes (CheckOut.c:81-84)
    total = np.float32(np.float32(np.float32(0.10) + np.float32(0.25)) + np.float32(0.40))
    assert set(new[changed]) == {total}
    assert "4 basin cells" in m.changes[0]
    with pytest.raises(MutationError):
        mutate_case(case, "shallow-soil", tmp_path / "x", {"cells": NROWS * NCOLS})


def test_state_mutants_bring_their_own_state_directory(synthetic, short_tmp):
    path, work = synthetic
    case = DhsvmCase(path, work)
    original = {s.label: s.path for s in state_files(case)}
    for name, sets in (("truncate-state", 5), ("state-extra-layer", 12)):
        out = short_tmp / name
        new_input, _ = mutate_case(case, name, out)
        mutant = DhsvmCase(new_input, work)
        assert mutant.state_prefix == f"{out.resolve()}/state/"
        files = {s.label: s.path for s in state_files(mutant)}
        assert files["soil state"].stat().st_size == sets * NROWS * NCOLS * 4
        for label in ("canopy interception state", "snow state"):  # the originals, linked
            assert files[label].resolve() == original[label].resolve()
        assert (out / "state" / f"Channel.State.{STAMP}").read_text()
    # the extra data sets are copies of the last moisture and temperature layers
    longer = np.fromfile(short_tmp / "state-extra-layer" / "state" / f"Soil.State.{STAMP}.bin",
                         dtype="<f4").reshape(12, NROWS, NCOLS)
    assert (longer[4] == longer[3]).all() and (longer[9] == longer[8]).all()
    _, m = mutate_case(case, "truncate-state", short_tmp / "snow", {"state": "snow state"})
    assert m.changes[0].startswith("snow state")
    with pytest.raises(MutationError, match="choose from"):
        mutate_case(case, "truncate-state", short_tmp / "x", {"state": "ice"})


def test_a_state_directory_too_long_for_dhsvm_is_refused(synthetic, tmp_path):
    path, work = synthetic
    with pytest.raises(MutationError, match="has room for 85"):
        mutate_case(DhsvmCase(path, work), "truncate-state", tmp_path / ("d" * 90))


def test_output_directory_mutations(synthetic, tmp_path):
    path, work = synthetic
    case = DhsvmCase(path, work)
    for name, value in (("missing-output-dir", "no-such-directory/"),
                        ("output-no-slash", "output")):
        new_input, _ = mutate_case(case, name, tmp_path / name)
        assert DhsvmCase(new_input, work).output_prefix == value
    no_slash = tmp_path / "output-no-slash" / "INPUT.output-no-slash"
    with pytest.raises(MutationError, match="no '/'"):
        mutate_case(DhsvmCase(no_slash, work), "output-no-slash", tmp_path / "again")
    (work / "no-such-directory").mkdir()
    with pytest.raises(MutationError, match="exists"):
        mutate_case(case, "missing-output-dir", tmp_path / "taken")
