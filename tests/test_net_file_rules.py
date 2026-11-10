"""NET-001, NET-003, NET-007 and NET-010 on the synthetic case: what DHSVM rejects in
the three channel files (L0), and what it accepts and then misuses (L1, L2)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    DOWN,
    INPUT_TEMPLATE,
    NCOLS,
    NROWS,
    ORDER,
    SOIL_DEPTH,
    make_synthetic_case,
    map_text,
    network_text,
    synthetic_mask,
)

from wader.core import Outcome, Severity
from wader.models import dhsvm

NET_RULES = ("DHSVM-NET-001", "DHSVM-NET-003", "DHSVM-NET-004", "DHSVM-NET-005",
             "DHSVM-NET-006", "DHSVM-NET-007", "DHSVM-NET-010", "DHSVM-NET-013")


def check(work, rule, **params):
    return dhsvm.check(work / "INPUT.synthetic", work, rules=[rule],
                       params=params or None).results[0]


def errors(res, level=None):
    return [f for f in res.findings
            if f.severity == Severity.ERROR and (level is None or f.level == level)]


def warnings(res):
    return [f for f in res.findings if f.severity == Severity.WARNING]


def write(work, name, text):
    (work / "input" / name).write_text(text)


@pytest.fixture
def work(tmp_path):
    make_synthetic_case(tmp_path)
    return tmp_path


# ------------------------------------------------------------------ NET-001
def test_net001_reports_every_record_dhsvm_rejects(work):
    write(work, "stream.network.dat", "\n".join([
        "1 1 0.05 120.0 1 3",
        "2 0 0.05 170.0 1 3",  # order 0
        "3 2 -0.01 150.0 1 5",  # negative slope
        "4 1 0.05 0 1 5",  # length 0 (an integer is accepted for a real field)
        '5 3 0.05 210.0 1 0 SAVE "Outlet"',
        "6 1 1.5E-3 72.4 1 5",  # read as 6 1 1.5 3 72.4 1 5: the class is not an integer
    ]) + "\n")
    res = check(work, "NET-001")
    assert res.outcome == Outcome.FAIL
    assert all(f.level == "L0" for f in errors(res))
    by_segment = {f.where["segment"]: f.message for f in errors(res) if "segment" in f.where}
    assert set(by_segment) == {2, 3, 4}
    assert "order 0" in by_segment[2]
    assert "slope -0.01" in by_segment[3] and "length 0" in by_segment[4]
    table = [f for f in errors(res) if f.where == {"line": 6}]
    assert any("type mismatch" in f.message for f in table)
    assert any("class required but not read" in f.message for f in table)
    assert res.metrics["rejected"] == 3 and res.metrics["table_errors"] == len(table) == 2
    # the trailing 5 fits no optional field: DHSVM prints a message and drops it
    [dropped] = warnings(res)
    assert dropped.level == "L1" and "unable to match field" in dropped.message


def test_net001_dropped_field_alone_does_not_reject_the_network(work):
    text = network_text(DOWN, ORDER).splitlines()
    text[1] += " 42"  # a seventh number on segment 1's line
    write(work, "stream.network.dat", "\n".join(text) + "\n")
    res = check(work, "NET-001")
    assert res.outcome == Outcome.FAIL and not errors(res)
    [w] = warnings(res)
    assert w.where == {"line": 2} and "int 42" in w.message
    assert res.metrics["table_errors"] == 0 and res.metrics["dropped_fields"] == 1


def test_net001_ids_must_be_unique_positive_and_fit_16_bits(work):
    write(work, "stream.network.dat", "\n".join([
        "0 1 0.05 120.0 1 3",  # DHSVM: 'channel id invalid'
        "2 1 0.05 170.0 1 3",
        "3 2 0.05 150.0 1 5",
        "2 1 0.05 170.0 1 5",  # the ID 2 again: links and map cells go to the first
        "5 3 0.05 210.0 1 0",
        "70000 1 0.05 72.4 1 5",  # stored as 70000 % 65536 = 4464
    ]) + "\n")
    res = check(work, "NET-001")
    assert [f.where["segment"] for f in errors(res, "L0")] == [0]
    silent = {f.where["segment"]: f for f in errors(res, "L1")}
    assert set(silent) == {2, 70000}
    assert silent[70000].values == {"read_as": 4464}
    assert silent[2].where["lines"] == [2, 4]


def test_net001_empty_network(work):
    write(work, "stream.network.dat", "# nothing but a comment\n")
    res = check(work, "NET-001")
    assert [f.level for f in errors(res)] == ["L0"]
    assert "no segment record" in res.findings[0].message
    assert check(work, "NET-003").outcome == Outcome.SKIP


# ------------------------------------------------------------------ NET-003
def test_net003_cycle_and_the_segments_it_traps(work):
    down = dict(DOWN)
    down[3] = 1  # 1 -> 3 -> 1; segment 2 drains into the cycle
    write(work, "stream.network.dat", network_text(down, ORDER))
    res = check(work, "NET-003")
    [f] = errors(res)
    assert f.level == "L1" and f.where["segments"] == [1, 3]
    assert f.values == {"segments_draining_into_cycle": 3}
    assert "1 -> 3 -> 1" in f.message
    assert res.metrics["segments_not_reaching_an_outlet"] == 3 and res.metrics["outlets"] == 1


def test_net003_segment_draining_into_itself(work):
    down = dict(DOWN)
    down[6] = 6
    write(work, "stream.network.dat", network_text(down, ORDER))
    [f] = errors(check(work, "NET-003"))
    assert f.where["segments"] == [6] and "6 -> 6" in f.message


def test_net003_outlet_ids_dhsvm_cannot_use(work):
    text = network_text(DOWN, ORDER).splitlines()
    text[1] = text[1].replace("    3", "65539")  # segment 1: DHSVM reads 65539 as 3
    text[2] = text[2].replace("    3", "   77")  # segment 2: names no segment
    write(work, "stream.network.dat", "\n".join(text) + "\n")
    res = check(work, "NET-003")
    levels = {f.where["segment"]: (f.level, f.values) for f in errors(res)}
    assert levels == {1: ("L1", {"written": 65539, "read_as": 3}), 2: ("L0", {"outlet": 77})}
    assert res.metrics["outlets"] == 1  # only segment 5 is written as an outlet


def test_net003_leaves_negative_outlet_ids_to_net013(tmp_path):
    make_synthetic_case(tmp_path, outlet_text="-9")
    assert check(tmp_path, "NET-003").outcome == Outcome.PASS
    assert check(tmp_path, "NET-013").outcome == Outcome.FAIL


# ------------------------------------------------------------------ NET-007
def record(col, row, seg, length=30.0, cut_h=0.5, cut_w=2.0):
    return f"{col} {row} {seg} {length} {cut_h} {cut_w} 180.0\n"


def test_net007_records_dhsvm_rejects(work):
    n = len(map_text().splitlines())  # a header line and 26 records
    write(work, "stream.map.dat", map_text()
          + record(NCOLS, 3, 1)  # column 12 does not exist (columns are 0..11)
          + record(5, NROWS, 1)  # row 16 does not exist
          + record(6, 9, 9)  # no segment 9
          + record(6, 9, 5, length=-1.0)
          + record(6, 9, 5, cut_h=-0.5)
          + record(6, 9, 5, cut_w=-2.0)
          + "6 9\n")  # four required fields are missing
    res = check(work, "NET-007")
    found = {f.where["line"] - n: f.message for f in errors(res)}
    assert all(f.level == "L0" for f in errors(res))
    assert "column 12, row 3 is outside the grid (columns 0..11, rows 0..15)" in found[1]
    assert "outside the grid" in found[2]
    assert "segment 9 is not in the stream network" in found[3]
    assert "length -1 is negative" in found[4]
    assert "cut height -0.5 is negative" in found[5]
    assert "cut width -2 is negative" in found[6]
    assert "required but not read" in found[7]
    assert res.metrics["outside_grid"] == 2 and res.metrics["table_errors"] == 4


def test_net007_record_outside_the_basin_mask(work):
    assert not synthetic_mask()[0, 0]
    write(work, "stream.map.dat", map_text() + record(0, 0, 2))
    res = check(work, "NET-007")
    assert res.outcome == Outcome.FAIL and not errors(res)
    [w] = warnings(res)
    assert w.level == "L1" and w.where == {"line": 28, "segment": 2, "row": 0, "column": 0}
    assert "never uses this record" in w.message

    # with a negative soil depth in that cell (a nodata value) DHSVM stops instead
    depth = np.full((NROWS, NCOLS), SOIL_DEPTH, dtype="<f4")
    depth[~synthetic_mask().astype(bool)] = -9999.0
    depth.tofile(work / "input" / "soild.bin")
    res = check(work, "NET-007")
    [e] = errors(res)
    assert e.level == "L0" and "bad cut_depth" in e.message and "-9999" in e.message
    assert "outside the basin mask" in e.message and not warnings(res)


def test_net007_zero_channel_length_in_a_cell(work):
    lines = map_text().splitlines()
    first = lines[1]  # segment 1 in cell (row 0, column 6), alone in its cell
    lines[1] = first.replace("  30.000", "   0.000")
    write(work, "stream.map.dat", "\n".join(lines) + "\n")
    [e] = errors(check(work, "NET-007"))
    assert e.level == "L1" and e.where == {"row": 0, "column": 6, "lines": [2]}
    assert "NaN" in e.message
    # a second record with a positive length in the same cell makes the total non-zero
    write(work, "stream.map.dat", "\n".join(lines) + "\n" + record(6, 0, 3))
    assert check(work, "NET-007").outcome == Outcome.PASS


def test_net007_segment_without_a_record_in_the_basin(work):
    lines = [ln for ln in map_text().splitlines() if ln.split()[2] != "6"]
    write(work, "stream.map.dat", "\n".join(lines) + "\n")
    res = check(work, "NET-007")
    [w] = warnings(res)
    assert w.where == {"segments": [6]} and w.values == {"count": 1}
    assert res.metrics["segments_without_in_basin_record"] == 1


def test_net007_empty_stream_map(work):
    write(work, "stream.map.dat", "# col row id length cut_h cut_w\n")
    res = check(work, "NET-007")
    [w] = warnings(res)  # DHSVM reads the empty map without an error
    assert w.values == {"count": 6} and res.metrics["records"] == 0


def test_net007_many_bad_records_are_listed_up_to_a_cap(work):
    write(work, "stream.map.dat", map_text() + "".join(record(NCOLS + i, 0, 1)
                                                      for i in range(50)))
    res = check(work, "NET-007")
    assert len(errors(res)) == 21 and res.metrics["rejected"] == 50
    assert "... and 30 more records that DHSVM rejects" in errors(res)[-1].message


def test_net007_without_a_readable_soil_depth_map(work):
    (work / "input" / "soild.bin").unlink()
    res = check(work, "NET-007")
    assert res.outcome == Outcome.PASS
    assert any("soil depth not checked" in f.message for f in res.findings)


# ------------------------------------------------------------------ NET-010
def test_net010_class_values_dhsvm_rejects(work):
    write(work, "stream.class.dat", "\n".join([
        "1 2.0 0.5 0.04 0.0",
        "2 0.0 0.5 0.04",  # width 0: DHSVM aborts, although it never uses the width
        "3 2.0 0.5 0",  # Manning's n 0
        "4 wide 0.5 0.04",  # a word where a number must be
    ]) + "\n")
    res = check(work, "NET-010")
    found = {f.where.get("class", f.where.get("line")): f for f in errors(res)}
    assert "width 0 is not above 0" in found[2].message
    assert "Manning's n 0 is not above 0" in found[3].message
    assert all(f.level == "L0" for f in errors(res))
    assert any("type mismatch" in f.message for f in errors(res) if f.where == {"line": 4})


def test_net010_class_ids(work):
    write(work, "stream.class.dat", "\n".join([
        "1 2.0 0.5 0.04",
        "0 2.0 0.5 0.04",  # DHSVM prints 'class id invalid' and keeps the class
        "1 9.0 9.5 0.09",  # a second class 1: DHSVM uses the first
        "70000 2.0 0.5 0.04",
    ]) + "\n")
    res = check(work, "NET-010")
    [wrap] = errors(res)
    assert wrap.level == "L1" and wrap.where["class"] == 70000
    [dup] = warnings(res)
    assert dup.where == {"class": 1, "lines": [1, 3]}
    assert any(f.severity == Severity.INFO and "class id invalid" in f.message
               for f in res.findings)
    assert res.metrics["bank_height_m"] == [0.5, 0.5]  # the first definition of class 1


def test_net010_implausible_values_in_classes_that_are_used(work):
    write(work, "stream.class.dat", "\n".join([
        "1 0.5 0.10 0.045",  # PNNL template placeholders, used by every segment
        "2 0.5 0.03 0.9",  # not used by any segment: not reported
    ]) + "\n")
    res = check(work, "NET-010")
    [w] = warnings(res)
    assert w.level == "L2" and w.where == {"class": 1, "line": 1}
    assert w.values == {"bank_height_m": 0.1, "segments": 6}
    assert "0.03-0.10 m as placeholders" in w.message
    assert check(work, "NET-010", min_bank_height_m="0.05").outcome == Outcome.PASS

    write(work, "stream.class.dat", "1 2.0 45.0 0.9\n")  # centimetres? and n far too high
    res = check(work, "NET-010")
    assert sorted(w.message.split(": ")[1][:11] for w in warnings(res)) == ["Manning's n",
                                                                           "bank height"]
    assert not errors(res)


def test_net010_empty_class_file(work):
    write(work, "stream.class.dat", "# ID W D n\n")
    res = check(work, "NET-010")
    assert any("defines no class" in f.message for f in errors(res, "L0"))
    assert any("class 1 is used by 6 segment(s)" in f.message for f in errors(res, "L0"))


# --------------------------------------------------- when there is no network
@pytest.mark.parametrize(("old", "new", "reason"), [
    ("Flow Routing         = NETWORK", "Flow Routing         = UNIT_HYDROGRAPH",
     "Flow Routing = UNIT_HYDROGRAPH"),
    ("Extent               = BASIN", "Extent               = POINT", "Extent = POINT"),
    ("Stream Class File    = input/stream.class.dat", "Stream Class File    = none",
     "Stream Class File = none"),
])
def test_network_rules_do_not_apply_when_dhsvm_reads_no_network(work, old, new, reason):
    assert old in INPUT_TEMPLATE
    (work / "INPUT.synthetic").write_text(INPUT_TEMPLATE.replace(old, new))
    report = dhsvm.check(work / "INPUT.synthetic", work)
    got = {r.rule.id: r for r in report.results}
    for rid in NET_RULES:
        assert got[rid].outcome == Outcome.SKIP, rid
        assert reason in got[rid].summary
    assert got["DHSVM-MAP-001"].outcome == Outcome.PASS
    assert report.exit_code() == 0 and "stream network" not in report.context


def test_missing_routing_option_is_not_evaluated(work):
    text = INPUT_TEMPLATE.replace("Flow Routing         = NETWORK\n", "")
    (work / "INPUT.synthetic").write_text(text)
    res = check(work, "NET-001")
    assert res.outcome == Outcome.ERROR and "Flow Routing" in res.summary


def test_findings_carry_their_level_in_the_json_report(work):
    write(work, "stream.network.dat", network_text(DOWN, ORDER).replace("0.05000", "0.00000", 1))
    report = dhsvm.check(work / "INPUT.synthetic", work, rules=["NET-001", "NET-004"])
    net001, net004 = report.to_dict()["results"]
    assert net001["rule"]["level"] == "L0/L1"
    assert [f["level"] for f in net001["findings"]] == ["L0"]
    assert net004["rule"]["level"] == "L1"
