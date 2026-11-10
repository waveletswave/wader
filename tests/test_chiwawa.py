"""Conformance on the PNNL Chiwawa reference case (needs cases/chiwawa/fetch.py)."""

from __future__ import annotations

import pytest

from wader.core import Outcome, Severity
from wader.models import dhsvm
from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.io import rewrite_network
from wader.models.dhsvm.mutate import MUTATIONS, mutate_case

pytestmark = pytest.mark.case_data
TOPOLOGY = ("DHSVM-NET-004", "DHSVM-NET-005", "DHSVM-NET-006")
FILES = ("DHSVM-MAP-001", "DHSVM-NET-001", "DHSVM-NET-003", "DHSVM-NET-007", "DHSVM-NET-010")


def by_id(report):
    return {r.rule.id: r for r in report.results}


def test_original_network_passes_topology_rules(chiwawa_input, chiwawa_workdir):
    report = dhsvm.check(chiwawa_input, chiwawa_workdir)
    res = by_id(report)
    for rid in TOPOLOGY:
        assert res[rid].outcome == Outcome.PASS, report.to_text()
    assert res["DHSVM-NET-004"].metrics["links"] == 493
    [outlet] = res["DHSVM-NET-006"].metrics["outlets"]
    assert outlet["segment"] == 494 and outlet["distance_cells"] == 0.0
    assert outlet["direct_inflows"] == 2
    assert report.exit_code() == 0  # the NET-013 finding below is a warning


def test_original_case_passes_the_file_rules(chiwawa_input, chiwawa_workdir):
    res = by_id(dhsvm.check(chiwawa_input, chiwawa_workdir))
    for rid in FILES:
        assert res[rid].outcome == Outcome.PASS and not res[rid].findings, res[rid].summary
    assert res["DHSVM-NET-001"].metrics["segments"] == 494
    assert res["DHSVM-NET-003"].metrics["outlets"] == 1
    assert res["DHSVM-NET-007"].metrics["records"] == 4566
    classes = res["DHSVM-NET-010"].metrics
    assert (classes["classes"], classes["classes_used"]) == (18, 14)
    assert classes["bank_height_m"] == [0.4, 2.508] and classes["manning_n"] == [0.03, 0.0506]
    # DEM, mask, soil type, soil depth, vegetation type, sky view, and one shading file
    # for each of the 12 months of the run (8 three-hourly data sets of 1 byte per cell)
    sizes = {f["map"]: f["bytes"] for f in res["DHSVM-MAP-001"].metrics["files"]}
    assert len(sizes) == 18 and sizes["DEM"] == 425 * 300 * 4
    assert sizes["shading, month 01"] == 425 * 300 * 8


def test_canopy_gap_map_is_too_short_once_dhsvm_reads_it(chiwawa_input, chiwawa_workdir,
                                                        tmp_path):
    # The shipped INPUT has Canopy Gapping = FALSE, so CanopyGap.bin (1 byte per cell) is
    # never read. DHSVM reads that map as 4-byte floats (VarID.c:67-69).
    case = DhsvmCase(chiwawa_input, chiwawa_workdir)
    case.config.with_value("OPTIONS", "Canopy Gapping", "TRUE").write(tmp_path / "INPUT")
    res = dhsvm.check(tmp_path / "INPUT", chiwawa_workdir, rules=["MAP-001"]).results[0]
    [f] = res.findings
    assert f.where["map"] == "canopy gap" and f.level == "L0"
    assert f.values == {"bytes": 127500, "expected_bytes": 510000}


def test_original_outlet_written_as_minus_one_is_reported(chiwawa_input, chiwawa_workdir):
    res = dhsvm.check(chiwawa_input, chiwawa_workdir, rules=["NET-013"]).results[0]
    assert res.outcome == Outcome.FAIL
    [f] = [f for f in res.findings if f.severity == Severity.WARNING]
    assert f.where["segment"] == 494
    assert f.values == {"written": -1, "read_as": 1, "dropped": True}


def test_outlet_written_as_zero_passes_net013(chiwawa_input, chiwawa_workdir, tmp_path):
    case = DhsvmCase(chiwawa_input, chiwawa_workdir)
    net = case.network
    fixed = tmp_path / "stream.network.dat"
    fixed.write_text(rewrite_network(net, net.intended_downstream(),
                                     {s.id: s.order for s in net.segments},
                                     outlet_sentinel="0"))
    cfg = case.config.with_value("ROUTING", "Stream Network File", str(fixed))
    cfg.write(tmp_path / "INPUT.outlet0")
    report = dhsvm.check(tmp_path / "INPUT.outlet0", chiwawa_workdir)
    assert {r.outcome for r in report.results} == {Outcome.PASS}, report.to_text()


@pytest.mark.parametrize("name", [n for n in sorted(MUTATIONS) if n != "negative-outlet"])
def test_mutations_are_caught(chiwawa_input, chiwawa_workdir, tmp_path, name):
    case = DhsvmCase(chiwawa_input, chiwawa_workdir)
    mutated, m = mutate_case(case, name, tmp_path / "mut")
    res = by_id(dhsvm.check(mutated, chiwawa_workdir))
    for rid in m.info.targets:
        assert res[rid].outcome == Outcome.FAIL, (rid, res[rid].summary)
    if name == "reverse-network":
        assert res["DHSVM-NET-005"].outcome == Outcome.PASS
        assert res["DHSVM-NET-004"].metrics["uphill"] >= 40
        [outlet] = res["DHSVM-NET-006"].metrics["outlets"]
        assert outlet["distance_cells"] > 400
    if name == "jump":
        assert res["DHSVM-NET-004"].outcome == Outcome.PASS
        assert res["DHSVM-NET-006"].outcome == Outcome.PASS
