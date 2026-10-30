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
