"""Stream-network rules on the synthetic case: no false alarms, every mutation caught."""

from __future__ import annotations

import re

import pytest
from conftest import DOWN, ORDER, REPO, make_synthetic_case

from wader.core import Outcome, Severity, Status, registered_rules
from wader.models import dhsvm
from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.mutate import MUTATIONS, mutate_case

RULES = registered_rules("dhsvm")
# the synthetic network has only two headwaters that do not already drain into the outlet
SYNTHETIC_PARAMS = {"force-wire": {"count": 2}}


def outcomes(report) -> dict[str, Outcome]:
    return {r.rule.id: r.outcome for r in report.results}


def test_clean_case_passes_every_rule(synthetic):
    path, work = synthetic
    report = dhsvm.check(path, work)
    assert set(outcomes(report).values()) == {Outcome.PASS}, report.to_text()
    assert report.exit_code() == 0


@pytest.mark.parametrize(
    ("rule_id", "mutation"),
    [(r.info.id, m) for r in RULES for m in r.info.mutations],
)
def test_rule_catches_its_mutations(synthetic, tmp_path, rule_id, mutation):
    path, work = synthetic
    case = DhsvmCase(path, work)
    mutated_input, _ = mutate_case(case, mutation, tmp_path / "mut",
                                   SYNTHETIC_PARAMS.get(mutation))
    report = dhsvm.check(mutated_input, work, rules=[rule_id])
    assert outcomes(report)[rule_id] == Outcome.FAIL, report.to_text()


def test_every_rule_is_documented():
    assert RULES, "no rules registered"
    for r in RULES:
        info = r.info
        assert info.evidence, f"{info.id} has no evidence"
        assert all(e.source and isinstance(e.status, Status) for e in info.evidence)
        assert info.mutations, f"{info.id} has no mutation test (CLAUDE.md rule 2)"
        assert set(info.mutations) <= set(MUTATIONS), info.id
    for name, (minfo, _) in MUTATIONS.items():
        assert minfo.targets, name


def test_cited_tests_exist():
    """Evidence that cites an experiment as ``tests/file.py::test_name`` must name a test
    that exists, so a ``verified-experiment`` status can always be reproduced."""
    cited = 0
    for r in RULES:
        for e in r.info.evidence:
            file = None
            for path, name in re.findall(r"(tests/\w+\.py)|::(test_\w+)", e.source):
                if path:
                    file = REPO / path
                    assert file.is_file(), f"{r.info.id}: {path} does not exist"
                else:
                    assert file is not None, f"{r.info.id}: {name} cited without a file"
                    assert f"def {name}(" in file.read_text(), f"{r.info.id}: no {name}"
                    cited += 1
            if e.status == Status.VERIFIED_EXPERIMENT:
                assert e.source.strip(), r.info.id
    assert cited >= 15


def test_reverse_network_findings(synthetic, tmp_path):
    path, work = synthetic
    mutated, m = mutate_case(DhsvmCase(path, work), "reverse-network", tmp_path / "mut")
    assert m.downstream == {1: 3, 2: None, 3: 2, 4: 5, 5: 3, 6: 5}
    report = dhsvm.check(mutated, work)
    res = {r.rule.id: r for r in report.results}
    rises = sorted(f.values["rise_m"] for f in res["DHSVM-NET-004"].findings)
    assert rises == [23.0, 28.0]  # E -> C and C -> B now climb
    assert outcomes(report)["DHSVM-NET-005"] == Outcome.PASS  # no jumps were introduced
    [far] = [f for f in res["DHSVM-NET-006"].findings if f.severity == Severity.ERROR]
    assert far.values["distance_cells"] == pytest.approx(12.04, abs=0.01)
    assert report.exit_code() == 1


def test_jump_is_isolated_to_net005(synthetic, tmp_path):
    path, work = synthetic
    mutated, m = mutate_case(DhsvmCase(path, work), "jump", tmp_path / "mut")
    assert m.downstream[2] == 5
    got = outcomes(dhsvm.check(mutated, work))
    assert got["DHSVM-NET-005"] == Outcome.FAIL
    assert got["DHSVM-NET-004"] == Outcome.PASS and got["DHSVM-NET-006"] == Outcome.PASS


def test_tolerance_parameter(synthetic, tmp_path):
    path, work = synthetic
    mutated, _ = mutate_case(DhsvmCase(path, work), "reverse-network", tmp_path / "mut")
    loose = dhsvm.check(mutated, work, rules=["NET-004"], params={"tolerance_m": "50"})
    assert outcomes(loose)["DHSVM-NET-004"] == Outcome.PASS


def test_negative_outlet_naming_no_segment_is_an_error(tmp_path):
    down = dict(DOWN)
    path = make_synthetic_case(tmp_path, down, ORDER, outlet_text="-9")
    res = dhsvm.check(path, tmp_path, rules=["NET-013"]).results[0]
    assert res.outcome == Outcome.FAIL
    assert res.findings[0].severity == Severity.ERROR and res.findings[0].values["read_as"] == 9


def test_network_without_an_outlet(tmp_path):
    down = dict(DOWN)
    down[5] = 1  # the outlet drains into a headwater: every segment has a downstream
    path = make_synthetic_case(tmp_path, down, ORDER)
    res = {r.rule.id: r for r in dhsvm.check(path, tmp_path).results}
    # NET-003 owns this error; the outlet-position rule has nothing to evaluate
    assert res["DHSVM-NET-003"].outcome == Outcome.FAIL
    assert any("no segment leaves the network" in f.message
               for f in res["DHSVM-NET-003"].findings)
    assert res["DHSVM-NET-006"].outcome == Outcome.SKIP
    assert "NET-003" in res["DHSVM-NET-006"].summary


def test_missing_file_is_not_evaluated(synthetic):
    path, work = synthetic
    (work / "input" / "dem.bin").unlink()
    report = dhsvm.check(path, work, rules=["NET-004"])
    assert report.results[0].outcome == Outcome.ERROR
    assert report.exit_code() == 2
