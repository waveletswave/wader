"""RUN-002 on the synthetic case: where DHSVM writes its output, and whether it can."""

from __future__ import annotations

import os

import pytest
from conftest import INPUT_TEMPLATE, make_synthetic_case

from wader.core import Outcome, Severity
from wader.models import dhsvm
from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.mutate import mutate_case

ENTRY = "Output Directory     = output/"


@pytest.fixture
def work(tmp_path):
    make_synthetic_case(tmp_path)
    return tmp_path


def check(work):
    return dhsvm.check(work / "INPUT.synthetic", work, rules=["RUN-002"]).results[0]


def set_output(work, value):
    assert ENTRY in INPUT_TEMPLATE
    new = f"Output Directory     = {value}" if value is not None else ""
    (work / "INPUT.synthetic").write_text(INPUT_TEMPLATE.replace(ENTRY, new))


def test_clean_case(work):
    res = check(work)
    assert res.outcome == Outcome.PASS and not res.findings
    assert res.summary == f"the output directory {work / 'output'} exists"
    assert res.metrics == {"value": "output/", "directory": str(work / "output"),
                           "file_name_prefix": ""}


def test_a_directory_that_does_not_exist(work):
    (work / "output").rmdir()
    res = check(work)
    assert res.outcome == Outcome.FAIL
    (f,) = res.findings
    assert f.severity == Severity.ERROR and f.level == "L0"
    assert "does not exist" in f.message and "Create it before the run" in f.message
    assert "'Cannot open file: output/Aggregated.Values', exit 3" in f.message
    (work / "output").write_text("a file, not a directory")
    assert "is not a directory" in check(work).findings[0].message


def test_an_absolute_path(work, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    set_output(work, f"{elsewhere}/")
    assert check(work).findings[0].where["directory"] == str(elsewhere)
    elsewhere.mkdir()
    assert check(work).outcome == Outcome.PASS


def test_without_a_final_slash_the_name_is_a_file_name_prefix(work):
    set_output(work, "output")
    res = check(work)
    assert res.outcome == Outcome.FAIL  # a warning: DHSVM runs, the files are elsewhere
    (f,) = res.findings
    assert f.severity == Severity.WARNING and f.level == "L1"
    assert f"writes outputAggregated.Values, outputMass.Balance and so on into {work}" \
        in f.message
    assert f"{work / 'output'} is a directory, but nothing is written into it" in f.message
    assert res.metrics["file_name_prefix"] == "output" and res.metrics["directory"] == str(work)
    # the directory is what precedes the last '/'
    set_output(work, "results/run1")
    res = check(work)
    assert [f.level for f in res.findings] == ["L0", "L1"]
    assert res.findings[0].where["directory"] == str(work / "results")
    (work / "results").mkdir()
    assert [f.level for f in check(work).findings] == ["L1"]


@pytest.mark.parametrize("value", ["", None])
def test_an_empty_or_missing_entry(work, value):
    set_output(work, value)
    res = check(work)
    assert res.outcome == Outcome.FAIL and res.summary == "Output Directory is not set"
    assert res.findings[0].level == "L0" and "exit 51" in res.findings[0].message


def test_a_directory_that_cannot_be_written(work, monkeypatch):
    # (a test running as root can write anywhere, so the answer is put in os.access's mouth)
    monkeypatch.setattr(os, "access", lambda path, mode: False)
    (f,) = check(work).findings
    assert f.level == "L0" and "no files can be created" in f.message


@pytest.mark.parametrize("mutation, severity, level", [
    ("missing-output-dir", Severity.ERROR, "L0"),
    ("output-no-slash", Severity.WARNING, "L1"),
])
def test_mutations_are_caught(work, tmp_path, mutation, severity, level):
    case = DhsvmCase(work / "INPUT.synthetic", work)
    mutated, _ = mutate_case(case, mutation, tmp_path / "mut")
    res = dhsvm.check(mutated, work, rules=["RUN-002"]).results[0]
    assert res.outcome == Outcome.FAIL
    assert [(f.severity, f.level) for f in res.findings] == [(severity, level)]
