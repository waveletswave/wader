"""Experiments with DHSVM itself: what it accepts and how it reads the outlet field.

Each run covers five days (10/01/1970-00 to 10/06/1970-00) of the Chiwawa case.
These tests are the experimental evidence cited by DHSVM-NET-004/005/006/013.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import REPO

from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.io import rewrite_network
from wader.models.dhsvm.mutate import mutate_case

pytestmark = [pytest.mark.dhsvm, pytest.mark.case_data]
RUN = REPO / "cases" / "chiwawa" / "run.py"
END = "10/06/1970-00"


def run_dhsvm(input_path: Path, name: str, dhsvm_bin: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(RUN), "--input", str(input_path), "--name", name, "--end", END,
         "--dhsvm", str(dhsvm_bin)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    work = Path(proc.stdout.strip().splitlines()[-1])
    summary = json.loads((work / "summary.json").read_text())
    log = (work / "dhsvm.stdout.log").read_text(errors="replace").lower()
    summary["warning_lines"] = sum("warning" in line for line in log.splitlines())
    summary["streamflow"] = (work / "output" / "Streamflow.Only").read_bytes()
    return summary


@pytest.fixture(scope="module")
def baseline(chiwawa_input, dhsvm_bin):
    return run_dhsvm(chiwawa_input, "pytest-baseline", dhsvm_bin)


@pytest.fixture(scope="module")
def outlet_zero(chiwawa_input, chiwawa_workdir, dhsvm_bin, tmp_path_factory):
    """The Chiwawa setup with its outlet written as 0 instead of -1."""
    tmp = tmp_path_factory.mktemp("outlet0")
    case = DhsvmCase(chiwawa_input, chiwawa_workdir)
    net = case.network
    fixed = tmp / "stream.network.dat"
    fixed.write_text(rewrite_network(net, net.intended_downstream(),
                                     {s.id: s.order for s in net.segments},
                                     outlet_sentinel="0"))
    case.config.with_value("ROUTING", "Stream Network File", str(fixed)).write(tmp / "INPUT")
    return run_dhsvm(tmp / "INPUT", "pytest-outlet0", dhsvm_bin)


def test_negative_outlet_is_read_as_segment_1(baseline, outlet_zero):
    # PNNL writes the outlet as -1: DHSVM reads 1, so no water leaves the network
    assert baseline["totals_outflow_m3"] == 0.0
    assert outlet_zero["totals_outflow_m3"] > 1e6
    # ... while the saved outlet hydrograph and the mass balance are unaffected
    assert baseline["streamflow"] == outlet_zero["streamflow"]
    assert baseline["mass_error_mm"] == outlet_zero["mass_error_mm"]


@pytest.mark.parametrize("mutation", ["reverse-network", "jump", "force-wire"])
def test_dhsvm_runs_mutants_without_complaint(mutation, baseline, outlet_zero, chiwawa_input,
                                              chiwawa_workdir, dhsvm_bin, tmp_path):
    mutated, _ = mutate_case(DhsvmCase(chiwawa_input, chiwawa_workdir), mutation,
                             tmp_path / "mut")
    got = run_dhsvm(mutated, f"pytest-{mutation}", dhsvm_bin)
    assert got["exit_code"] == 0 and not got["problems"]
    assert got["warning_lines"] == baseline["warning_lines"]
    assert got["mass_error_mm"] == baseline["mass_error_mm"]
    if mutation == "reverse-network":
        # the true outlet (segment 494, saved as "Chiwawa") loses almost all its water ...
        q0 = baseline["mean_flow_m3_per_step"]["Chiwawa"]
        assert got["mean_flow_m3_per_step"]["Chiwawa"] < 0.05 * q0
        # ... yet the network's total outflow hardly changes
        assert got["totals_outflow_m3"] == pytest.approx(outlet_zero["totals_outflow_m3"],
                                                         rel=0.02)
