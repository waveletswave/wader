"""Experiments with DHSVM itself: what it rejects, what it accepts, and what it does
with what it accepts.

Every run uses the Chiwawa case and differs from it in one thing. The tests are the
experimental evidence (status ``verified-experiment``) that the rules cite. The first
group runs five days (to 10/06/1970-00) and backs DHSVM-NET-004/005/006/013; the second
runs two days (to 10/03/1970-00), or stops at start-up, and backs DHSVM-NET-001/003/007/
010 and DHSVM-MAP-001; the third does the same for DHSVM-MAP-003, DHSVM-STA-001 and
DHSVM-RUN-002. The last tests record upstream behaviour that no rule uses yet.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO

from wader.cache import chiwawa_dir
from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.io import replace_token, rewrite_network
from wader.models.dhsvm.maps import state_files
from wader.models.dhsvm.mutate import MUTATIONS, mutate_case

pytestmark = [pytest.mark.dhsvm, pytest.mark.case_data]
RUN = REPO / "cases" / "chiwawa" / "run.py"
END = "10/06/1970-00"
SHORT_END = "10/03/1970-00"


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


# ====================================================================================
# Two-day runs: what DHSVM does with the inputs that NET-001/003/007/010 and MAP-001
# report. ``run_any`` does not require the run to succeed.
# ====================================================================================
NETWORK = ("ROUTING", "Stream Network File")
STREAM_MAP = ("ROUTING", "Stream Map File")
CLASSES = ("ROUTING", "Stream Class File")
SOIL_DEPTH = ("SOILS", "Soil Depth File")
READ_ERROR = "Error reading input in file"  # ReportError 5 (ReportError.c:28)


def run_any(input_path: Path, name: str, dhsvm_bin: Path, end: str = SHORT_END,
            keep_output: bool = False) -> dict:
    """Run DHSVM whether it succeeds or not: run.py's summary, the log, the output files.
    With ``keep_output`` the INPUT's own Output Directory is used."""
    subprocess.run(
        [sys.executable, str(RUN), "--input", str(input_path), "--name", name, "--end", end,
         "--dhsvm", str(dhsvm_bin), *(["--keep-output"] if keep_output else [])],
        capture_output=True, text=True,
    )
    work = chiwawa_dir() / "run" / name
    summary = json.loads((work / "summary.json").read_text())
    summary["log"] = ((work / "dhsvm.stdout.log").read_text(errors="replace")
                      + (work / "dhsvm.stderr.log").read_text(errors="replace"))
    summary["warning_lines"] = sum("warning" in ln for ln in summary["log"].lower().splitlines())
    summary["outputs"] = {p.name: p.read_bytes()
                          for p in sorted((work / "output").iterdir()) if p.is_file()}
    summary["beside_output"] = {p.name: p.read_bytes() for p in sorted(work.iterdir())
                                if p.is_file() and p.name.startswith("output")}
    return summary


def variant(case: DhsvmCase, out_dir: Path, files: dict | None = None,
            values: dict | None = None) -> Path:
    """An INPUT that differs from the case in the given files and/or entry values."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = case.config
    for (section, key), content in (files or {}).items():
        path = out_dir / case.path(section, key).name
        path.write_bytes(content if isinstance(content, bytes) else content.encode("latin-1"))
        cfg = cfg.with_value(section, key, str(path))
    for (section, key), value in (values or {}).items():
        cfg = cfg.with_value(section, key, value)
    return cfg.write(out_dir / "INPUT")


def edit_line(lines: list[str], lineno: int, index: int, text: str) -> str:
    lines = list(lines)
    lines[lineno - 1] = replace_token(lines[lineno - 1], index, text)
    return "\n".join(lines)


@pytest.fixture(scope="module")
def case(chiwawa_input, chiwawa_workdir) -> DhsvmCase:
    return DhsvmCase(chiwawa_input, chiwawa_workdir)


@pytest.fixture(scope="module")
def short_baseline(chiwawa_input, dhsvm_bin):
    got = run_any(chiwawa_input, "pytest-short", dhsvm_bin)
    assert got["exit_code"] == 0 and not got["problems"]
    return got


# mutation -> (exit status, message) when DHSVM stops at start-up
STOPS = {
    "zero-slope": (5, "segment 1: channel slope (0.000000) invalid"),
    "dangling-outlet": (5, "cannot find outlet (495) for segment 1"),
    "map-swap-row-col": (5, "bad coordinates"),
    "missing-class": (5, "segment 1: channel class 19 not found"),
    "truncate-map": (2, "Error while reading file"),
    "shallow-soil": (255, "Soil depth is 0.400000, Root depth is 0.400000"),
    "truncate-state": (2, "Error while reading file"),
    "missing-output-dir": (3, "Cannot open file: no-such-directory/Aggregated.Values"),
}
KEEP_OUTPUT = {"missing-output-dir"}  # mutants of the Output Directory entry itself


@pytest.mark.parametrize("mutation", sorted(STOPS))
def test_dhsvm_stops_on_mutant(mutation, case, dhsvm_bin, short_tmp):
    status, message = STOPS[mutation]
    mutated, _ = mutate_case(case, mutation, short_tmp / "mut")
    got = run_any(mutated, f"pytest-{mutation}", dhsvm_bin,
                  keep_output=mutation in KEEP_OUTPUT)
    assert got["exit_code"] == status, got["log"][-2000:]
    assert message in got["log"]
    assert (READ_ERROR in got["log"]) == (status == 5)
    assert "END OF MODEL RUN" not in got["log"] and not got["outputs"].get("Streamflow.Only")


@pytest.mark.parametrize(("index", "text", "message"), [
    (0, "0", "segment 0: channel id invalid"),
    (1, "0", "segment 1: channel order (0) invalid"),
    (3, "0.00000", "segment 1: channel length (0.000000) invalid"),
])
def test_dhsvm_rejects_segment_field(index, text, message, case, dhsvm_bin, tmp_path):
    net = case.network
    new = edit_line(net.lines, net.by_id()[1].lineno, index, text)
    got = run_any(variant(case, tmp_path / "v", {NETWORK: new}), f"pytest-field{index}",
                  dhsvm_bin)
    assert got["exit_code"] == 5 and message in got["log"] and READ_ERROR in got["log"]


@pytest.mark.parametrize(("index", "text", "message"), [
    (2, "999", "unable to locate segment 999"),
    (3, "-1.0", "bad length"),
    (4, "-0.5", "bad cut_depth"),
    (5, "-2.0", "bad cut_width"),
])
def test_dhsvm_rejects_map_field(index, text, message, case, dhsvm_bin, tmp_path):
    sm = case.stream_map
    new = edit_line(sm.lines, int(sm.lineno[0]), index, text)
    got = run_any(variant(case, tmp_path / "v", {STREAM_MAP: new}), f"pytest-mapfield{index}",
                  dhsvm_bin)
    assert got["exit_code"] == 5 and message in got["log"] and READ_ERROR in got["log"]


def test_dhsvm_stops_on_a_missing_map(case, dhsvm_bin, tmp_path):
    missing = tmp_path / "nowhere" / "dem.bin"
    got = run_any(variant(case, tmp_path / "v", values={("TERRAIN", "DEM File"): str(missing)}),
                  "pytest-nomap", dhsvm_bin)
    assert got["exit_code"] == 3 and f"Cannot open file: {missing}" in got["log"]


def test_unmatched_field_is_only_a_message(case, short_baseline, dhsvm_bin, tmp_path):
    net = case.network
    lines = list(net.lines)
    lines[net.by_id()[1].lineno - 1] += " 42"  # a seventh field: fits no optional field
    got = run_any(variant(case, tmp_path / "v", {NETWORK: "\n".join(lines)}), "pytest-extra",
                  dhsvm_bin)
    assert "line 1, field 6: unable to match field" in got["log"]
    assert got["exit_code"] == 0 and got["outputs"] == short_baseline["outputs"]


def test_dhsvm_runs_a_cycle_and_loses_the_water(case, short_baseline, dhsvm_bin, tmp_path):
    mutated, m = mutate_case(case, "cycle", tmp_path / "mut")
    assert m.downstream[492] == 470  # the main stem above the outlet now feeds its tributary
    got = run_any(mutated, "pytest-cycle", dhsvm_bin)
    assert got["exit_code"] == 0 and not got["problems"]
    assert got["warning_lines"] == short_baseline["warning_lines"]
    assert got["mass_error_mm"] == short_baseline["mass_error_mm"]
    q0 = short_baseline["mean_flow_m3_per_step"]["Chiwawa"]
    assert got["mean_flow_m3_per_step"]["Chiwawa"] < 0.01 * q0


def test_outlet_id_above_65535_wraps(case, short_baseline, dhsvm_bin, tmp_path):
    net = case.network
    seg = net.by_id()[1]
    assert seg.outlet == 3
    new = edit_line(net.lines, seg.lineno, 5, str(3 + 65536))
    got = run_any(variant(case, tmp_path / "v", {NETWORK: new}), "pytest-wrap", dhsvm_bin)
    assert got["exit_code"] == 0
    assert got["outputs"] == short_baseline["outputs"]  # 65539 is read as segment 3


def test_zero_length_map_record_gives_nan_streamflow(case, short_baseline, dhsvm_bin, tmp_path):
    mutated, _ = mutate_case(case, "zero-length-record", tmp_path / "mut")
    got = run_any(mutated, "pytest-zerolen", dhsvm_bin)
    assert got["exit_code"] == 0 and "END OF MODEL RUN" in got["log"]
    assert got["problems"] == ["Streamflow.Only contains NaN"]
    assert got["warning_lines"] == short_baseline["warning_lines"]
    assert got["mass_error_mm"] == short_baseline["mass_error_mm"]
    assert b"nan" in got["outputs"]["Streamflow.Only"].lower()


def test_map_record_outside_the_basin(case, short_baseline, dhsvm_bin, tmp_path):
    row, col = (int(v) for v in np.argwhere(~case.mask)[0])
    assert case.soil_depth[row, col] == -9999.0  # Chiwawa's soil depth outside the basin
    extra = f"{col:8d}{row:8d}{1:8d}{90.0:12.4f}{0.5:10.4f}{2.0:10.4f}{0.0:10.4f}\n"
    new_map = "\n".join(case.stream_map.lines).rstrip("\n") + "\n" + extra
    # with the negative soil depth of that cell DHSVM stops ...
    got = run_any(variant(case, tmp_path / "a", {STREAM_MAP: new_map}), "pytest-outside-a",
                  dhsvm_bin)
    assert got["exit_code"] == 5 and "bad cut_depth" in got["log"]
    # ... and with a soil depth of 1 m there it accepts the record and never uses it
    depth = case.soil_depth.astype("<f4")
    depth[~case.mask] = 1.0
    without = run_any(variant(case, tmp_path / "b", {SOIL_DEPTH: depth.tobytes()}),
                      "pytest-outside-b", dhsvm_bin)
    with_record = run_any(variant(case, tmp_path / "c", {SOIL_DEPTH: depth.tobytes(),
                                                         STREAM_MAP: new_map}),
                          "pytest-outside-c", dhsvm_bin)
    assert with_record["exit_code"] == 0
    assert with_record["outputs"] == without["outputs"] == short_baseline["outputs"]


def test_stream_map_without_records_starves_the_network(case, short_baseline, dhsvm_bin,
                                                        tmp_path):
    got = run_any(variant(case, tmp_path / "v", {STREAM_MAP: "# no records\n"}),
                  "pytest-emptymap", dhsvm_bin)
    assert got["exit_code"] == 0 and not got["problems"]  # DHSVM reads it without an error
    # no cell gives water to a segment; the outlet only drains the initial channel storage
    q0 = short_baseline["mean_flow_m3_per_step"]["Chiwawa"]
    assert got["mean_flow_m3_per_step"]["Chiwawa"] < 0.05 * q0


def test_class_file_quirks(case, short_baseline, dhsvm_bin, tmp_path):
    lines = case.classes.lines
    by_id = case.classes.by_id()
    # a width of 0 aborts DHSVM (SIGABRT), although the width is never used ...
    zero_width = edit_line(lines, by_id[13].lineno, 1, "0.0")
    got = run_any(variant(case, tmp_path / "a", {CLASSES: zero_width}), "pytest-class-a",
                  dhsvm_bin)
    assert got["exit_code"] == -6 and "width cannot be 0.0" in got["log"]
    # ... every width set to 0.03 m changes nothing ...
    narrow = "\n".join(replace_token(ln, 1, "0.03") if ln[:1].isdigit() else ln for ln in lines)
    got = run_any(variant(case, tmp_path / "b", {CLASSES: narrow}), "pytest-class-b", dhsvm_bin)
    assert got["exit_code"] == 0 and got["outputs"] == short_baseline["outputs"]
    # ... and a class with ID 0 draws a message but is accepted
    with_zero = "\n".join(lines) + "\n0 1.0 0.5 0.05 0.0\n"
    got = run_any(variant(case, tmp_path / "c", {CLASSES: with_zero}), "pytest-class-c",
                  dhsvm_bin)
    assert "class 0: class id invalid" in got["log"]
    assert got["exit_code"] == 0 and got["outputs"] == short_baseline["outputs"]


def test_placeholder_classes_change_the_hydrograph(case, short_baseline, dhsvm_bin, tmp_path):
    mutated, _ = mutate_case(case, "placeholder-classes", tmp_path / "mut")
    got = run_any(mutated, "pytest-placeholder", dhsvm_bin)
    assert got["exit_code"] == 0 and not got["problems"]
    assert got["warning_lines"] == short_baseline["warning_lines"]
    # only the channel routing changes: the hillslope water balance is the same
    assert got["outputs"]["Aggregated.Values"] == short_baseline["outputs"]["Aggregated.Values"]
    assert got["outputs"]["Mass.Balance"] == short_baseline["outputs"]["Mass.Balance"]
    assert got["outputs"]["Streamflow.Only"] != short_baseline["outputs"]["Streamflow.Only"]
    q0 = short_baseline["mean_flow_m3_per_step"]["Chiwawa"]
    assert got["mean_flow_m3_per_step"]["Chiwawa"] < 0.5 * q0  # shallow depth: slow routing


def test_float64_dem_is_read_without_a_read_error(case, dhsvm_bin, tmp_path):
    mutated, _ = mutate_case(case, "float64-dem", tmp_path / "mut")
    got = run_any(mutated, "pytest-float64", dhsvm_bin)
    assert "Error while reading file" not in got["log"] and "Cannot open" not in got["log"]
    assert "Initializing Road/Stream Networks" in got["log"]  # start-up went on past the maps
    # on Chiwawa the run then stops at its first time step, with an unrelated message
    assert got["exit_code"] == 47 and "Table index out of bounds" in got["log"]


def test_chiwawa_canopy_gap_map_is_too_short(case, dhsvm_bin, tmp_path):
    gap = case.path("VEGETATION", "Canopy Gap Map File")
    assert gap.stat().st_size == case.grid.nrows * case.grid.ncols  # 1 byte per cell
    got = run_any(variant(case, tmp_path / "v", values={("OPTIONS", "Canopy Gapping"): "TRUE"}),
                  "pytest-gap", dhsvm_bin)
    assert got["exit_code"] == 2
    assert "Error while reading file: ../../TestCase/Chiwawa/input/CanopyGap.bin" in got["log"]


# ====================================================================================
# Root depth, state files and the output directory: what DHSVM does with the inputs that
# MAP-003, STA-001 and RUN-002 report.
# ====================================================================================
OUTPUT_DIR = ("OUTPUT", "Output Directory")
STATE_DIR = ("OUTPUT", "Initial State Directory")
STAMP = "10.01.1970.00.00.00"  # Model Start of the case, as in its state file names


def test_soil_one_step_deeper_than_the_roots_is_accepted(case, dhsvm_bin, tmp_path):
    """The other side of ``shallow-soil``: DHSVM stops on soil depth <= root depth and
    accepts the next single-precision number (CheckOut.c:139)."""
    mutated, _ = mutate_case(case, "shallow-soil", tmp_path / "mut")
    old = np.fromfile(case.path(*SOIL_DEPTH), dtype="<f4")
    new = np.fromfile(DhsvmCase(mutated, case.workdir).path(*SOIL_DEPTH), dtype="<f4")
    changed = old != new
    assert changed.sum() == 9
    new[changed] = np.nextafter(new[changed], np.float32(np.inf))
    got = run_any(variant(case, tmp_path / "v", {SOIL_DEPTH: new.tobytes()}), "pytest-deeper",
                  dhsvm_bin)
    assert got["exit_code"] == 0 and not got["problems"]


def linked_states(case: DhsvmCase, directory: Path, skip: tuple = ()) -> Path:
    """Links in ``directory`` to the state files DHSVM reads when it starts the case."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [s.path for s in state_files(case) if s.label not in skip]
    paths.append(case.start_path(f"{case.state_prefix}Channel.State.{STAMP}"))
    for path in paths:
        (directory / path.name).symlink_to(path.resolve())
    return directory


def test_dhsvm_stops_on_a_missing_state_file(case, dhsvm_bin, short_tmp):
    states = linked_states(case, short_tmp / "state", skip=("snow state",))
    got = run_any(variant(case, short_tmp / "v", values={STATE_DIR: f"{states}/"}),
                  "pytest-nostate", dhsvm_bin)
    assert got["exit_code"] == 3
    assert f"Cannot open file: {states}/Snow.State.{STAMP}.bin" in got["log"]


def first_flow(streamflow: bytes) -> float:
    """Outlet flow of the first time step in Streamflow.Only (m3 per step)."""
    for line in streamflow.decode().splitlines()[1:]:
        fields = line.split()
        if len(fields) > 1:
            return float(fields[1])
    raise AssertionError("Streamflow.Only has no flow record")


def test_state_written_for_more_soil_layers_floods_the_basin(case, short_baseline, dhsvm_bin,
                                                             short_tmp):
    """Soil.State with 12 data sets where DHSVM reads 10: it reads the first 10 and says
    nothing. Moisture is right; the surface temperature is the lowest layer's moisture,
    the layer temperatures are shifted by one, and the ground heat storage and the
    surface water are soil temperatures (InitModelState.c:308-390)."""
    mutated, _ = mutate_case(case, "state-extra-layer", short_tmp / "mut")
    got = run_any(mutated, "pytest-state-extra-layer", dhsvm_bin)
    assert got["exit_code"] == 0 and "END OF MODEL RUN" in got["log"]
    assert first_flow(got["outputs"]["Streamflow.Only"]) > 100 * first_flow(
        short_baseline["outputs"]["Streamflow.Only"])


def run_directly(name: str, config, dhsvm_bin: Path, setup) -> subprocess.CompletedProcess:
    """Run DHSVM in ``<case>/run/<name>/`` after ``setup(work)`` has put files there
    (run.py always starts from an empty run directory)."""
    work = chiwawa_dir() / "run" / name
    shutil.rmtree(work, ignore_errors=True)
    (work / "output").mkdir(parents=True)
    setup(work)
    config = config.with_value("TIME", "Model End", SHORT_END)
    config.with_value(*OUTPUT_DIR, "output/").write(work / "INPUT")
    return subprocess.run([str(dhsvm_bin), "INPUT"], cwd=work, capture_output=True, text=True,
                          errors="replace")


def test_empty_state_directory_means_the_start_directory(case, dhsvm_bin):
    """InitDump.c:101-103 falls back to the output directory when Initial State Directory
    is empty, and then overwrites the fallback with the empty text."""
    config = case.config.with_value(*STATE_DIR, "")
    proc = run_directly("pytest-state-in-output", config, dhsvm_bin,
                        lambda work: linked_states(case, work / "output"))
    assert proc.returncode == 3
    assert f"Cannot open file: Channel.State.{STAMP}" in proc.stdout + proc.stderr  # bare name
    proc = run_directly("pytest-state-in-start-dir", config, dhsvm_bin,
                        lambda work: linked_states(case, work))
    assert proc.returncode == 0 and "END OF MODEL RUN" in proc.stdout


def test_dhsvm_stops_on_an_empty_output_directory(case, dhsvm_bin, tmp_path):
    got = run_any(variant(case, tmp_path / "v", values={OUTPUT_DIR: ""}), "pytest-no-output",
                  dhsvm_bin, keep_output=True)
    assert got["exit_code"] == 51 and "Key not found or invalid entry" in got["log"]


def test_chiwawa_as_distributed_has_no_output_directory(chiwawa_input, dhsvm_bin):
    """The INPUT of the test case names TestCase/Chiwawa/output/, which the repository
    does not contain: DHSVM stops until someone creates it."""
    assert not (chiwawa_dir() / "TestCase" / "Chiwawa" / "output").exists()
    got = run_any(chiwawa_input, "pytest-as-distributed", dhsvm_bin, keep_output=True)
    assert got["exit_code"] == 3
    assert "Cannot open file: ../../TestCase/Chiwawa/output/Aggregated.Values" in got["log"]


def test_output_directory_without_a_slash(case, short_baseline, dhsvm_bin, tmp_path):
    base = variant(case, tmp_path / "base", values={OUTPUT_DIR: "output/"})
    mutated, _ = mutate_case(DhsvmCase(base, case.workdir), "output-no-slash",
                             tmp_path / "mut")
    got = run_any(mutated, "pytest-output-no-slash", dhsvm_bin, keep_output=True)
    assert got["exit_code"] == 0 and "END OF MODEL RUN" in got["log"]
    assert not got["outputs"]  # nothing in output/ ...
    # ... the same results are next to it, under names that start with "output"
    assert sorted(got["beside_output"]) == [
        "outputAggregated.Values", "outputMass.Balance", "outputMass.Final.Balance",
        "outputStream.Flow", "outputStreamflow.Only", "outputsaturation_extent.txt"]
    assert got["beside_output"]["outputStreamflow.Only"] == short_baseline["outputs"][
        "Streamflow.Only"]


def test_every_mutation_has_a_dhsvm_experiment():
    runs = {"reverse-network", "jump", "force-wire",  # test_dhsvm_runs_mutants_without_complaint
            "negative-outlet",  # Chiwawa as shipped: test_negative_outlet_is_read_as_segment_1
            "cycle", "zero-length-record", "placeholder-classes", "float64-dem",
            "state-extra-layer", "output-no-slash"}
    assert runs | set(STOPS) == set(MUTATIONS)


# ====================================================================================
# Upstream behaviour that no rule uses yet (DHSVM-PNNL e6f333d)
# ====================================================================================
def test_sky_view_map_has_no_effect(case, short_baseline, dhsvm_bin, tmp_path):
    """With MM5 = FALSE, InitMetMaps reads the sky view map and then resets the sky view
    factor of every cell to 1.0 (InitMetMaps.c:73-86), although MakeLocalMetData.c:215
    would scale the diffuse radiation with it."""
    assert case.flag("Shading") and not case.flag("MM5")
    entry = ("OPTIONS", "Skyview data path")
    sky = np.fromfile(case.path(*entry), dtype="<f4")
    assert sky.min() < 0.6 and sky.max() == 1.0  # the map does hold a real sky view field
    flat = np.full_like(sky, 0.2).tobytes()
    got = run_any(variant(case, tmp_path / "v", {entry: flat}), "pytest-skyview", dhsvm_bin)
    assert got["exit_code"] == 0 and got["outputs"] == short_baseline["outputs"]


def test_road_network_cannot_be_read(case, dhsvm_bin, tmp_path):
    """InitChannel stores the result of a comparison instead of the road class list
    (DHSVMChannel.c:107-108), so every road segment fails its class lookup."""
    d = tmp_path / "road"
    d.mkdir()
    free = case.mask.copy()
    free[case.stream_map.row, case.stream_map.col] = False
    row, col = (int(v) for v in np.argwhere(free)[0])
    (d / "road.class.dat").write_text("1 5.0 0.5 0.02 0.0 OUTSLOPED\n")
    (d / "road.network.dat").write_text("1 1 0.05 90.0 1 0\n")
    (d / "road.map.dat").write_text(f"{col} {row} 1 90.0 0.5 5.0 0.0\n")
    anchor = "Riparian Veg File    = NA"
    text = case.config.text
    assert anchor in text
    (d / "INPUT").write_text(text.replace(
        anchor, f"{anchor}\nRoad Map File = {d / 'road.map.dat'}\n"
                f"Road Network File = {d / 'road.network.dat'}\n"
                f"Road Class File = {d / 'road.class.dat'}"))
    got = run_any(d / "INPUT", "pytest-road", dhsvm_bin)
    assert got["exit_code"] == 5
    assert "road.network.dat: segment 1: channel class 1 not found" in got["log"]
    assert "road.class.dat" not in got["log"].split("Reading Road data")[1]  # classes read fine
