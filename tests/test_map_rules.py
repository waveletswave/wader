"""MAP-001 on the synthetic case: which binary maps DHSVM reads and how many bytes."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import INPUT_TEMPLATE, NCOLS, NROWS, make_synthetic_case

from wader.core import Outcome, Severity
from wader.models import dhsvm
from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.maps import _shadow_layers, expected_maps, max_soil_layers

CELLS = NROWS * NCOLS


@pytest.fixture
def work(tmp_path):
    make_synthetic_case(tmp_path)
    return tmp_path


def check(work):
    return dhsvm.check(work / "INPUT.synthetic", work, rules=["MAP-001"]).results[0]


def set_input(work, *replacements, add=""):
    text = INPUT_TEMPLATE
    for old, new in replacements:
        assert old in text, old
        text = text.replace(old, new)
    (work / "INPUT.synthetic").write_text(text.replace("[End]\n", add + "[End]\n"))


def test_clean_case(work):
    res = check(work)
    assert res.outcome == Outcome.PASS and not res.findings
    files = {f["map"]: f for f in res.metrics["files"]}
    assert list(files) == ["DEM", "basin mask", "soil type", "soil depth", "vegetation type"]
    assert files["DEM"]["expected_bytes"] == files["DEM"]["bytes"] == CELLS * 4
    assert files["basin mask"]["expected_bytes"] == CELLS
    assert files["DEM"]["entry"] == "[TERRAIN] DEM File"
    assert "VarID.c:51-53" in files["DEM"]["source"]


def test_short_long_and_missing_files(work):
    inp = work / "input"
    np.zeros(CELLS - 1, dtype="u1").tofile(inp / "mask.bin")  # one byte short
    np.zeros(CELLS, dtype="<f8").tofile(inp / "soild.bin")  # 8-byte floats
    np.zeros(CELLS + 5, dtype="u1").tofile(inp / "soil.bin")  # five stray bytes
    (inp / "veg.bin").unlink()
    res = check(work)
    assert res.outcome == Outcome.FAIL
    found = {f.where["map"]: f for f in res.findings}
    assert set(found) == {"basin mask", "soil depth", "soil type", "vegetation type"}
    assert all(f.severity == Severity.ERROR for f in found.values())
    assert found["basin mask"].level == "L0" and "exit 2" in found["basin mask"].message
    assert found["vegetation type"].level == "L0" and "exit 3" in found["vegetation type"].message
    assert found["soil depth"].level == "L1" and "8-byte floats" in found["soil depth"].message
    assert found["soil type"].level == "L1"
    assert "another grid size or number type" in found["soil type"].message
    assert res.metrics["missing"] == res.metrics["too_short"] == 1
    assert res.metrics["too_long"] == 2


def test_optional_and_layered_maps(work):
    inp = work / "input"
    np.zeros(CELLS * 12, dtype="<f4").tofile(inp / "lai.bin")  # 12 monthly data sets
    np.zeros(CELLS * 2, dtype="<f4").tofile(inp / "porosity.bin")  # 2 layers, 3 are read
    np.zeros(CELLS, dtype="u1").tofile(inp / "gap.bin")  # 1 byte per cell, a float is read
    np.zeros(CELLS, dtype="<f4").tofile(inp / "ts.bin")
    set_input(
        work,
        ("Canopy Gapping       = FALSE", "Canopy Gapping       = TRUE"),
        ("Snow Threshold       = 0.5", "Snow Threshold       = input/ts.bin"),
        ("Vegetation Map File  = input/veg.bin",
         "Vegetation Map File  = input/veg.bin\n"
         "Vegetation LAI Map File = input/lai.bin\n"
         "Vegetation FC Map File = none\n"
         "Canopy Gap Map File  = input/gap.bin"),
        ("Soil Depth File      = input/soild.bin",
         "Soil Depth File      = input/soild.bin\nSoil Porosity Map File = input/porosity.bin"),
    )
    case = DhsvmCase(work / "INPUT.synthetic", work)
    assert max_soil_layers(case) == 3
    specs = {s.label: s for s in expected_maps(case)}
    assert specs["vegetation LAI"].layers == 12 and specs["soil porosity"].layers == 3
    assert "vegetation fractional cover" not in specs and "lateral conductivity" not in specs
    assert specs["snow threshold (snow parameter)"].path == work / "input" / "ts.bin"
    res = check(work)
    found = {f.where["map"]: f for f in res.findings}
    assert set(found) == {"soil porosity", "canopy gap"}
    assert "x 3 data sets" in found["soil porosity"].message
    assert f"{CELLS} bytes, but DHSVM reads {CELLS * 4} bytes" in found["canopy gap"].message


def test_shading_maps_are_read_month_by_month(work):
    shadow = work / "input" / "shadow"
    shadow.mkdir()
    np.ones(CELLS, dtype="<f4").tofile(shadow / "SkyView.bin")
    for month in (11, 12, 1):  # 8 steps per day; the run starts on 11/20 and ends on 2/1
        np.zeros(CELLS * 8, dtype="u1").tofile(shadow / f"Shadow.{month:02d}.bin")
    np.zeros(CELLS * 24, dtype="u1").tofile(shadow / "Shadow.12.bin")  # an hourly file
    set_input(
        work,
        ("Shading              = FALSE",
         "Shading              = TRUE\n"
         "Shading data path    = input/shadow/Shadow\n"
         "Shading data extension = bin\n"
         "Skyview data path    = input/shadow/SkyView.bin"),
        ("Model Start          = 10/01/1970-00", "Model Start          = 11/20/1970-00"),
        ("Model End            = 10/02/1970-00", "Model End            = 02/01/1971-00"),
    )
    case = DhsvmCase(work / "INPUT.synthetic", work)
    labels = [s.label for s in expected_maps(case)]
    assert labels[-5:] == ["sky view", "shading, month 11", "shading, month 12",
                           "shading, month 01", "shading, month 02"]
    res = check(work)
    found = {f.where["map"]: f for f in res.findings}
    assert set(found) == {"shading, month 12", "shading, month 02"}
    # the last time step (02/01, 00:00) is in February, so DHSVM opens that file too
    assert found["shading, month 02"].level == "L0"
    assert "when the run reaches month 02" in found["shading, month 02"].message
    assert found["shading, month 12"].level == "L1"
    assert "holds 24 data sets of that size where DHSVM reads 8" in (
        found["shading, month 12"].message)


def test_shadow_layers_follow_the_time_step():
    # InitNewMonth.c:98-106; above 24 steps per day the index uses integer division
    assert [_shadow_layers(n) for n in (1, 8, 24, 48, 36, 96)] == [1, 8, 24, 24, 36, 24]


def test_inputs_that_are_not_checked_are_named(work):
    set_input(work, ("Gridded Met data     = FALSE",
                     "Gridded Met data     = FALSE\nPRISM                = TRUE\n"
                     "Precipitation lapse rate = MAP"))
    res = check(work)
    assert res.outcome == Outcome.PASS
    [note] = res.findings
    assert note.severity == Severity.INFO
    assert "PRISM" in note.message and "lapse rate map" in note.message


def test_formats_other_than_bin_are_not_evaluated(work):
    set_input(work, ("Format               = BIN", "Format               = NETCDF"))
    res = check(work)
    assert res.outcome == Outcome.ERROR and "NETCDF" in res.summary


def test_missing_entry_for_a_required_map_is_not_evaluated(work):
    set_input(work, ("Soil Map File        = input/soil.bin\n", ""))
    res = check(work)
    assert res.outcome == Outcome.ERROR and "Soil Map File" in res.summary
