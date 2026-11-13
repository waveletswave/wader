"""MAP rules on the synthetic case: which binary maps DHSVM reads and how many bytes
(MAP-001), and soil depth against root depth (MAP-003)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import INPUT_TEMPLATE, NCOLS, NROWS, ROOT_DEPTH, make_synthetic_case

from wader.core import Outcome, Severity
from wader.models import dhsvm
from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.maps import _shadow_layers, expected_maps, max_soil_layers
from wader.models.dhsvm.mutate import mutate_case

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


# ------------------------------------------------------------------ MAP-003
def check3(work, **params):
    return dhsvm.check(work / "INPUT.synthetic", work, rules=["MAP-003"],
                       params=params or None).results[0]


def single_sum(*depths):
    """The sum as DHSVM forms it: accumulated in single precision (CheckOut.c:81-84)."""
    total = np.float32(0.0)
    for d in depths:
        total = np.float32(total + np.float32(d))
    return total


def write_depth(work, cells):
    depth = np.full((NROWS, NCOLS), 1.5, dtype="<f4")
    for (row, col), value in cells.items():
        depth[row, col] = value
    depth.tofile(work / "input" / "soild.bin")


def test_soil_deeper_than_roots_everywhere(work):
    res = check3(work)
    assert res.outcome == Outcome.PASS and not res.findings
    assert "188 basin cells" in res.summary and "0.75 m" in res.summary
    assert res.metrics["tested_cells"] == CELLS - 4  # the four corners are outside the basin
    assert res.metrics["min_margin_m"] == pytest.approx(1.5 - ROOT_DEPTH, abs=1e-6)
    # only the class that is on the map is listed; class 2 is in the table only
    assert [c["class"] for c in res.metrics["classes"]] == [1]
    assert res.metrics["classes"][0]["total_root_depth_m"] == float(single_sum(0.10, 0.25, 0.40))


def test_the_comparison_is_dhsvms_single_precision_one(work):
    total = single_sum(0.10, 0.25, 0.40)
    write_depth(work, {(5, 5): total})  # equal: DHSVM stops on "<="
    res = check3(work)
    assert res.outcome == Outcome.FAIL
    (f,) = res.findings
    assert f.severity == Severity.ERROR and res.rule.level == "L0"
    assert f.where["n_cells"] == 1 and f.where["cells"] == [[5, 5]]
    assert "exit 255" in f.message and "row 5, column 5" in f.message
    assert "vegetation class 1 (Forest)" in f.message
    # one single-precision step deeper is enough for DHSVM, and for wader
    write_depth(work, {(5, 5): np.nextafter(total, np.float32(np.inf))})
    res = check3(work)
    assert res.outcome == Outcome.PASS
    assert "in 1 cells the layer below the root zones is thinner than 0.05 m" in res.summary
    assert check3(work, thin_layer_m=1e-9).summary.endswith("m)")


def test_cells_outside_the_basin_do_not_count(work):
    write_depth(work, {(0, 0): 0.0, (15, 11): -9999.0})  # corners, outside the mask
    assert check3(work).outcome == Outcome.PASS


def test_each_cell_is_compared_with_its_own_class(work):
    veg = np.ones((NROWS, NCOLS), dtype="u1")
    veg[8:, :] = 2  # class 2: root zones 0.10 + 0.20 + 0.30
    veg.tofile(work / "input" / "veg.bin")
    write_depth(work, {(2, 2): 0.7, (10, 2): 0.7, (12, 3): single_sum(0.10, 0.20, 0.30)})
    res = check3(work)
    assert res.outcome == Outcome.FAIL
    by_class = {f.where["vegetation_class"]: f for f in res.findings}
    assert sorted(by_class) == [1, 2]
    assert by_class[1].where["cells"] == [[2, 2]]  # 0.7 m is too shallow for class 1 only
    assert by_class[2].where["cells"] == [[12, 3]]
    assert res.metrics["failing_cells"] == 2


def test_a_class_that_is_not_in_the_table_is_reported(work):
    veg = np.ones((NROWS, NCOLS), dtype="u1")
    veg[3, 3], veg[4, 4], veg[0, 0] = 0, 7, 9  # (0, 0) is outside the basin
    veg.tofile(work / "input" / "veg.bin")
    res = check3(work)
    assert res.outcome == Outcome.FAIL
    (f,) = res.findings
    assert f.where["cells"] == [[3, 3], [4, 4]] and f.values["vegetation_classes"] == [0, 7]
    assert "veg value N out of range" in f.message
    assert res.metrics["untested_cells"] == 2 and "2 more could not be tested" in res.summary


def test_shallow_soil_mutation_is_caught(work, tmp_path):
    case = DhsvmCase(work / "INPUT.synthetic", work)
    mutated, _ = mutate_case(case, "shallow-soil", tmp_path / "mut")
    res = dhsvm.check(mutated, work, rules=["MAP-003"]).results[0]
    assert res.outcome == Outcome.FAIL
    assert [f.where["n_cells"] for f in res.findings] == [9]


@pytest.mark.parametrize("old, new, message", [
    ("Root Zone Depths 1   = 0.10 0.25 0.40", "Root Zone Depths 1   = 0.10 0.25",
     "is not 3 numbers"),
    ("Root Zone Depths 1   = 0.10 0.25 0.40", "Root Zone Depths 1   = 0.10 0.25 0.40 0.5",
     "is not 3 numbers"),
    ("Overstory Present 1  = TRUE", "Overstory Present 1  = yes", "is not TRUE or FALSE"),
    ("Number of Root Zones 1 = 3\n", "", "Number of Root Zones 1"),
])
def test_a_vegetation_table_dhsvm_rejects_is_an_input_problem(work, old, new, message):
    set_input(work, (old, new))
    res = check3(work)
    assert res.outcome == Outcome.ERROR and message in res.summary
