"""STA-001 on the synthetic case: which gridded state files DHSVM restores, under which
names, and how many data sets of each."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import INPUT_TEMPLATE, NCOLS, NROWS, STAMP, make_synthetic_case

from wader.core import Outcome, Severity
from wader.models import dhsvm
from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.maps import max_veg_layers, state_files, state_stamp
from wader.models.dhsvm.mutate import mutate_case

SET = NROWS * NCOLS * 4  # bytes of one data set


@pytest.fixture
def work(tmp_path):
    make_synthetic_case(tmp_path)
    return tmp_path


def check(work):
    return dhsvm.check(work / "INPUT.synthetic", work, rules=["STA-001"]).results[0]


def set_input(work, *replacements):
    text = INPUT_TEMPLATE
    for old, new in replacements:
        assert old in text, old
        text = text.replace(old, new)
    (work / "INPUT.synthetic").write_text(text)


def write_state(work, kind, sets, directory="state"):
    np.zeros(sets * NROWS * NCOLS, dtype="<f4").tofile(
        work / directory / f"{kind}.State.{STAMP}.bin")


def test_clean_case(work):
    res = check(work)
    assert res.outcome == Outcome.PASS
    assert res.summary == "3 gridded state files have the size DHSVM restores"
    files = {f["state"]: f for f in res.metrics["files"]}
    assert [f["data_sets"] for f in files.values()] == [5, 8, 10]
    assert files["soil state"]["bytes"] == files["soil state"]["expected_bytes"] == 10 * SET
    assert files["snow state"]["path"].endswith(f"state/Snow.State.{STAMP}.bin")
    # the channel state is a table of its own: named, not checked
    (note,) = res.findings
    assert note.severity == Severity.INFO and f"state/Channel.State.{STAMP}" in note.message


def test_names_and_layer_counts_come_from_the_input(work):
    case = DhsvmCase(work / "INPUT.synthetic", work)
    assert state_stamp(case) == STAMP and max_veg_layers(case) == 2
    assert [s.path.name for s in state_files(case)] == [
        f"Interception.State.{STAMP}.bin", f"Snow.State.{STAMP}.bin", f"Soil.State.{STAMP}.bin"]
    # a start at 03:00 on another day is another set of files
    set_input(work, ("Model Start          = 10/01/1970-00",
                     "Model Start          = 11/09/1970-03"))
    res = check(work)
    assert res.outcome == Outcome.FAIL and res.metrics["missing"] == 3
    assert "Soil.State.11.09.1970.03.00.00.bin" in res.findings[2].message


def test_short_long_and_missing_files(work):
    write_state(work, "Interception", 3)  # as for one canopy layer
    write_state(work, "Soil", 12)  # as for four soil layers
    (work / "state" / f"Snow.State.{STAMP}.bin").unlink()
    res = check(work)
    assert res.outcome == Outcome.FAIL
    assert (res.metrics["too_short"], res.metrics["missing"], res.metrics["too_long"]) == (1, 1, 1)
    by = {f.where["state"]: f for f in res.findings if f.where}
    short, missing, long_ = (by[k] for k in ("canopy interception state", "snow state",
                                             "soil state"))
    assert short.level == "L0" and "exit 2" in short.message
    assert "it holds 3 data sets, as a state written for 1 canopy layer does" in short.message
    assert missing.level == "L0" and "exit 3" in missing.message
    assert long_.level == "L1" and long_.severity == Severity.ERROR
    assert "it holds 12 data sets, as a state written for 4 soil layers does" in long_.message
    assert f"reads only the first {10 * SET} bytes" in long_.message
    assert long_.values == {"bytes": 12 * SET, "expected_bytes": 10 * SET}


def test_a_size_that_is_no_whole_number_of_data_sets(work):
    path = work / "state" / f"Snow.State.{STAMP}.bin"
    path.write_bytes(path.read_bytes() + b"\0\0\0")
    (f,) = [f for f in check(work).findings if f.where]
    assert f.level == "L1" and "another grid size or number type" in f.message


def test_layer_counts_are_the_largest_of_the_tables(work):
    # class 1 loses its overstory, but class 2 (not on the map) gains one: still 2 layers
    set_input(work, ("Overstory Present 1  = TRUE", "Overstory Present 1  = FALSE"),
              ("Overstory Present 2  = FALSE", "Overstory Present 2  = TRUE"))
    assert check(work).outcome == Outcome.PASS
    # no class with an overstory: 1 layer, 3 data sets, and the 5 in the file are too many
    set_input(work, ("Overstory Present 1  = TRUE", "Overstory Present 1  = FALSE"))
    res = check(work)
    assert res.outcome == Outcome.FAIL and res.metrics["too_long"] == 1
    assert res.metrics["files"][0]["data_sets"] == 3
    # two soil layers: 8 data sets
    set_input(work, ("Number of Soil Layers 1 = 3", "Number of Soil Layers 1 = 2"))
    res = check(work)
    assert res.metrics["files"][2]["data_sets"] == 8 and res.metrics["too_long"] == 1


def test_an_empty_state_directory_is_the_start_directory(work):
    set_input(work, ("Initial State Directory = state/", "Initial State Directory = "))
    res = check(work)
    assert res.outcome == Outcome.FAIL and res.metrics["missing"] == 3
    assert "not in the output directory" in res.findings[0].message
    assert res.metrics["files"][0]["path"] == str(work / f"Interception.State.{STAMP}.bin")
    for kind, sets in (("Interception", 5), ("Snow", 8), ("Soil", 10)):
        write_state(work, kind, sets, directory="output")  # where the fallback would look
    assert check(work).metrics["missing"] == 3
    for kind, sets in (("Interception", 5), ("Snow", 8), ("Soil", 10)):
        write_state(work, kind, sets, directory=".")
    assert check(work).outcome == Outcome.PASS
    # an entry that is not there at all reads the same
    set_input(work, ("Initial State Directory = state/\n", ""))
    assert check(work).outcome == Outcome.PASS


def test_a_state_directory_without_a_slash_is_a_file_name_prefix(work):
    set_input(work, ("Initial State Directory = state/", "Initial State Directory = state"))
    res = check(work)
    assert res.metrics["missing"] == 3
    assert res.metrics["files"][2]["path"] == str(work / f"stateSoil.State.{STAMP}.bin")
    assert "does not end with '/'" in res.findings[0].message


@pytest.mark.parametrize("mutation, level, counter", [
    ("truncate-state", "L0", "too_short"),
    ("state-extra-layer", "L1", "too_long"),
])
def test_mutations_are_caught(work, short_tmp, mutation, level, counter):
    case = DhsvmCase(work / "INPUT.synthetic", work)
    mutated, _ = mutate_case(case, mutation, short_tmp / "mut")
    res = dhsvm.check(mutated, work, rules=["STA-001"]).results[0]
    assert res.outcome == Outcome.FAIL and res.metrics[counter] == 1
    (f,) = [f for f in res.findings if f.where]
    assert f.level == level and f.where["state"] == "soil state"


def test_other_formats_are_not_supported_yet(work):
    set_input(work, ("Format               = BIN", "Format               = NETCDF"))
    res = check(work)
    assert res.outcome == Outcome.ERROR and "NETCDF" in res.summary
