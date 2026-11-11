"""The INPUT reader follows GetInit.c (DHSVM-PNNL@e6f333d)."""

from __future__ import annotations

from datetime import datetime

from wader.models.dhsvm.config import (
    DhsvmConfig,
    c_float,
    c_floats,
    c_int,
    make_key_string,
    scan_date,
    strip_line,
)


def test_make_key_string_matches_dhsvm():
    # GetInit.c:263-291: upper-case, a tab is deleted, other whitespace runs collapse
    assert make_key_string("Number of Rows") == "NUMBER OF ROWS"
    assert make_key_string("Number  of   Rows") == "NUMBER OF ROWS"
    assert make_key_string("Number of\tGrids") == "NUMBER OFGRIDS"
    assert make_key_string("Number \t of Grids") == "NUMBER OF GRIDS"


def test_strip_line_removes_comment_and_whitespace():
    assert strip_line("  Key = value   # comment\r\n") == "Key = value"
    assert strip_line("# only a comment") == ""


def test_lookup_rules():
    text = (
        "Key = before any section\n"
        "[AREA]  # model area\n"
        "Number of Rows = 0425   # leading zero: octal for strtol(..., 0)\n"
        "number  of rows = 7\n"
        "Grid spacing = 90\n"
        "[TERRAIN]\n"
        "DEM File = dem.bin\n"
        "[AREA]\n"
        "Grid spacing = 30\n"
        "[End]\n"
    )
    c = DhsvmConfig(text)
    assert c.get("AREA", "Number of Rows") == "0425"  # first key wins
    assert c.get_int("AREA", "Number of Rows") == 277  # octal
    assert c.get_float("AREA", "Grid spacing") == 90.0  # first [AREA] section wins
    assert c.get("terrain", "dem file") == "dem.bin"  # case-insensitive names
    assert c.get("AREA", "Key") is None


def test_last_line_without_newline_is_not_read():
    c = DhsvmConfig("[AREA]\nGrid spacing = 90\nNumber of Rows = 10")
    assert c.get("AREA", "Number of Rows") is None
    assert any("no trailing newline" in n.message for n in c.notes)


def test_long_line_is_split():
    c = DhsvmConfig("[AREA]\nGrid spacing = " + "9" * 300 + "\nNumber of Rows = 10\n")
    assert any("characters long" in n.message for n in c.notes)
    # the split pushes the last line out of DHSVM's read loop (GetInit.c:434-438)
    assert c.get("AREA", "Number of Rows") is None


def test_c_int_and_c_float_follow_strtol_strtod():
    assert c_int("42") == 42
    assert c_int("0x1A") == 26
    assert c_int("010") == 8
    assert c_int("08") is None  # strtol stops at '8', so CopyInt fails
    assert c_int("-5") == -5
    assert c_int("12 m") is None
    assert c_float("1e3") == 1000.0
    assert c_float("90.") == 90.0
    assert c_float("ninety") is None


def test_with_value_keeps_everything_else():
    text = "[TIME]\nModel End            = 10/01/1971-00     # Model end time\n[End]\n"
    c = DhsvmConfig(text).with_value("TIME", "Model End", "11/01/1970-00")
    assert c.get("TIME", "Model End") == "11/01/1970-00"
    assert c.text.splitlines()[1] == (
        "Model End            = 11/01/1970-00     # Model end time")
    assert c.text.splitlines()[0] == "[TIME]"


def test_scan_date_follows_sscandate():
    # Calendar.c:100-167: numbers split at every non-digit, read as month day year [h [m [s]]]
    assert scan_date("10/01/1970-00") == datetime(1970, 10, 1, 0)
    assert scan_date("2/29/1972-21") == datetime(1972, 2, 29, 21)
    assert scan_date("10.01.1970 03:30:15") == datetime(1970, 10, 1, 3, 30, 15)
    assert scan_date("10/01/1970") == datetime(1970, 10, 1)
    assert scan_date("2/29/1971-00") is None  # not a leap year
    assert scan_date("13/01/1970-00") is None
    assert scan_date("10/01/1970-24") is None  # hours are 0..23
    assert scan_date("1970-10") is None  # fewer than three numbers
    assert scan_date("1970-10-01") is None  # read as month 1970


def test_c_floats_reads_like_copyfloat():
    # GetInit.c:311-327: n numbers, one strtod after another
    assert c_floats("0.060 0.130 0.210", 3) == [0.06, 0.13, 0.21]
    assert c_floats("1e-1\t2", 2) == [0.1, 2.0]
    assert c_floats("0.1 0.2", 3) is None  # a number is missing
    assert c_floats("0.1 0.2 0.3", 2) is None  # something follows the last one
    assert c_floats("0.1, 0.2", 2) is None
    assert c_floats("anything", 0) == []  # nothing is read and nothing is checked
