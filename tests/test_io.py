"""Readers follow DHSVM's table scanner (tableio.lex) and binary map reader."""

from __future__ import annotations

import numpy as np
import pytest

from wader.models.dhsvm.io import (
    NETWORK_FIELDS,
    MapReadError,
    as_c_float,
    match_fields,
    read_bin_map,
    read_stream_classes,
    read_stream_map,
    read_stream_network,
    replace_token,
    rewrite_network,
    scan_line,
)

CHIWAWA_OUTLET = '  494  69     0.00519        1139.11688  12      -1\tSAVE"Chiwawa"'


def test_minus_sign_is_skipped():
    tokens, _, echoed = scan_line(CHIWAWA_OUTLET)
    assert [t.kind for t in tokens] == ["int", "int", "real", "real", "int", "int", "word",
                                        "string"]
    assert tokens[5].value == 1 and tokens[5].echoed_before == "-"
    assert tokens[7].value == "Chiwawa"
    assert echoed == "-"


def test_exponents_as_dhsvm_reads_them():
    tokens, _, echoed = scan_line("1.5E-3")
    assert [(t.kind, t.value) for t in tokens] == [("real", 1.5), ("int", 3)]
    assert echoed == "E-"
    assert [t.value for t in scan_line("1.5E3")[0]] == [1500.0]
    assert [t.value for t in scan_line("1.5D3")[0]] == [1.5]  # strtod ignores 'D'
    assert [t.value for t in scan_line("1e-05")[0]] == [1, 5]


def test_comments_and_optional_fields():
    tokens, _, _ = scan_line('7 1 0.1 100 3 SAVE "x y"  # comment 99')
    values, errors, ignored = match_fields(tokens, NETWORK_FIELDS)
    assert not errors and not ignored
    assert "outlet" not in values  # optional field skipped: outlet ID defaults to 0
    assert values["save"][0] == "SAVE" and values["save_name"][0] == "x y"
    assert values["length"][0] == 100.0  # INTEGER accepted for a REAL field


def test_missing_required_field_is_an_error():
    _, errors, _ = match_fields(scan_line("7 1 0.1")[0], NETWORK_FIELDS)
    assert any("required" in e for e in errors)


def test_field_that_matches_no_optional_field_is_dropped_not_an_error():
    # tableio.lex:270-283: 'unable to match field' is printed without counting an error
    values, errors, ignored = match_fields(scan_line("7 1 0.1 100 3 5 42")[0], NETWORK_FIELDS)
    assert not errors and ignored == ["int 42 matches no remaining field"]
    assert values["outlet"][0] == 5
    # an unquoted name after SAVE is a WORD, not the STRING the name field expects
    values, errors, ignored = match_fields(scan_line("7 1 0.1 100 3 5 SAVE Out")[0],
                                           NETWORK_FIELDS)
    assert not errors and "save" in values and "save_name" not in values
    assert ignored == ["word 'Out' matches no remaining field"]
    # but a ninth field is one too many, and that is an error
    _, errors, _ = match_fields(scan_line('7 1 0.1 100 3 5 SAVE "Out" 9')[0], NETWORK_FIELDS)
    assert errors == ["too many fields, expected 8"]


def test_network_outlet_written_as_minus_one(tmp_path):
    p = tmp_path / "net.dat"
    p.write_text("1 1 0.1 100 3 494\n" + CHIWAWA_OUTLET + "\n")
    net = read_stream_network(p)
    seg = net.by_id()[494]
    assert seg.outlet == 1 and seg.outlet_signed == -1 and seg.save
    assert net.intended_downstream() == {1: 494, 494: None}
    assert any("skips" in n.message for n in net.notes)


def test_rewrite_changes_only_the_named_fields(tmp_path):
    p = tmp_path / "net.dat"
    original = "# header\n    1   1  0.10000   100.000   3     2\n    2   2  0.10000   " \
               "100.000   3    -1\tSAVE\"Out\"\n"
    p.write_text(original)
    net = read_stream_network(p)
    flipped = rewrite_network(net, {1: None, 2: 1}, {1: 2, 2: 1})
    assert flipped.splitlines() == [
        "# header",
        "    1   2  0.10000   100.000   3     0",
        "    2   1  0.10000   100.000   3     1\tSAVE\"Out\"",
    ]
    # outlets keep their text unless a sentinel is asked for
    same = rewrite_network(net, net.intended_downstream(), {1: 1, 2: 2})
    assert same == original
    assert "     0\tSAVE" in rewrite_network(net, net.intended_downstream(), {1: 1, 2: 2},
                                            outlet_sentinel="0")


def test_binary_map_size(tmp_path):
    p = tmp_path / "dem.bin"
    np.arange(12, dtype="<f4").tofile(p)
    m = read_bin_map(p, 3, 4, np.dtype("<f4"))
    assert m.data[0, 0] == 0 and m.data[2, 3] == 11 and m.extra_bytes == 0
    assert read_bin_map(p, 2, 4, np.dtype("<f4")).extra_bytes == 16  # DHSVM ignores extras
    with pytest.raises(MapReadError, match="FileIOBin.c"):
        read_bin_map(p, 4, 4, np.dtype("<f4"))


def test_table_notes_tell_errors_from_skipped_characters(tmp_path):
    p = tmp_path / "net.dat"
    p.write_text("1 1 0.1 100 3 -1\n2 1 0.1\n")
    net = read_stream_network(p)
    assert [(n.lineno, n.kind) for n in net.notes] == [(1, "echo"), (2, "error"), (2, "error")]
    assert [s.id for s in net.segments] == [1]  # the incomplete record is not a segment


def test_class_file_as_dhsvm_reads_it(tmp_path):
    p = tmp_path / "class.dat"
    # '15.' is INTEGER 15 followed by a '.' that the scanner skips (as in PNNL's Chiwawa file);
    # the fifth and sixth fields are optional for stream classes (channel.c:97-107)
    p.write_text("#ID W D n inf\n1 2.0 0.400 0.0503 0.0\n10 15. 0.806 0.0506\n"
                 "1 9 9 9\n7 1.0\n")
    table = read_stream_classes(p)
    assert [(c.id, c.width, c.bank_height, c.friction, c.lineno) for c in table.classes] == [
        (1, 2.0, 0.4, 0.0503, 2), (10, 15.0, 0.806, 0.0506, 3), (1, 9.0, 9.0, 9.0, 4)]
    assert table.by_id()[1].bank_height == 0.4  # first match wins (channel.c:71-79)
    assert [(n.lineno, n.kind) for n in table.notes] == [(3, "echo"), (5, "error"), (5, "error")]
    with pytest.raises(MapReadError, match="stream class file not found"):
        read_stream_classes(tmp_path / "nope.dat")


def test_stream_map_keeps_its_lines(tmp_path):
    p = tmp_path / "map.dat"
    p.write_text("# col row id len cut_h cut_w\n  6   0   1   30.000  0.500  2.000    180.0\n")
    sm = read_stream_map(p)
    assert sm.lines[1].split()[:3] == ["6", "0", "1"] and list(sm.lineno) == [2]


def test_replace_token_keeps_the_columns():
    line = "    1   1     0.37931         381.83766  13       3"
    assert replace_token(line, 2, "0.00000") == line.replace("0.37931", "0.00000")
    assert replace_token(line, 4, "7") == "    1   1     0.37931         381.83766   7       3"
    assert replace_token(line, 5, "65539") == line[:-1] + "65539"


def test_table_reals_are_c_floats():
    assert as_c_float(1e-50) == 0.0  # underflows in a C float, so DHSVM sees a slope of 0
    assert as_c_float(0.37931) == pytest.approx(0.37931, rel=1e-7)
