"""Readers follow DHSVM's table scanner (tableio.lex) and binary map reader."""

from __future__ import annotations

import numpy as np
import pytest

from wader.models.dhsvm.io import (
    NETWORK_FIELDS,
    MapReadError,
    match_fields,
    read_bin_map,
    read_stream_network,
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
    values, errors = match_fields(tokens, NETWORK_FIELDS)
    assert not errors
    assert "outlet" not in values  # optional field skipped: outlet ID defaults to 0
    assert values["save"][0] == "SAVE" and values["save_name"][0] == "x y"
    assert values["length"][0] == 100.0  # INTEGER accepted for a REAL field


def test_missing_required_field_is_an_error():
    _, errors = match_fields(scan_line("7 1 0.1")[0], NETWORK_FIELDS)
    assert any("required" in e for e in errors)


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
