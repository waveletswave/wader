"""Gridded-met file selection follows InitGridMet (InitMetSources.c) and deg2utm.c."""

from __future__ import annotations

import numpy as np
import pytest

from wader.models.dhsvm.met import c_round, deg2utm, dhsvm_file_name, locate_met_files

# Printed by DHSVM e6f333d itself ("lat, lon, East, North", InitMetSources.c:332) for
# Chiwawa forcing points in UTM zone 10; the port matched all 320 points exactly.
DHSVM_PRINTED = [
    (48.21875, -121.03125, 646234.625, 5342488.0),
    (47.21875, -121.09375, 644329.25, 5231236.5),
    (47.96875, -120.84375, 660939.75, 5315077.0),
]


@pytest.mark.parametrize(("lat", "lon", "east", "north"), DHSVM_PRINTED)
def test_deg2utm_matches_dhsvm_bit_for_bit(lat, lon, east, north):
    assert deg2utm(lat, lon, 10) == (east, north)


def test_deg2utm_on_the_central_meridian():
    east, north = deg2utm(45.0, -123.0, 10)
    assert east == 500000.0
    assert north == pytest.approx(4982950.4, abs=2.0)  # 0.9996 x meridian arc to 45 N


def test_c_round_halves_away_from_zero():
    assert [c_round(x) for x in (0.49, 0.5, 1.5, -0.5, -1.5, -2.49)] == [0, 1, 2, -1, -2, -2]


def test_locate_and_name():
    # a 10 x 10 grid of 1 km cells whose north-west corner sits on a point's UTM location
    east, north = DHSVM_PRINTED[2][2:]
    in_basin = np.zeros((10, 10), dtype=bool)
    in_basin[0, 0] = True
    names = ["data_47.96875_-120.84375", "data_48.21875_-121.03125", "README"]
    pts = locate_met_files(names, "data", 10, north + 500.0, east - 500.0, 1000.0, 10, 10,
                           in_basin)
    assert [p.name for p in pts] == names[:2]  # non-matching names are not points
    assert (pts[0].row, pts[0].col, pts[0].in_basin) == (0, 0, True)
    assert not pts[1].in_grid and not pts[1].in_basin
    assert dhsvm_file_name("data", 47.96875, -120.84375, 5) == "data_47.96875_-120.84375"
