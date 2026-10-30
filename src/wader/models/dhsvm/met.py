"""Which gridded meteorological files DHSVM uses.

With ``[OPTIONS] Gridded Met data = TRUE``, ``InitGridMet`` (InitMetSources.c:
230-385, verified-source) does the following:

1. lists every entry of ``[METEOROLOGY] Met File Path`` (``opendir``/``readdir``,
   :320-322) and parses ``<File Prefix>_<lat>_<lon>`` with ``sscanf("%f")``
   (:313, :327);
2. converts (lat, lon) to UTM in the fixed ``UTM Zone`` with ``deg2utm``
   (:330; deg2utm.c:21-70, single-precision arithmetic);
3. maps the point to a model cell, rounding half away from zero
   (:334-335; Round.c);
4. ignores points outside the grid (:339-343) and, when ``Outside = FALSE``,
   points whose cell is outside the basin mask (:346-347);
5. reopens the file as ``<path><prefix>_%.<GRID_DECIMAL>f_%.<GRID_DECIMAL>f``
   (:315, :350) and skips it with a message if that fails (:352-355).

A file that is absent from the directory is never considered, so leaving out a
file that DHSVM would have selected changes the forcing silently.
``locate_met_files`` reproduces steps 1-4 so that a reference case can be
fetched without the 3.4 GB of unused forcing. Verified-experiment: the
``deg2utm`` port reproduces DHSVM's printed UTM coordinates bit for bit for all
320 Chiwawa forcing points, and DHSVM selects exactly the 13 files this module
selects (``cases/chiwawa/fetch.py --verify-selection``).

Not handled: ``Outside = TRUE``. In that branch the longitude test compares
against ``LonWest`` twice and the file name is never set (:361-362), so it
cannot select a file as written.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

_F = np.float32
PI = 3.14159265358979323846  # constants.h:42


def deg2utm(la: float, lo: float, zone: int) -> tuple[float, float]:
    """Port of DHSVM ``deg2utm`` (deg2utm.c:21-70), including its float rounding.

    Each C ``float`` assignment is rounded to single precision; expressions that
    involve a double constant or a libm call are evaluated in double first.
    """
    sa, sb = _F(6378137.0), _F(6356752.314245)
    la, lo = _F(la), _F(lo)
    e2 = _F(math.pow(math.pow(sa, 2) - math.pow(sb, 2), 0.5) / float(sb))
    e2c = _F(math.pow(e2, 2))
    c = _F(math.pow(sa, 2) / float(sb))
    lat = _F(float(la) * (PI / 180))
    lon = _F(float(lo) * (PI / 180))
    huso = _F(abs(zone))
    s = _F(_F(6) * huso) - _F(183)
    delta_s = _F(float(lon) - float(s) * (PI / 180))
    cl = math.cos(float(lat))
    a = _F(cl * math.sin(float(delta_s)))
    epsilon = _F(0.5 * math.log((1.0 + float(a)) / (1.0 - float(a))))
    nu = _F(math.atan(math.tan(float(lat)) / math.cos(float(delta_s))) - float(lat))
    v = _F(float(c) / math.pow(1.0 + float(e2c) * math.pow(cl, 2.0), 0.5) * 0.9996)
    ta = _F((float(e2c) / 2.0) * math.pow(float(epsilon), 2.0) * math.pow(cl, 2.0))
    a1 = _F(math.sin(2.0 * float(lat)))
    a2 = _F(float(a1) * math.pow(cl, 2.0))
    j2 = _F(float(lat) + float(a1) / 2.0)
    j4 = _F((3.0 * float(j2) + float(a2)) / 4.0)
    j6 = _F((5.0 * float(j4) + float(a2) * math.pow(cl, 2.0)) / 3.0)
    alfa = _F(3.0 / 4.0 * float(e2c))
    beta = _F(5.0 / 3.0 * math.pow(float(alfa), 2.0))
    gama = _F(35.0 / 27.0 * math.pow(float(alfa), 3.0))
    inner = lat - alfa * j2 + beta * j4 - gama * j6  # all float operands: float arithmetic
    bm = _F(0.9996 * float(c) * float(inner))
    xx = _F(float(epsilon * v) * (1.0 + float(ta) / 3.0) + 500000)
    yy = _F(float(nu * v) * (1.0 + float(ta)) + float(bm))
    if yy < 0:
        yy = yy + _F(9999999)
    return float(xx), float(yy)


def c_round(x: float) -> int:
    """DHSVM ``Round`` (Round.c): halves round away from zero."""
    f = math.fabs(x) - math.floor(math.fabs(x))
    if f < 0.5:
        return int(math.ceil(x) if x < 0 else math.floor(x))
    return int(math.floor(x) if x < 0 else math.ceil(x))


@dataclass(frozen=True)
class MetPoint:
    name: str  # directory entry
    lat: float
    lon: float
    row: int  # Loc.N
    col: int  # Loc.E
    in_grid: bool
    in_basin: bool
    near_edge: bool  # within `tol` cells of a rounding boundary (checked both ways)


def locate_met_files(names: list[str], prefix: str, zone: int, north: float, west: float,
                     cellsize: float, nrows: int, ncols: int, in_basin: np.ndarray,
                     tol: float = 1e-6) -> list[MetPoint]:
    """Model cell and in-basin status of each forcing file name (steps 1-4)."""
    pattern = re.compile(re.escape(prefix) + r"_([-+0-9.eE]+)_([-+0-9.eE]+)$")
    dy = float(_F(cellsize))
    points = []
    for name in names:
        m = pattern.match(name)
        if not m:
            continue
        lat, lon = float(_F(float(m.group(1)))), float(_F(float(m.group(2))))
        east, northing = deg2utm(lat, lon, zone)
        fn = ((north - 0.5 * dy) - northing) / dy
        fe = (east - (west + 0.5 * dy)) / dy
        rows = {c_round(fn)} | ({c_round(fn - tol), c_round(fn + tol)})
        cols = {c_round(fe)} | ({c_round(fe - tol), c_round(fe + tol)})
        row, col = c_round(fn), c_round(fe)
        cands = [(r, c) for r in rows for c in cols if 0 <= r < nrows and 0 <= c < ncols]
        points.append(MetPoint(
            name=name, lat=lat, lon=lon, row=row, col=col,
            in_grid=0 <= row < nrows and 0 <= col < ncols,
            in_basin=any(bool(in_basin[r, c]) for r, c in cands),
            near_edge=len(rows) > 1 or len(cols) > 1,
        ))
    return points


def dhsvm_file_name(prefix: str, lat: float, lon: float, decimals: int) -> str:
    """Name DHSVM opens for a point (InitMetSources.c:315, :350)."""
    return f"{prefix}_{lat:.{decimals}f}_{lon:.{decimals}f}"
