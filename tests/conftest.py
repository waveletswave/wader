"""Shared fixtures: a small synthetic DHSVM case, the fetched Chiwawa case, DHSVM.

Tests that need fetched data or a DHSVM executable skip when these are missing,
unless WADER_REQUIRE_CASES=1 (set in CI), in which case they fail.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wader.cache import chiwawa_dir, dhsvm_executable

REPO = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------- synthetic case
# A 16 x 12 grid of 30 m cells; the valley runs down column 6 to the south edge.
NROWS, NCOLS, CELL = 16, 12, 30.0
SEGMENTS = {  # id: (row, col) cells, upstream end first
    1: [(0, 6), (1, 6), (2, 6), (3, 6)],
    2: [(0, 2), (1, 3), (2, 4), (3, 5)],
    3: [(4, 6), (5, 6), (6, 6), (7, 6), (8, 6)],
    4: [(5, 10), (6, 9), (7, 8), (8, 7)],
    5: [(r, 6) for r in range(9, 16)],  # outlet; contains the lowest cell (15, 6)
    6: [(11, 8), (12, 7)],
}
DOWN = {1: 3, 2: 3, 3: 5, 4: 5, 5: None, 6: 5}
ORDER = {1: 1, 2: 1, 3: 2, 4: 1, 5: 3, 6: 1}
LENGTH = {1: 120.0, 2: 170.0, 3: 150.0, 4: 170.0, 5: 210.0, 6: 72.4}

INPUT_TEMPLATE = """\
# Synthetic DHSVM setup for wader tests (only the entries wader reads)
[OPTIONS]                     # Model options
Format               = BIN    # BIN, BYTESWAP or NETCDF
Extent               = BASIN
Flow Routing         = NETWORK
Gridded Met data     = FALSE
Shading              = FALSE
Canopy Gapping       = FALSE
[AREA]
Extreme North        = 5000480.0
Extreme West         = 500000.0
Number of Rows       = 16
Number of Columns    = 12
Grid spacing         = 30
[TIME]
Time Step            = 3
Model Start          = 10/01/1970-00
Model End            = 10/02/1970-00
[CONSTANTS]
Outside Basin Value  = 0
Rain Threshold       = -0.5
Snow Threshold       = 0.5
[TERRAIN]
DEM File             = input/dem.bin
Basin Mask File      = input/mask.bin
[ROUTING]
Stream Map File      = input/stream.map.dat
Stream Network File  = input/stream.network.dat
Stream Class File    = input/stream.class.dat
[SOILS]
Soil Map File        = input/soil.bin
Soil Depth File      = input/soild.bin
Number of Soil Types = 1
Number of Soil Layers 1 = 3
[VEGETATION]
Vegetation Map File  = input/veg.bin
Number of Vegetation Types = 2
Vegetation Description 1 = Forest
Overstory Present 1  = TRUE
Understory Present 1 = TRUE
Number of Root Zones 1 = 3
Root Zone Depths 1   = 0.10 0.25 0.40
Vegetation Description 2 = Grass   # in the table, not on the map
Overstory Present 2  = FALSE
Understory Present 2 = TRUE
Number of Root Zones 2 = 3
Root Zone Depths 2   = 0.10 0.20 0.30
[OUTPUT]
Output Directory     = output/
Initial State Directory = state/
[End]
"""
SOIL_DEPTH = 1.5  # m, everywhere (also outside the basin mask)
ROOT_DEPTH = 0.75  # m, total of vegetation class 1, the only class on the map
STAMP = "10.01.1970.00.00.00"  # Model Start, as in state file names
STATE_SETS = {"Interception": 5, "Snow": 8, "Soil": 10}  # 2 canopy layers, 3 soil layers


def synthetic_dem() -> np.ndarray:
    rows, cols = np.mgrid[0:NROWS, 0:NCOLS]
    return (100.0 + 4.0 * (NROWS - 1 - rows) + 3.0 * np.abs(cols - 6)).astype("<f4")


def synthetic_mask() -> np.ndarray:
    mask = np.ones((NROWS, NCOLS), dtype="u1")
    mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = 0
    return mask


def network_text(down: dict, order: dict, outlet_text: str = "0") -> str:
    lines = ["# id order slope length class outlet [SAVE \"name\"]"]
    for sid in sorted(down):
        out = outlet_text if down[sid] is None else str(down[sid])
        save = ' SAVE "Outlet"' if down[sid] is None else ""
        lines.append(f"{sid:5d} {order[sid]:3d} {0.05:9.5f} {LENGTH[sid]:10.3f} {1:3d} "
                     f"{out:>5s}{save}")
    return "\n".join(lines) + "\n"


def map_text() -> str:
    lines = ["#  col  row   id   length  cut_h  cut_w  azimuth"]
    for sid, cells in SEGMENTS.items():
        for r, c in cells:
            lines.append(f"{c:5d} {r:4d} {sid:4d} {CELL:8.3f} {0.5:6.3f} {2.0:6.3f} {180.0:8.1f}")
    return "\n".join(lines) + "\n"


def make_synthetic_case(root: Path, down: dict | None = None, order: dict | None = None,
                        outlet_text: str = "0") -> Path:
    """Write the synthetic case under ``root`` and return its INPUT path.
    Relative paths in it resolve against ``root`` (the working directory)."""
    inp = root / "input"
    inp.mkdir(parents=True, exist_ok=True)
    synthetic_dem().tofile(inp / "dem.bin")
    synthetic_mask().tofile(inp / "mask.bin")
    np.ones((NROWS, NCOLS), dtype="u1").tofile(inp / "soil.bin")
    np.full((NROWS, NCOLS), SOIL_DEPTH, dtype="<f4").tofile(inp / "soild.bin")
    np.ones((NROWS, NCOLS), dtype="u1").tofile(inp / "veg.bin")
    (inp / "stream.network.dat").write_text(network_text(down or DOWN, order or ORDER,
                                                         outlet_text))
    (inp / "stream.map.dat").write_text(map_text())
    (inp / "stream.class.dat").write_text("1 2.0 0.5 0.04 0.0\n")
    (root / "output").mkdir(exist_ok=True)
    (root / "state").mkdir(exist_ok=True)
    for kind, sets in STATE_SETS.items():
        np.zeros((sets, NROWS, NCOLS), dtype="<f4").tofile(
            root / "state" / f"{kind}.State.{STAMP}.bin")
    (root / "state" / f"Channel.State.{STAMP}").write_text(
        "".join(f"{sid:12d} {0.0:12g}\n" for sid in sorted(SEGMENTS)))
    path = root / "INPUT.synthetic"
    path.write_text(INPUT_TEMPLATE)
    return path


@pytest.fixture
def short_tmp():
    """A temporary directory with a short path. DHSVM has room for 85 characters in front
    of a state file name (InitModelState.c:57), and pytest's own paths can be longer."""
    path = Path(tempfile.mkdtemp(prefix="wader-", dir="/tmp" if Path("/tmp").is_dir() else None))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def synthetic(tmp_path: Path) -> tuple[Path, Path]:
    """(INPUT path, working directory) of a clean synthetic case."""
    return make_synthetic_case(tmp_path), tmp_path


# ------------------------------------------------------- reference data
def _require() -> bool:
    return os.environ.get("WADER_REQUIRE_CASES", "") not in ("", "0")


def _missing(what: str) -> None:
    if _require():
        pytest.fail(f"{what} is required (WADER_REQUIRE_CASES=1) but missing")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def chiwawa() -> Path:
    """Root of the fetched Chiwawa case (cases/chiwawa/fetch.py)."""
    root = chiwawa_dir()
    if not (root / "MANIFEST.json").is_file():
        _missing(f"the fetched Chiwawa case ({root}; run cases/chiwawa/fetch.py)")
    return root


@pytest.fixture(scope="session")
def chiwawa_workdir(chiwawa: Path) -> Path:
    """A run directory two levels below the case root, as the INPUT paths require."""
    work = chiwawa / "run" / "wader-tests"
    work.mkdir(parents=True, exist_ok=True)
    return work


@pytest.fixture(scope="session")
def chiwawa_input(chiwawa: Path) -> Path:
    return chiwawa / "TestCase" / "Chiwawa" / "INPUT.Chiwawa.Baseline"


@pytest.fixture(scope="session")
def dhsvm_bin() -> Path:
    exe = dhsvm_executable()
    if exe is None or not exe.is_file():
        _missing("a DHSVM executable (scripts/build_dhsvm.sh or WADER_DHSVM)")
    return exe
