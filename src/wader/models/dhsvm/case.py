"""A DHSVM setup as DHSVM sees it: the INPUT file plus the files it names.

Relative paths in an INPUT file are opened with ``fopen`` exactly as written
(``OpenFile``, Files.c:38-50), i.e. relative to the directory DHSVM is started
from, not to the INPUT file. ``DhsvmCase`` therefore takes a working
directory; it defaults to the current directory, like DHSVM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np

from wader.core.rules import InputProblem, NotApplicable
from wader.models.dhsvm.config import ConfigError, DhsvmConfig, c_floats, scan_date
from wader.models.dhsvm.io import (
    DEM_DTYPE,
    MASK_DTYPE,
    SOIL_DEPTH_DTYPE,
    VEG_TYPE_DTYPE,
    MapReadError,
    StreamClasses,
    StreamMap,
    StreamNetwork,
    read_bin_map,
    read_stream_classes,
    read_stream_map,
    read_stream_network,
)


@dataclass(frozen=True)
class Grid:
    nrows: int  # [AREA] Number of Rows -> Map->NY (InitConstants.c:479)
    ncols: int  # [AREA] Number of Columns -> Map->NX (InitConstants.c:482)
    cellsize: float  # [AREA] Grid spacing -> Map->DY = Map->DX (InitConstants.c:485-488)
    north: float  # [AREA] Extreme North -> Map->Yorig, north edge (InitConstants.c:460)
    west: float  # [AREA] Extreme West -> Map->Xorig, west edge (InitConstants.c:463)


@dataclass(frozen=True)
class VegType:
    """One class of the [VEGETATION] table, as far as wader uses it."""

    index: int  # class number as in the vegetation map, from 1
    description: str
    layers: int  # NVegLayers: overstory plus understory (InitTables.c:353-373)
    root_depths: tuple[float, ...]  # Root Zone Depths, single precision (InitTables.c:583-585)
    total_root_depth: float  # VType.TotalDepth, their single-precision sum (CheckOut.c:81-84)


class DhsvmCase:
    def __init__(self, input_path: str | Path, workdir: str | Path | None = None) -> None:
        self.input_path = Path(input_path)
        self.workdir = Path(workdir) if workdir is not None else Path.cwd()
        if not self.workdir.is_dir():
            raise InputProblem(f"working directory {self.workdir} does not exist")
        try:
            self.config = DhsvmConfig.read(self.input_path)
        except OSError as exc:
            raise InputProblem(f"cannot read DHSVM input file {self.input_path}: {exc}") from exc

    # ------------------------------------------------------------- helpers
    def path(self, section: str, key: str) -> Path:
        """Path named by ``[section] key``, resolved like DHSVM resolves it."""
        try:
            value = self.config.require(section, key).value
        except ConfigError as exc:
            raise InputProblem(str(exc)) from exc
        p = Path(value)
        return p if p.is_absolute() else self.workdir / p

    def _guard(self, what: str):
        return _Guard(what)

    def option(self, key: str, default: str = "") -> str:
        """Value of ``[OPTIONS] key`` as DHSVM compares it: a case-sensitive prefix test
        on the text (e.g. ``strncmp(value, "TRUE", 4)``, InitConstants.c:142-374)."""
        return self.config.get("OPTIONS", key, default) or ""

    def flag(self, key: str) -> bool:
        """Whether ``[OPTIONS] key`` reads as TRUE. A missing or malformed entry is
        treated as FALSE here; DHSVM itself stops on it (ReportError 51)."""
        return self.option(key)[:4] == "TRUE"

    # ---------------------------------------------------------- properties
    @cached_property
    def grid(self) -> Grid:
        with self._guard("[AREA]"):
            c = self.config
            return Grid(
                nrows=c.get_int("AREA", "Number of Rows"),
                ncols=c.get_int("AREA", "Number of Columns"),
                cellsize=c.get_float("AREA", "Grid spacing"),
                north=c.get_float("AREA", "Extreme North"),
                west=c.get_float("AREA", "Extreme West"),
            )

    @cached_property
    def file_format(self) -> str:
        """``BIN``, ``NETCDF`` or ``BYTESWAP``; only the first 3 characters are compared
        (InitConstants.c:142-149)."""
        with self._guard("[OPTIONS] Format"):
            value = self.config.require("OPTIONS", "Format").value
        for name in ("BIN", "NETCDF", "BYTESWAP"):
            if value[:3] == name[:3]:
                return name
        raise InputProblem(f"[OPTIONS] Format = {value!r} is not BIN, NETCDF or BYTESWAP")

    @cached_property
    def outside_basin_value(self) -> int:
        """Mask value meaning 'outside the basin' (InitConstants.c:549, settings.h:27)."""
        with self._guard("[CONSTANTS] Outside Basin Value"):
            return self.config.get_uchar("CONSTANTS", "Outside Basin Value")

    def _require_bin(self) -> None:
        if self.file_format != "BIN":
            raise InputProblem(
                f"map format {self.file_format} is not supported yet (wader reads BIN only)"
            )

    @cached_property
    def dem(self) -> np.ndarray:
        self._require_bin()
        g = self.grid
        with self._guard("DEM"):
            m = read_bin_map(self.path("TERRAIN", "DEM File"), g.nrows, g.ncols, DEM_DTYPE)
        return m.data.astype("float64")

    @cached_property
    def mask(self) -> np.ndarray:
        """Boolean in-basin mask: ``INBASIN(x) = x != OUTSIDEBASIN`` (settings.h:27)."""
        self._require_bin()
        g = self.grid
        with self._guard("basin mask"):
            m = read_bin_map(self.path("TERRAIN", "Basin Mask File"), g.nrows, g.ncols,
                             MASK_DTYPE)
        return m.data != self.outside_basin_value

    @cached_property
    def soil_depth(self) -> np.ndarray:
        """Total soil depth in m (Soil.Depth, VarID.c:58-60; InitTerrainMaps.c:250-275)."""
        self._require_bin()
        g = self.grid
        with self._guard("soil depth"):
            m = read_bin_map(self.path("SOILS", "Soil Depth File"), g.nrows, g.ncols,
                             SOIL_DEPTH_DTYPE)
        return m.data.astype("float64")

    @cached_property
    def veg_type(self) -> np.ndarray:
        """Vegetation class of each cell (Veg.Type, VarID.c:61-63;
        InitTerrainMaps.c:570-575)."""
        self._require_bin()
        g = self.grid
        with self._guard("vegetation type"):
            m = read_bin_map(self.path("VEGETATION", "Vegetation Map File"), g.nrows, g.ncols,
                             VEG_TYPE_DTYPE)
        return m.data

    @cached_property
    def veg_types(self) -> list[VegType]:
        """The classes of the [VEGETATION] table: canopy layers and root zones
        (InitTables.c:314-316, 353-378, 583-585; CheckOut.c:81-84)."""
        cfg = self.config
        with self._guard("[VEGETATION]"):
            n_types = cfg.get_int("VEGETATION", "Number of Vegetation Types")
        out = []
        for i in range(1, n_types + 1):
            with self._guard(f"[VEGETATION] class {i}"):
                layers = 0
                for key in ("Overstory Present", "Understory Present"):
                    value = cfg.require("VEGETATION", f"{key} {i}").value
                    if value[:4] == "TRUE":
                        layers += 1
                    elif value[:5] != "FALSE":
                        raise InputProblem(
                            f"[VEGETATION] {key} {i} = {value!r} is not TRUE or FALSE; DHSVM "
                            "stops (error 51, InitTables.c:355-371)")
                n_zones = cfg.get_int("VEGETATION", f"Number of Root Zones {i}")
                entry = cfg.require("VEGETATION", f"Root Zone Depths {i}")
            depths = c_floats(entry.value, n_zones)
            if depths is None:
                raise InputProblem(
                    f"line {entry.lineno}: [VEGETATION] Root Zone Depths {i} = {entry.value!r} "
                    f"is not {n_zones} numbers (Number of Root Zones {i}); DHSVM stops "
                    "(error 51, InitTables.c:583-585)")
            singles = [np.float32(d) for d in depths]
            total = np.float32(0.0)
            for d in singles:  # accumulated in single precision, as in CheckOut.c:81-84
                total = np.float32(total + d)
            out.append(VegType(
                index=i,
                description=cfg.get("VEGETATION", f"Vegetation Description {i}", "") or "",
                layers=layers,
                root_depths=tuple(float(d) for d in singles),
                total_root_depth=float(total),
            ))
        return out

    # ------------------------------------------------ output and state files
    def start_path(self, name: str) -> Path:
        """A file name that DHSVM builds itself, resolved against the directory DHSVM is
        started from."""
        p = Path(name)
        return p if p.is_absolute() else self.workdir / p

    @cached_property
    def output_prefix(self) -> str:
        """``[OUTPUT] Output Directory`` exactly as written. DHSVM puts it in front of
        every output file name without adding a separator (``"%sAggregated.Values"``,
        InitDump.c:94, 137-146)."""
        return self.config.get("OUTPUT", "Output Directory", "") or ""

    @cached_property
    def state_prefix(self) -> str:
        """Prefix of the state files read at start-up: ``[OUTPUT] Initial State
        Directory`` exactly as written (InitDump.c:103; MainDHSVM.c:221, 229-231). An
        empty or missing entry gives an empty prefix, i.e. the directory DHSVM is started
        from: the fallback to the output directory (InitDump.c:101-102) is overwritten by
        the unconditional copy on line 103."""
        return self.config.get("OUTPUT", "Initial State Directory", "") or ""

    # -------------------------------------------------------------- time
    @cached_property
    def period(self) -> tuple[datetime, datetime]:
        """``[TIME] Model Start`` and ``Model End`` (InitConstants.c:517-521)."""
        out = []
        for key in ("Model Start", "Model End"):
            with self._guard(f"[TIME] {key}"):
                entry = self.config.require("TIME", key)
            date = scan_date(entry.value)
            if date is None:
                raise InputProblem(f"line {entry.lineno}: [TIME] {key} = {entry.value!r} is "
                                   "not a date DHSVM accepts (MM/DD/YYYY-HH)")
            out.append(date)
        return out[0], out[1]

    @cached_property
    def steps_per_day(self) -> int:
        """``NDaySteps`` = 86400 / time step in seconds (InitConstants.c:513-515,
        Calendar.c:403-405)."""
        with self._guard("[TIME] Time Step"):
            hours = np.float32(self.config.get_float("TIME", "Time Step"))  # a C float
            seconds = int(hours * np.float32(3600))
        if seconds <= 0 or 86400 % seconds:
            raise InputProblem(f"[TIME] Time Step of {seconds} s does not divide a day; "
                               "DHSVM stops (Calendar.c:403-405)")
        return 86400 // seconds

    # ----------------------------------------------------- channel network
    @cached_property
    def network_reason(self) -> str | None:
        """Why DHSVM reads no stream files for this setup, or None if it reads them.

        ``InitChannel`` runs only if ``Extent`` is not POINT and ``Flow Routing`` is
        NETWORK (InitConstants.c:153-161, 213-222; MainDHSVM.c:169-170), and it skips
        the stream files if the class file name starts with ``none``
        (DHSVMChannel.c:73).
        """
        with self._guard("[OPTIONS]"):
            extent = self.config.require("OPTIONS", "Extent").value
            if extent[:5] == "POINT":
                return "[OPTIONS] Extent = POINT"
            if extent[:5] != "BASIN":
                raise InputProblem(f"[OPTIONS] Extent = {extent!r} is not POINT or BASIN")
            routing = self.config.require("OPTIONS", "Flow Routing").value
            if routing[:4] == "UNIT":
                return f"[OPTIONS] Flow Routing = {routing}"
            if routing[:7] != "NETWORK":
                raise InputProblem(f"[OPTIONS] Flow Routing = {routing!r} is not NETWORK or "
                                   "UNIT_HYDROGRAPH")
        if (self.config.get("ROUTING", "Stream Class File") or "")[:4] == "none":
            return "[ROUTING] Stream Class File = none"
        return None

    def _require_network(self) -> None:
        if self.network_reason is not None:
            raise NotApplicable(f"DHSVM reads no stream network here: {self.network_reason}")

    @cached_property
    def network(self) -> StreamNetwork:
        self._require_network()
        with self._guard("stream network"):
            return read_stream_network(self.path("ROUTING", "Stream Network File"))

    @cached_property
    def stream_map(self) -> StreamMap:
        self._require_network()
        with self._guard("stream map"):
            return read_stream_map(self.path("ROUTING", "Stream Map File"))

    @cached_property
    def classes(self) -> StreamClasses:
        self._require_network()
        with self._guard("stream classes"):
            return read_stream_classes(self.path("ROUTING", "Stream Class File"))

    @cached_property
    def cells(self) -> dict[int, np.ndarray]:
        """(row, col) stream-map cells of each segment that lie inside the grid."""
        g = self.grid
        return self.stream_map.cells_by_segment(g.nrows, g.ncols)

    def summary(self) -> dict[str, str]:
        out: dict[str, str] = {}
        try:
            g = self.grid
            out["grid"] = f"{g.nrows} rows x {g.ncols} columns, {g.cellsize:g} m cells"
        except InputProblem:
            pass
        try:
            out["stream network"] = (
                f"{len(self.network.segments)} segments, {len(self.stream_map)} map records"
            )
        except (InputProblem, NotApplicable):
            pass
        return out


class _Guard:
    """Turn reading/parsing errors into InputProblem with some context."""

    def __init__(self, what: str) -> None:
        self.what = what

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None or isinstance(exc, (InputProblem, NotApplicable)):
            return False
        if isinstance(exc, (ConfigError, MapReadError, OSError, ValueError)):
            raise InputProblem(f"{self.what}: {exc}") from exc
        return False
