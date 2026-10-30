"""A DHSVM setup as DHSVM sees it: the INPUT file plus the files it names.

Relative paths in an INPUT file are opened with ``fopen`` exactly as written
(``OpenFile``, Files.c:38-50), i.e. relative to the directory DHSVM is started
from, not to the INPUT file. ``DhsvmCase`` therefore takes a working
directory; it defaults to the current directory, like DHSVM.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from wader.core.rules import InputProblem
from wader.models.dhsvm.config import ConfigError, DhsvmConfig
from wader.models.dhsvm.io import (
    DEM_DTYPE,
    MASK_DTYPE,
    MapReadError,
    StreamMap,
    StreamNetwork,
    read_bin_map,
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
    def network(self) -> StreamNetwork:
        with self._guard("stream network"):
            return read_stream_network(self.path("ROUTING", "Stream Network File"))

    @cached_property
    def stream_map(self) -> StreamMap:
        with self._guard("stream map"):
            return read_stream_map(self.path("ROUTING", "Stream Map File"))

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
        except InputProblem:
            pass
        return out


class _Guard:
    """Turn reading/parsing errors into InputProblem with some context."""

    def __init__(self, what: str) -> None:
        self.what = what

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None or isinstance(exc, InputProblem):
            return False
        if isinstance(exc, (ConfigError, MapReadError, OSError, ValueError)):
            raise InputProblem(f"{self.what}: {exc}") from exc
        return False
