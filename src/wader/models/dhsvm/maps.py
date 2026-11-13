"""Which binary maps DHSVM reads for a setup, and how many bytes of each.

``Read2DMatrixBin`` (FileIOBin.c:77-99) opens the file (a missing file is
error 3, Files.c:49-50), seeks to ``NY * NX * size * NDataSet`` and reads
``NY * NX`` values; it stops with error 2 if fewer are read and never looks at
what follows. A file must therefore hold at least ``layers`` data sets of
``NY * NX`` values of the variable's number type (VarID.c, SizeOfNT.c:39-69),
and anything beyond that is ignored.

``expected_maps`` lists the maps this module knows how to predict, each with
the source location of the read. ``unchecked_inputs`` names gridded inputs
that the setup enables but that are not predicted yet. ``state_files`` lists
the gridded state files that DHSVM restores at start-up, which it reads with
the same routine.

All source references are to pnnl/DHSVM-PNNL@e6f333d, DHSVM/sourcecode/
(verified-source).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wader.core.rules import InputProblem
from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.config import c_float

F4 = np.dtype("<f4")  # NC_FLOAT
U1 = np.dtype("u1")  # NC_BYTE


@dataclass(frozen=True)
class MapSpec:
    label: str  # e.g. "DEM"
    entry: tuple[str, str]  # INPUT (section, key) that names the file or its path stem
    path: Path
    dtype: np.dtype
    layers: int  # data sets DHSVM reads from the file
    source: str  # where DHSVM reads it
    when: str = "at start-up"

    def expected_bytes(self, nrows: int, ncols: int) -> int:
        return nrows * ncols * self.dtype.itemsize * self.layers

    def layer_bytes(self, nrows: int, ncols: int) -> int:
        return nrows * ncols * self.dtype.itemsize

    def reads(self, nrows: int, ncols: int) -> str:
        """What DHSVM reads from the file, in words."""
        kind = "float" if self.dtype.kind == "f" else "integer"
        sets = f" x {self.layers} data sets" if self.layers > 1 else ""
        return (f"{self.expected_bytes(nrows, ncols)} bytes ({nrows} rows x {ncols} columns x "
                f"{self.dtype.itemsize}-byte {kind}{sets})")


# Snow parameters that take a constant or, if the value is not a number, a map
# file (InitConstants.c:593-728; all NC_FLOAT, VarID.c:261-278).
_SNOW_PARAMETERS = (
    ("Rain Threshold", "InitConstants.c:619-634"),
    ("Snow Threshold", "InitConstants.c:636-649"),
    ("Albedo Accumulation Lambda", "InitConstants.c:651-664"),
    ("Albedo Melting Lambda", "InitConstants.c:667-680"),
    ("Albedo Accumulation Min", "InitConstants.c:683-696"),
    ("Albedo Melting Min", "InitConstants.c:698-711"),
    ("Fresh Snow Albedo", "InitConstants.c:714-728"),
)


def _months(case: DhsvmCase) -> list[int]:
    """Calendar months DHSVM enters between Model Start and Model End (both
    included): ``InitNewMonth`` runs at start-up and at the first step of each
    month (MainDHSVM.c:233, 279-281)."""
    start, end = case.period
    if end < start:
        raise InputProblem("[TIME] Model End is before Model Start")
    months, year, month = [], start.year, start.month
    while (year, month) <= (end.year, end.month) and len(months) < 12:
        months.append(month)
        year, month = (year, month + 1) if month < 12 else (year + 1, 1)
    return months


def _shadow_layers(steps_per_day: int) -> int:
    """Data sets read from each monthly shading file (InitNewMonth.c:98-106): one per
    step of the day; with more than 24 steps, set ``round(i / (NDaySteps / 24))`` in
    integer arithmetic."""
    if steps_per_day <= 24:
        return steps_per_day
    return (steps_per_day - 1) // (steps_per_day // 24) + 1


def max_soil_layers(case: DhsvmCase) -> int:
    """``Soil->MaxLayers``: the largest ``Number of Soil Layers i`` over the soil types
    (InitTables.c:105-108, 160-165)."""
    cfg = case.config
    with case._guard("[SOILS]"):
        n_types = cfg.get_int("SOILS", "Number of Soil Types")
        return max((cfg.get_int("SOILS", f"Number of Soil Layers {i}")
                    for i in range(1, n_types + 1)), default=0)


def expected_maps(case: DhsvmCase) -> list[MapSpec]:
    """The binary maps DHSVM reads for this setup (format BIN)."""
    case._require_bin()
    cfg = case.config
    specs: list[MapSpec] = []

    def required(label: str, section: str, key: str, dtype: np.dtype, source: str) -> None:
        specs.append(MapSpec(label, (section, key), case.path(section, key), dtype, 1, source))

    def optional(label: str, section: str, key: str, dtype: np.dtype, layers: int,
                 source: str) -> None:
        # the default is "none"; any name starting with "none" means no map
        value = cfg.get(section, key, "none")
        if value and value[:4] != "none":
            specs.append(MapSpec(label, (section, key), case.path(section, key), dtype,
                                 layers, source))

    required("DEM", "TERRAIN", "DEM File", F4, "InitTerrainMaps.c:88-95; VarID.c:51-53")
    required("basin mask", "TERRAIN", "Basin Mask File", U1,
             "InitTerrainMaps.c:117-123; VarID.c:54-55")
    required("soil type", "SOILS", "Soil Map File", U1,
             "InitTerrainMaps.c:219-225; VarID.c:56-57")
    required("soil depth", "SOILS", "Soil Depth File", F4,
             "InitTerrainMaps.c:250-256; VarID.c:58-60")
    optional("lateral conductivity", "SOILS", "Soil Conductivity Map File", F4, 1,
             "InitTerrainMaps.c:279-288; VarID.c:76-78")
    for label, key, source in (
        ("soil field capacity", "Soil Field Capacity File",
         "InitTerrainMaps.c:338-346; VarID.c:82-84"),
        ("soil porosity", "Soil Porosity Map File", "InitTerrainMaps.c:418-426; VarID.c:79-81"),
    ):
        value = cfg.get("SOILS", key, "none")
        if value and value[:4] != "none":  # one data set per soil layer (Soil->MaxLayers)
            specs.append(MapSpec(label, ("SOILS", key), case.path("SOILS", key), F4,
                                 max_soil_layers(case), source))
    required("vegetation type", "VEGETATION", "Vegetation Map File", U1,
             "InitTerrainMaps.c:570-575; VarID.c:61-63")
    optional("vegetation fractional cover", "VEGETATION", "Vegetation FC Map File", F4, 1,
             "InitTerrainMaps.c:600-608; VarID.c:70-72")
    optional("vegetation LAI", "VEGETATION", "Vegetation LAI Map File", F4, 12,
             "InitTerrainMaps.c:695-718; VarID.c:73-75")
    if case.flag("Canopy Gapping"):  # InitTerrainMaps.c:46-47
        required("canopy gap", "VEGETATION", "Canopy Gap Map File", F4,
                 "InitTerrainMaps.c:836-847; VarID.c:67-69")

    for key, source in _SNOW_PARAMETERS:
        value = cfg.get("CONSTANTS", key)
        if value and c_float(value) is None:  # not a number: DHSVM takes it as a map file
            specs.append(MapSpec(f"{key.lower()} (snow parameter)", ("CONSTANTS", key),
                                 case.path("CONSTANTS", key), F4, 1,
                                 f"{source}; InitParameterMaps.c:50-56"))
    value = cfg.get("CONSTANTS", "Precipitation Multiplier Map")
    if value and c_float(value) is None:
        specs.append(MapSpec("precipitation multiplier", ("CONSTANTS",
                                                           "Precipitation Multiplier Map"),
                             case.path("CONSTANTS", "Precipitation Multiplier Map"), F4, 1,
                             "InitConstants.c:731-742; InitMetMaps.c:468-474; VarID.c:94-96"))

    if case.flag("Shading"):  # InitConstants.c:285-290, 386-396
        required("sky view", "OPTIONS", "Skyview data path", F4,
                 "InitMetMaps.c:63-64, 73-74, 425-430; VarID.c:148-150")
        with case._guard("[OPTIONS] shading files"):
            stem = cfg.require("OPTIONS", "Shading data path").value
            ext = cfg.require("OPTIONS", "Shading data extension").value
        layers = _shadow_layers(case.steps_per_day)
        for month in _months(case):
            name = Path(f"{stem}.{month:02d}.{ext}")  # InitNewMonth.c:92-93
            specs.append(MapSpec(
                f"shading, month {month:02d}", ("OPTIONS", "Shading data path"),
                name if name.is_absolute() else case.workdir / name, U1, layers,
                "InitNewMonth.c:90-112; VarID.c:145-147",
                when="at start-up" if month == case.period[0].month
                else f"when the run reaches month {month:02d}",
            ))
    return specs


def unchecked_inputs(case: DhsvmCase) -> list[str]:
    """Gridded inputs this setup enables that ``expected_maps`` does not cover."""
    out = []
    if case.flag("PRISM"):
        out.append("monthly PRISM maps (InitNewMonth.c:64-72)")
    if case.flag("MM5"):
        out.append("MM5 fields (InitNewMonth.c:199)")
    else:
        if case.option("Precipitation Source")[:5] == "RADAR":
            out.append("radar precipitation (ReadRadarMap.c:52)")
        if case.option("Wind Source")[:5] == "MODEL":
            out.append("wind model maps (InitMetMaps.c:269-275)")
        if case.option("Precipitation lapse rate")[:3] == "MAP":
            out.append("precipitation lapse rate map (InitMetMaps.c:346-348)")
    try:
        if case.network_reason and "UNIT" in case.network_reason:
            out.append("travel time map (InitUnitHydrograph.c:84)")
    except InputProblem:
        pass
    return out


# ------------------------------------------------------------------ state files
STATE_DIRECTORY = ("OUTPUT", "Initial State Directory")


def max_veg_layers(case: DhsvmCase) -> int:
    """``Veg->MaxLayers``: the largest number of canopy layers over the vegetation types
    of the table, whether the basin uses them or not (InitTables.c:329, 373-375)."""
    return max((t.layers for t in case.veg_types), default=0)


def state_stamp(case: DhsvmCase) -> str:
    """The date in the names of the state files DHSVM restores: month, day, year, hour,
    minute and second of Model Start (InitModelState.c:77-78; ChannelState.c:64-65)."""
    t = case.period[0]
    return f"{t.month:02d}.{t.day:02d}.{t.year:02d}.{t.hour:02d}.{t.minute:02d}.{t.second:02d}"


def state_files(case: DhsvmCase) -> list[MapSpec]:
    """The gridded state files DHSVM restores at start-up (format BIN).

    Each name is the text of ``[OUTPUT] Initial State Directory`` followed directly by
    ``<Kind>.State.<date>.bin`` (InitModelState.c:80, 166, 299; InitFileIO.c:95). Every
    variable is a 4-byte float (VarID.c:119-128, 151-182, 195-232) and is one data set.
    """
    case._require_bin()
    stamp = state_stamp(case)
    n_veg, n_soil = max_veg_layers(case), max_soil_layers(case)

    def spec(label: str, kind: str, layers: int, source: str) -> MapSpec:
        name = f"{case.state_prefix}{kind}.State.{stamp}.bin"
        return MapSpec(label, STATE_DIRECTORY, case.start_path(name), F4, layers, source)

    return [
        # rain and snow interception per canopy layer, then the temporary storage
        spec("canopy interception state", "Interception", 2 * n_veg + 1,
             "InitModelState.c:77-145"),
        # snow flag, last snowfall, water equivalent, pack water, pack temperature,
        # surface water, surface temperature, cold content
        spec("snow state", "Snow", 8, "InitModelState.c:166-266"),
        # moisture of each soil layer and of the layer below them, surface temperature,
        # temperature of each soil layer, ground heat storage, surface water
        spec("soil state", "Soil", 2 * n_soil + 4, "InitModelState.c:299-390"),
    ]


def unchecked_states(case: DhsvmCase) -> list[str]:
    """State files DHSVM also reads at start-up that ``state_files`` does not cover."""
    stamp = state_stamp(case)
    try:
        reason = case.network_reason
    except InputProblem:
        return []
    if reason is None:
        return [f"{case.state_prefix}Channel.State.{stamp}, a table with the water stored in "
                "each segment (ChannelState.c:64-67)"]
    if "UNIT" in reason:
        return [f"{case.state_prefix}Hydrograph.State.{stamp} (InitModelState.c:429-435)"]
    return []
