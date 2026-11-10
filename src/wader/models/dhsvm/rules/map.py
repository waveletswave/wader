"""Binary map rules (DHSVM-MAP-*).

Source references are to pnnl/DHSVM-PNNL@e6f333d (DHSVM/sourcecode/).
"""

from __future__ import annotations

from wader.core.rules import (
    CheckOutput,
    Evidence,
    Finding,
    RuleInfo,
    Severity,
    Status,
    register,
)
from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.maps import MapSpec, expected_maps, unchecked_inputs

MODEL = "dhsvm"
SRC = "DHSVM-PNNL@e6f333d "

# ------------------------------------------------------------------ MAP-001
MAP_001 = RuleInfo(
    id="DHSVM-MAP-001",
    title="Binary maps hold exactly the data DHSVM reads",
    level="L0/L1",
    severity=Severity.ERROR,
    rationale=(
        "Every binary map that the setup makes DHSVM read must exist and hold rows x "
        "columns x data sets values of the map's number type. DHSVM stops on a missing or "
        "short file (L0); for the monthly shading files it does so only when the run "
        "reaches that month. It does not notice a file that is too long: it reads the "
        "leading bytes, so a map written with another number type, grid size or number of "
        "data sets is read as different numbers (L1). The maps covered are the DEM, the "
        "basin mask, the soil and vegetation maps, the canopy gap map, snow parameter and "
        "precipitation multiplier maps, the sky view map and the shading maps; format BIN "
        "only."
    ),
    evidence=(
        Evidence(
            "Read2DMatrixBin opens the file (failure is error 3), seeks to NY x NX x size x "
            "data set number, reads NY x NX values of the variable's number type and stops "
            "with error 2 if it gets fewer. Nothing looks at bytes after the data read.",
            Status.VERIFIED_SOURCE,
            SRC + "FileIOBin.c:77-99; Files.c:38-52; SizeOfNT.c:39-69",
        ),
        Evidence(
            "Which maps are read, with which number type and how many data sets: see the "
            "source location given with each map (wader.models.dhsvm.maps). Maps are stored "
            "row by row from north to south, each row from west to east.",
            Status.VERIFIED_SOURCE,
            SRC + "InitTerrainMaps.c:88-847; InitMetMaps.c:54-74, 425-474; "
                  "InitNewMonth.c:90-112; InitConstants.c:593-742; VarID.c:49-280; "
                  "InitFileIO.c:62-66",
        ),
        Evidence(
            "Chiwawa with its DEM cut to half stops with 'Error while reading file' and exit "
            "status 2; with a DEM path that does not exist it stops with 'Cannot open file' "
            "and exit status 3. With the DEM written as 8-byte floats (twice the size) DHSVM "
            "reports no read error, initializes, and stops at the first time step with error "
            "47 ('Table index out of bounds'), which does not name the DEM.",
            Status.VERIFIED_EXPERIMENT,
            "tests/test_dhsvm_runs.py::test_dhsvm_stops_on_mutant[truncate-map], "
            "::test_dhsvm_stops_on_a_missing_map, "
            "::test_float64_dem_is_read_without_a_read_error (2026-11-10)",
        ),
        Evidence(
            "The Chiwawa case ships CanopyGap.bin with 1 byte per cell, while DHSVM reads the "
            "canopy gap map as 4-byte floats. The shipped INPUT has Canopy Gapping = FALSE "
            "and never reads the file; with Canopy Gapping = TRUE DHSVM stops with error 2.",
            Status.VERIFIED_EXPERIMENT,
            "tests/test_dhsvm_runs.py::test_chiwawa_canopy_gap_map_is_too_short (2026-11-10)",
        ),
    ),
    mutations=("truncate-map", "float64-dem"),
)


def _reads(spec: MapSpec, nrows: int, ncols: int) -> str:
    kind = "float" if spec.dtype.kind == "f" else "integer"
    sets = f" x {spec.layers} data sets" if spec.layers > 1 else ""
    return (f"{spec.expected_bytes(nrows, ncols)} bytes ({nrows} rows x {ncols} columns x "
            f"{spec.dtype.itemsize}-byte {kind}{sets})")


def _too_long_hint(spec: MapSpec, size: int, nrows: int, ncols: int) -> str:
    layer = spec.layer_bytes(nrows, ncols)
    if size % layer:
        return "the file was probably written for another grid size or number type"
    held = size // layer
    hint = f"the file holds {held} data sets of that size where DHSVM reads {spec.layers}"
    if spec.dtype.kind == "f" and held == 2 * spec.layers:
        hint += ", or its values are 8-byte floats"
    return hint


@register(MODEL, MAP_001)
def check_map_001(case: DhsvmCase, params: dict) -> CheckOutput:
    g = case.grid
    specs = expected_maps(case)
    findings: list[Finding] = []
    files = []
    counts = {"missing": 0, "too_short": 0, "too_long": 0}
    for spec in specs:
        expected = spec.expected_bytes(g.nrows, g.ncols)
        where = {"map": spec.label, "path": str(spec.path),
                 "entry": f"[{spec.entry[0]}] {spec.entry[1]}"}
        name = f"{spec.label} ({spec.path.name})"
        size = spec.path.stat().st_size if spec.path.is_file() else None
        files.append({**where, "number_type": spec.dtype.name, "data_sets": spec.layers,
                      "expected_bytes": expected, "bytes": size, "read": spec.when,
                      "source": spec.source})
        values = {"bytes": size, "expected_bytes": expected}
        if size is None:
            counts["missing"] += 1
            findings.append(Finding(
                MAP_001.id, Severity.ERROR,
                f"{name}: no such file ({spec.path}); DHSVM stops {spec.when} ('Cannot open "
                "file', exit 3)", where=where, values=values, level="L0"))
        elif size < expected:
            counts["too_short"] += 1
            findings.append(Finding(
                MAP_001.id, Severity.ERROR,
                f"{name}: {size} bytes, but DHSVM reads {_reads(spec, g.nrows, g.ncols)}; it "
                f"stops {spec.when} ('Error while reading file', exit 2)",
                where=where, values=values, level="L0"))
        elif size > expected:
            counts["too_long"] += 1
            findings.append(Finding(
                MAP_001.id, Severity.ERROR,
                f"{name}: {size} bytes, but DHSVM reads only the first "
                f"{_reads(spec, g.nrows, g.ncols)} and ignores the rest without a message; "
                f"{_too_long_hint(spec, size, g.nrows, g.ncols)}",
                where=where, values=values, level="L1"))
    skipped = unchecked_inputs(case)
    if skipped:
        findings.append(Finding(MAP_001.id, Severity.INFO,
                                "not checked by this rule: " + "; ".join(skipped)))
    bad = sum(counts.values())
    summary = f"{len(specs)} map files" + (
        " have the size DHSVM reads" if not bad else
        f": {counts['missing']} missing, {counts['too_short']} too short, "
        f"{counts['too_long']} too long")
    return CheckOutput(summary=summary, findings=findings,
                       metrics={"files": files, **counts})
