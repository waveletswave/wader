"""Binary map rules (DHSVM-MAP-*).

Source references are to pnnl/DHSVM-PNNL@e6f333d (DHSVM/sourcecode/).
"""

from __future__ import annotations

import numpy as np

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
                f"{name}: {size} bytes, but DHSVM reads {spec.reads(g.nrows, g.ncols)}; it "
                f"stops {spec.when} ('Error while reading file', exit 2)",
                where=where, values=values, level="L0"))
        elif size > expected:
            counts["too_long"] += 1
            findings.append(Finding(
                MAP_001.id, Severity.ERROR,
                f"{name}: {size} bytes, but DHSVM reads only the first "
                f"{spec.reads(g.nrows, g.ncols)} and ignores the rest without a message; "
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


# ------------------------------------------------------------------ MAP-003
MAP_003 = RuleInfo(
    id="DHSVM-MAP-003",
    title="The soil is deeper than the root zones in every basin cell",
    level="L0",
    severity=Severity.ERROR,
    rationale=(
        "In every cell inside the basin the soil depth must be greater than the total root "
        "depth of the cell's vegetation class, which is the sum of the class's root zone "
        "depths. The root zones are the upper soil layers, and the layer below them is as "
        "thick as what is left of the soil depth, so that must be more than nothing. DHSVM "
        "tests this before it reads the stream network or the forcing and stops at the "
        "first cell that fails. A cell whose vegetation class is not in the vegetation "
        "table cannot be tested: DHSVM stops on it in the same routine, and this rule "
        "reports it until DHSVM-MAP-002 covers class values."
    ),
    evidence=(
        Evidence(
            "CheckOut adds up the root zone depths of each vegetation class in single "
            "precision (VType.TotalDepth), then visits the basin cells row by row and, at the "
            "first one whose soil depth is not greater than the total root depth of its "
            "class, prints the class, the cell and both depths and calls exit(-1). It runs "
            "after the terrain, soil and vegetation maps are read and before the network and "
            "the forcing. The layer below the root zones gets the soil depth minus the total "
            "root depth as its thickness.",
            Status.VERIFIED_SOURCE,
            SRC + "CheckOut.c:81-84, 136-148; InitTables.c:377-378, 583-585; "
                  "MainDHSVM.c:157-163; AdjustStorage.c:55-73",
        ),
        Evidence(
            "Before that, the same routine stops on the first basin cell whose vegetation "
            "class is below 1 or above the number of classes ('veg value N out of range', "
            "exit(-1)).",
            Status.VERIFIED_SOURCE,
            SRC + "CheckOut.c:42-49",
        ),
        Evidence(
            "Chiwawa has one vegetation class with root zones of 0.06, 0.13 and 0.21 m. With "
            "the soil depth of nine basin cells set to their single-precision sum, DHSVM "
            "prints 'Soil depth is 0.400000, Root depth is 0.400000' for the first of them "
            "and exits with status 255. With those cells deeper by the smallest "
            "single-precision step, it runs to the end.",
            Status.VERIFIED_EXPERIMENT,
            "tests/test_dhsvm_runs.py::test_dhsvm_stops_on_mutant[shallow-soil], "
            "::test_soil_one_step_deeper_than_the_roots_is_accepted (2026-11-11)",
        ),
    ),
    mutations=("shallow-soil",),
    params={"thin_layer_m": 0.05},
)

CAP = 20  # cells listed per finding


@register(MODEL, MAP_003)
def check_map_003(case: DhsvmCase, params: dict) -> CheckOutput:
    thin = float(params["thin_layer_m"])
    types = case.veg_types
    veg, depth, basin = case.veg_type, case.soil_depth, case.mask
    known = basin & (veg >= 1) & (veg <= len(types))
    # both sides are single-precision values held in doubles, so <= is DHSVM's comparison
    totals = np.array([0.0] + [t.total_root_depth for t in types])
    total = totals[np.where(known, veg, 0)]
    failing = known & (depth <= total)

    findings: list[Finding] = []
    classes = []
    for t in types:
        here = known & (veg == t.index)
        if not here.any():
            continue
        classes.append({"class": t.index, "description": t.description,
                        "root_zone_depths_m": list(t.root_depths),
                        "total_root_depth_m": t.total_root_depth, "cells": int(here.sum())})
        cells = np.argwhere(failing & here)  # in DHSVM's scan order
        if not len(cells):
            continue
        row, col = (int(v) for v in cells[0])
        shallowest = float(depth[failing & here].min())
        name = f"vegetation class {t.index}" + (f" ({t.description})" if t.description else "")
        findings.append(Finding(
            MAP_003.id, Severity.ERROR,
            f"{len(cells)} basin cells of {name} have a soil depth that is not greater than "
            f"the class's total root depth of {t.total_root_depth:.6g} m (shallowest "
            f"{shallowest:.6g} m; first at row {row}, column {col}); DHSVM stops before it "
            "reads the network ('Soil depth is ..., Root depth is ...', exit 255)",
            where={"vegetation_class": t.index, "n_cells": len(cells),
                   "cells": [[int(r), int(c)] for r, c in cells[:CAP]]},
            values={"total_root_depth_m": t.total_root_depth, "min_soil_depth_m": shallowest},
        ))

    unknown = basin & ~known
    n_unknown = int(unknown.sum())
    if n_unknown:
        cells = np.argwhere(unknown)
        values = sorted({int(v) for v in veg[unknown]})
        findings.append(Finding(
            MAP_003.id, Severity.ERROR,
            f"{n_unknown} basin cells have a vegetation class that is not one of the "
            f"{len(types)} classes of the vegetation table (values {values[:CAP]}; first at "
            f"row {int(cells[0][0])}, column {int(cells[0][1])}), so their soil depth was not "
            "tested; DHSVM stops on the first of them ('veg value N out of range', exit 255)",
            where={"n_cells": n_unknown, "cells": [[int(r), int(c)] for r, c in cells[:CAP]]},
            values={"vegetation_classes": values[:CAP]},
        ))

    n_known, n_failing = int(known.sum()), int(failing.sum())
    margins = (depth - total)[known]
    n_thin = int(((margins > 0) & (margins < thin)).sum())
    metrics = {
        "basin_cells": int(basin.sum()), "tested_cells": n_known, "failing_cells": n_failing,
        "untested_cells": n_unknown,
        "min_margin_m": float(margins.min()) if n_known else None,
        "cells_with_layer_below_roots_thinner_than_thin_layer_m": n_thin,
        "classes": classes,
    }
    if n_failing or n_unknown:
        summary = (f"{n_failing} of {n_known} basin cells have soil no deeper than their "
                   "root zones")
        if n_unknown:
            summary += f"; {n_unknown} more could not be tested"
    else:
        summary = (f"{n_known} basin cells: the soil is deeper than the root zones "
                   f"everywhere (by at least {float(margins.min()):.3g} m)" if n_known
                   else "no basin cells")
        if n_thin:
            summary += (f"; in {n_thin} cells the layer below the root zones is thinner than "
                        f"{thin:g} m")
    return CheckOutput(summary=summary, findings=findings, metrics=metrics)
