"""Model state rules (DHSVM-STA-*).

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
from wader.models.dhsvm.maps import MapSpec, state_files, unchecked_states

MODEL = "dhsvm"
SRC = "DHSVM-PNNL@e6f333d "

# ------------------------------------------------------------------ STA-001
STA_001 = RuleInfo(
    id="DHSVM-STA-001",
    title="Gridded state files hold exactly the data DHSVM restores",
    level="L0/L1",
    severity=Severity.ERROR,
    rationale=(
        "At start-up DHSVM restores the canopy interception, snow and soil state from three "
        "binary files. Each must exist under the name DHSVM builds from Initial State "
        "Directory and Model Start, and hold rows x columns x data sets 4-byte floats: 2 x "
        "canopy layers + 1 data sets for interception, 8 for snow and 2 x soil layers + 4 "
        "for soil, where the layer counts are the largest in the vegetation and soil "
        "tables. DHSVM stops on a missing or short file (L0). It does not notice a file "
        "that is too long: from a state written for more layers it takes the later "
        "variables from the wrong data sets (L1). Format BIN only."
    ),
    evidence=(
        Evidence(
            "InitModelState builds each name from the text of Initial State Directory, "
            "'Interception.State.', 'Snow.State.' or 'Soil.State.', the month, day, year, "
            "hour, minute and second of Model Start separated by dots, and '.bin'. It reads "
            "rain and snow interception for each of Veg.MaxLayers layers and the temporary "
            "interception storage; eight snow variables; soil moisture for Soil.MaxLayers + 1 "
            "layers, the surface temperature, the temperature of Soil.MaxLayers layers, the "
            "ground heat storage and the surface water. Each is one data set of 4-byte "
            "floats.",
            Status.VERIFIED_SOURCE,
            SRC + "InitModelState.c:77-80, 89-145, 166-266, 299-390; MainDHSVM.c:229-231; "
                  "InitFileIO.c:95; VarID.c:119-128, 151-182, 195-232",
        ),
        Evidence(
            "Veg.MaxLayers is the largest number of canopy layers (overstory plus understory) "
            "and Soil.MaxLayers the largest Number of Soil Layers over all classes of the "
            "vegetation and soil tables, whether the basin uses them or not.",
            Status.VERIFIED_SOURCE,
            SRC + "InitTables.c:121, 160-165, 329, 353-375",
        ),
        Evidence(
            "The files are read with Read2DMatrixBin: a missing file is error 3, a short one "
            "is error 2, and nothing looks at bytes after the data read.",
            Status.VERIFIED_SOURCE,
            SRC + "FileIOBin.c:77-99; Files.c:38-52",
        ),
        Evidence(
            "An empty or missing Initial State Directory does not fall back to the output "
            "directory: the fallback is overwritten on the next line, the prefix stays empty "
            "and DHSVM looks in the directory it is started from.",
            Status.VERIFIED_SOURCE,
            SRC + "InitDump.c:101-103",
        ),
        Evidence(
            "Chiwawa has 2 canopy layers and 3 soil layers (5, 8 and 10 data sets). With "
            "Soil.State cut to half, DHSVM stops with 'Error while reading file' and exit "
            "status 2; without Snow.State it stops with 'Cannot open file' and exit status 3. "
            "With a Soil.State written for 4 soil layers (12 data sets) it reports nothing "
            "and runs to the end, but takes soil temperatures for the ground heat storage and "
            "the surface water, and the outlet flow of the first time step is more than 100 "
            "times larger. With Initial State Directory left empty it runs if the state files "
            "are in the directory it is started from, and stops with 'Cannot open file' if "
            "they are in the output directory.",
            Status.VERIFIED_EXPERIMENT,
            "tests/test_dhsvm_runs.py::test_dhsvm_stops_on_mutant[truncate-state], "
            "::test_dhsvm_stops_on_a_missing_state_file, "
            "::test_state_written_for_more_soil_layers_floods_the_basin, "
            "::test_empty_state_directory_means_the_start_directory (2026-11-11)",
        ),
    ),
    mutations=("truncate-state", "state-extra-layer"),
)


def _holds(spec: MapSpec, size: int, nrows: int, ncols: int) -> str:
    """What a file of the wrong size seems to hold."""
    layer = spec.layer_bytes(nrows, ncols)
    if size % layer:
        return ("its size is not a whole number of data sets, so it was probably written for "
                "another grid size or number type")
    held = size // layer
    text = f"it holds {held} data sets"
    kind = spec.label.split()[0]  # the layer count each kind of state implies
    layers = {"canopy": (held - 1) / 2, "soil": (held - 4) / 2}.get(kind)
    if layers is not None and layers >= 0 and layers == int(layers):
        text += (f", as a state written for {int(layers)} {kind} "
                 f"layer{'' if layers == 1 else 's'} does")
    return text


def _prefix_note(prefix: str) -> str:
    if not prefix:
        return (" Initial State Directory is empty or not set, so DHSVM looks in the "
                "directory it is started from, not in the output directory.")
    if not prefix.endswith("/"):
        return (f" Initial State Directory = {prefix!r} does not end with '/', and DHSVM puts "
                "that text directly in front of the file name.")
    return ""


@register(MODEL, STA_001)
def check_sta_001(case: DhsvmCase, params: dict) -> CheckOutput:
    g = case.grid
    specs = state_files(case)
    note = _prefix_note(case.state_prefix)
    findings: list[Finding] = []
    files = []
    counts = {"missing": 0, "too_short": 0, "too_long": 0}
    for spec in specs:
        expected = spec.expected_bytes(g.nrows, g.ncols)
        where = {"state": spec.label, "path": str(spec.path),
                 "entry": f"[{spec.entry[0]}] {spec.entry[1]}"}
        name = f"{spec.label} ({spec.path.name})"
        size = spec.path.stat().st_size if spec.path.is_file() else None
        files.append({**where, "data_sets": spec.layers, "expected_bytes": expected,
                      "bytes": size, "source": spec.source})
        values = {"bytes": size, "expected_bytes": expected}
        if size is None:
            counts["missing"] += 1
            findings.append(Finding(
                STA_001.id, Severity.ERROR,
                f"{name}: no such file ({spec.path}); DHSVM stops when it restores the model "
                f"state ('Cannot open file', exit 3).{note}",
                where=where, values=values, level="L0"))
        elif size < expected:
            counts["too_short"] += 1
            findings.append(Finding(
                STA_001.id, Severity.ERROR,
                f"{name}: {size} bytes, but DHSVM reads {spec.reads(g.nrows, g.ncols)}; it "
                "stops when it restores the model state ('Error while reading file', exit "
                f"2). As it is, {_holds(spec, size, g.nrows, g.ncols)}",
                where=where, values=values, level="L0"))
        elif size > expected:
            counts["too_long"] += 1
            findings.append(Finding(
                STA_001.id, Severity.ERROR,
                f"{name}: {size} bytes, but DHSVM reads only the first "
                f"{spec.reads(g.nrows, g.ncols)} and ignores the rest without a message. As "
                f"it is, {_holds(spec, size, g.nrows, g.ncols)}; variables after the first "
                "layers are then taken from the wrong data sets",
                where=where, values=values, level="L1"))
    skipped = unchecked_states(case)
    if skipped:
        findings.append(Finding(STA_001.id, Severity.INFO,
                                "not checked by this rule: " + "; ".join(skipped)))
    bad = sum(counts.values())
    summary = f"{len(specs)} gridded state files" + (
        " have the size DHSVM restores" if not bad else
        f": {counts['missing']} missing, {counts['too_short']} too short, "
        f"{counts['too_long']} too long")
    return CheckOutput(summary=summary, findings=findings,
                       metrics={"files": files, **counts})
