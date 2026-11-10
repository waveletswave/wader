"""Stream-network rules (DHSVM-NET-*).

NET-001, NET-007 and NET-010 each cover one of the three channel files (network,
map, classes): everything that makes DHSVM reject the file, plus the records it
accepts and then misuses. NET-003 checks the links between segments.

The topology rules (NET-003, NET-004, NET-005, NET-006) evaluate the network
the file author intended: an outlet ID <= 0 means "leaves the network". NET-013
reports where DHSVM reads the outlet field differently from that intent.

Source references are to pnnl/DHSVM-PNNL@e6f333d (DHSVM/sourcecode/ unless a
longer path is given).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from wader.core.rules import (
    CheckOutput,
    Evidence,
    Finding,
    InputProblem,
    NotApplicable,
    RuleInfo,
    Severity,
    Status,
    register,
)
from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.io import ID_MAX, TableNote, as_c_float

MODEL = "dhsvm"
SRC = "DHSVM-PNNL@e6f333d "
CAP = 20  # findings listed per kind of problem; the rest is counted in one more finding

_ROUTES_BY_OUTLET_FIELD = Evidence(
    "DHSVM links a segment to the segment named in its outlet field and routes in passes "
    "of increasing order; it never compares elevations or positions of linked segments.",
    Status.VERIFIED_SOURCE,
    SRC + "channel.c:458-470 (outlet lookup), channel.c:491-552 (routing)",
)
_LEGACY_RUNS = Evidence(
    "A network with reversed links, long jumps and a mid-basin outlet runs without any "
    "DHSVM error (Toolkit legacy network on Chiwawa: outlet NSE 0.98, interior reaches "
    "85-99.7% low).",
    Status.VERIFIED_EXPERIMENT,
    "project brief section 6; DHSVM_tools_comparison.md section 2",
)
_MUTANTS_RUN = Evidence(
    "The reverse-network, jump and force-wire mutants of the PNNL Chiwawa network run in "
    "DHSVM with exit status 0. The reverse-network mutant (64 links flipped) gives the same "
    "warnings and mass balance error as the original and a network outflow within 0.2%, "
    "while the outlet hydrograph (segment 494) falls from 10.19 to 0.01 m3/s (October 1970).",
    Status.VERIFIED_EXPERIMENT,
    "tests/test_dhsvm_runs.py::test_dhsvm_runs_mutants_without_complaint; "
    "scripts/demo_reverse_network.sh (2026-10-30)",
)


_TABLE_ERRORS = Evidence(
    "In a channel table file, a missing required field, a required field of the wrong "
    "type, an unknown keyword or a field beyond the last one is a table error. With any "
    "table error DHSVM discards the whole file. A field that fits none of the remaining "
    "optional fields only prints 'unable to match field': it is dropped and the record "
    "is kept.",
    Status.VERIFIED_SOURCE,
    SRC + "tableio.lex:201-286, 297-365; channel.c:220-225, 472-483; "
          "channel_grid.c:336-351",
)
_TABLE_ERRORS_RUN = Evidence(
    "Chiwawa with an extra number at the end of one network record prints 'unable to "
    "match field' and gives byte-identical output.",
    Status.VERIFIED_EXPERIMENT,
    "tests/test_dhsvm_runs.py::test_unmatched_field_is_only_a_message (2026-11-10)",
)
_EXIT_5 = Evidence(
    "If the stream class, network or map file is discarded, InitChannel stops DHSVM with "
    "error 5 ('Error reading input in file').",
    Status.VERIFIED_SOURCE,
    SRC + "DHSVMChannel.c:73-90; ReportError.c:97-101",
)
_SHORT_IDS = Evidence(
    "Segment and class IDs are stored in an unsigned short and looked up through unsigned "
    "short arguments, so an ID above 65535 is used modulo 65536. A lookup returns the first "
    "record with that ID.",
    Status.VERIFIED_SOURCE,
    SRC + "channel.h:16, 26, 61; channel.c:71-79, 284-299, 383, 459-462; "
          "channel_grid.c:279-281",
)
_SHORT_IDS_RUN = Evidence(
    "Chiwawa with the outlet ID 3 of segment 1 written as 65539 gives byte-identical "
    "output: DHSVM reads 65539 as segment 3.",
    Status.VERIFIED_EXPERIMENT,
    "tests/test_dhsvm_runs.py::test_outlet_id_above_65535_wraps (2026-11-10)",
)


def _down(case: DhsvmCase) -> dict[int, int | None]:
    return case.network.intended_downstream()


def _table_error_findings(rule_id: str, notes: list[TableNote], what: str) -> list[Finding]:
    """One L0 finding per table error DHSVM counts for the file."""
    return [
        Finding(rule_id, Severity.ERROR,
                f"line {n.lineno}: {n.message}; DHSVM rejects the {what} (exit 5)",
                where={"line": n.lineno}, level="L0")
        for n in notes if n.kind == "error"
    ]


def _ignored_field_findings(rule_id: str, notes: list[TableNote]) -> list[Finding]:
    """L1 warnings for fields DHSVM drops with only a message (the record is kept)."""
    found = [
        Finding(rule_id, Severity.WARNING, f"line {n.lineno}: {n.message}",
                where={"line": n.lineno}, level="L1")
        for n in notes if n.kind == "ignored"
    ]
    return _capped(found, "fields that DHSVM drops")


def _capped(findings: list[Finding], what: str) -> list[Finding]:
    """The first CAP findings of one kind, plus one that counts the rest."""
    if len(findings) <= CAP:
        return findings
    first = findings[0]
    rest = len(findings) - CAP
    return findings[:CAP] + [Finding(first.rule_id, first.severity,
                                     f"... and {rest} more {what}",
                                     values={"count": rest}, level=first.level)]


def _ids(values, limit: int = 8) -> str:
    """'3, 7, 12, ...' for a list of IDs or line numbers."""
    values = list(values)
    text = ", ".join(str(v) for v in values[:limit])
    return text + (", ..." if len(values) > limit else "")


# ------------------------------------------------------------------ NET-001
NET_001 = RuleInfo(
    id="DHSVM-NET-001",
    title="DHSVM accepts every stream network record",
    level="L0/L1",
    severity=Severity.ERROR,
    rationale=(
        "Every record of the stream network file must be readable as ID, order, slope, "
        "length, class and optional outlet ID, with an ID and an order of at least 1 and a "
        "slope and a length above 0; otherwise DHSVM discards the network and stops (L0). "
        "IDs must be unique and at most 65535: DHSVM keeps them in 16 bits and resolves "
        "every reference to the first record with that ID, without a message (L1)."
    ),
    evidence=(
        Evidence(
            "channel_read_network counts an error for a segment ID of 0, an order of 0, a "
            "slope <= 0 or a length <= 0 (table reals are C floats), and returns no network "
            "if there is any error. A file without records also returns no network.",
            Status.VERIFIED_SOURCE,
            SRC + "channel.c:354-424, 472-485; tableio.h:32-35",
        ),
        _TABLE_ERRORS,
        _TABLE_ERRORS_RUN,
        _EXIT_5,
        Evidence(
            "Chiwawa with one segment's slope set to 0 stops with 'channel slope (0.000000) "
            "invalid' and exit status 5; a length of 0, an order of 0 and an ID of 0 stop it "
            "the same way.",
            Status.VERIFIED_EXPERIMENT,
            "tests/test_dhsvm_runs.py::test_dhsvm_stops_on_mutant[zero-slope], "
            "::test_dhsvm_rejects_segment_field (2026-11-10)",
        ),
        _SHORT_IDS,
        _SHORT_IDS_RUN,
    ),
    mutations=("zero-slope",),
)


@register(MODEL, NET_001)
def check_net_001(case: DhsvmCase, params: dict) -> CheckOutput:
    net = case.network
    findings = _table_error_findings(NET_001.id, net.notes, "network")
    n_table = len(findings)
    ignored = _ignored_field_findings(NET_001.id, net.notes)
    if not net.segments and not findings:
        findings.append(Finding(NET_001.id, Severity.ERROR,
                                "the file holds no segment record; DHSVM rejects the network "
                                "(exit 5)", level="L0"))
    rejected: list[Finding] = []
    for s in net.segments:
        problems = []
        if s.id == 0:
            problems.append("segment ID 0 ('channel id invalid')")
        if s.order == 0:
            problems.append("order 0 ('channel order invalid')")
        if as_c_float(s.slope) <= 0:
            problems.append(f"slope {s.slope:g} is not above 0 ('channel slope invalid')")
        if as_c_float(s.length) <= 0:
            problems.append(f"length {s.length:g} is not above 0 ('channel length invalid')")
        if problems:
            rejected.append(Finding(
                NET_001.id, Severity.ERROR,
                f"segment {s.id} (line {s.lineno}): {'; '.join(problems)}; DHSVM rejects the "
                "network (exit 5)",
                where={"segment": s.id, "line": s.lineno},
                values={"order": s.order, "slope": s.slope, "length": s.length}, level="L0",
            ))
    wrapped = [
        Finding(NET_001.id, Severity.ERROR,
                f"segment {s.id} (line {s.lineno}): the ID exceeds 65535; DHSVM keeps IDs in "
                f"16 bits and uses {s.id % 65536} instead",
                where={"segment": s.id, "line": s.lineno},
                values={"read_as": s.id % 65536}, level="L1")
        for s in net.segments if s.id > ID_MAX
    ]
    lines_by_id: dict[int, list[int]] = defaultdict(list)
    for s in net.segments:
        lines_by_id[s.id].append(s.lineno)
    duplicates = [
        Finding(NET_001.id, Severity.ERROR,
                f"segment ID {sid} is used by {len(lines)} records (lines {_ids(lines)}); "
                "DHSVM attaches upstream links and stream-map cells to the first one only",
                where={"segment": sid, "lines": lines}, level="L1")
        for sid, lines in sorted(lines_by_id.items()) if len(lines) > 1
    ]
    findings += (_capped(rejected, "segments that DHSVM rejects")
                 + _capped(wrapped, "segment IDs above 65535")
                 + _capped(duplicates, "duplicated segment IDs")
                 + ignored)
    n_bad = n_table + len(rejected) + len(wrapped) + len(duplicates) + len(ignored)
    n = len(net.segments)
    return CheckOutput(
        summary=(f"all {n} segment records are accepted" if not findings else
                 f"{n_bad} problem(s) in {n} segment records"),
        findings=findings,
        metrics={"segments": n, "table_errors": n_table, "rejected": len(rejected),
                 "ids_above_65535": len(wrapped), "duplicate_ids": len(duplicates),
                 "dropped_fields": sum(n.kind == "ignored" for n in net.notes)},
    )


# ------------------------------------------------------------------ NET-003
NET_003 = RuleInfo(
    id="DHSVM-NET-003",
    title="Outlet links name existing segments and form no cycle",
    level="L0/L1",
    severity=Severity.ERROR,
    rationale=(
        "A positive outlet ID must name a segment of the file; otherwise DHSVM stops (L0). "
        "Following the outlet links from any segment must also end at a segment that leaves "
        "the network. DHSVM does not check that: with a cycle it runs to the end and "
        "silently loses the water of every segment that drains into the cycle, and with no "
        "outlet at all it loses the whole network's water (L1). Negative outlet IDs are the "
        "subject of DHSVM-NET-013."
    ),
    evidence=(
        Evidence(
            "After reading the records DHSVM looks up every non-zero outlet ID; an ID that "
            "names no segment is an error and the network is discarded. Nothing checks that "
            "the links are acyclic or that any segment is an outlet.",
            Status.VERIFIED_SOURCE,
            SRC + "channel.c:458-485",
        ),
        _EXIT_5,
        Evidence(
            "A segment is routed once per time step, in passes of increasing order, and its "
            "outflow is added to the inflow of its outlet segment. Inflow that arrives after "
            "the receiving segment was routed is never routed: the next step moves it to "
            "last_inflow, which the routing equations do not use. Around a cycle not every "
            "link can lead to a segment that is routed later, so at least one delivers too "
            "late.",
            Status.VERIFIED_SOURCE,
            SRC + "channel.c:491-526, 531-552, 557-562",
        ),
        Evidence(
            "Chiwawa with one outlet ID that names no segment stops with 'cannot find "
            "outlet' and exit status 5. With segment 492 draining into its own tributary 470 "
            "(474 of 494 segments then drain into the cycle), DHSVM runs to the end with the "
            "same warnings and mass balance error, and the outlet hydrograph loses more than "
            "99% of its flow.",
            Status.VERIFIED_EXPERIMENT,
            "tests/test_dhsvm_runs.py::test_dhsvm_stops_on_mutant[dangling-outlet], "
            "::test_dhsvm_runs_a_cycle_and_loses_the_water (2026-11-10)",
        ),
        _SHORT_IDS,
        _SHORT_IDS_RUN,
    ),
    mutations=("dangling-outlet", "cycle"),
)


def _trace_links(down: dict[int, int | None]) -> tuple[list[list[int]], dict[int, int]]:
    """Cycles among the links, and for every segment the index of the cycle it drains
    into (-1 if it reaches a segment that leaves the network)."""
    fate: dict[int, int] = {}
    cycles: list[list[int]] = []
    for start in down:
        path: list[int] = []
        position: dict[int, int] = {}
        x: int | None = start
        while x is not None and x not in fate and x not in position:
            position[x] = len(path)
            path.append(x)
            x = down[x]
        if x is None:
            result = -1
        elif x in fate:
            result = fate[x]
        else:
            cycles.append(path[position[x]:])
            result = len(cycles) - 1
        for p in path:
            fate[p] = result
    return cycles, fate


@register(MODEL, NET_003)
def check_net_003(case: DhsvmCase, params: dict) -> CheckOutput:
    net = case.network
    if not net.segments:
        raise NotApplicable("the network has no segment records (see DHSVM-NET-001)")
    by_id = net.by_id()
    dangling: list[Finding] = []
    wrapped: list[Finding] = []
    for s in net.segments:
        out = s.outlet_signed
        if out <= 0:
            continue
        if out > ID_MAX:
            wrapped.append(Finding(
                NET_003.id, Severity.ERROR,
                f"segment {s.id} (line {s.lineno}): outlet ID {out} exceeds 65535; DHSVM "
                f"looks up segment {out % 65536} instead",
                where={"segment": s.id, "line": s.lineno},
                values={"written": out, "read_as": out % 65536}, level="L1"))
        elif out not in by_id:
            dangling.append(Finding(
                NET_003.id, Severity.ERROR,
                f"segment {s.id} (line {s.lineno}): outlet ID {out} names no segment; DHSVM "
                "rejects the network ('cannot find outlet', exit 5)",
                where={"segment": s.id, "line": s.lineno}, values={"outlet": out},
                level="L0"))

    down = _down(case)  # an outlet ID that names no segment counts as leaving the network
    cycles, fate = _trace_links(down)
    trapped = Counter(i for i in fate.values() if i >= 0)
    n_outlets = sum(s.outlet_signed <= 0 for s in by_id.values())
    cyclic: list[Finding] = []
    for i in sorted(range(len(cycles)), key=lambda i: -trapped[i]):
        ring = cycles[i]
        shown = ring if len(ring) <= 8 else [*ring[:4], "...", *ring[-2:]]
        cyclic.append(Finding(
            NET_003.id, Severity.ERROR,
            f"segments {' -> '.join(str(x) for x in [*shown, ring[0]])} form a cycle; "
            f"{trapped[i]} segment(s) drain into it and never reach an outlet. DHSVM does "
            "not detect this and loses their water",
            where={"segments": ring}, values={"segments_draining_into_cycle": trapped[i]},
            level="L1"))
    findings = (_capped(dangling, "outlet IDs that name no segment")
                + _capped(wrapped, "outlet IDs above 65535")
                + _capped(cyclic, "cycles"))
    if n_outlets == 0:
        findings.append(Finding(
            NET_003.id, Severity.ERROR,
            "no segment leaves the network: no outlet ID is 0, so no water reaches an "
            "outlet", level="L1"))
    n_trapped = sum(trapped.values())
    return CheckOutput(
        summary=(f"{n_outlets} outlet(s); {len(dangling)} outlet ID(s) that name no segment; "
                 f"{len(cycles)} cycle(s), {n_trapped} of {len(down)} segments never reach "
                 "an outlet"),
        findings=findings,
        metrics={"outlets": n_outlets, "dangling_outlet_ids": len(dangling),
                 "outlet_ids_above_65535": len(wrapped), "cycles": len(cycles),
                 "segments_not_reaching_an_outlet": n_trapped},
    )


def _zmin(case: DhsvmCase, cells: np.ndarray) -> float:
    return float(case.dem[cells[:, 0], cells[:, 1]].min())


# ------------------------------------------------------------------ NET-004
NET_004 = RuleInfo(
    id="DHSVM-NET-004",
    title="Channel links do not run uphill",
    level="L1",
    severity=Severity.ERROR,
    rationale=(
        "Each segment is represented by the lowest DEM elevation among its stream-map cells. "
        "For every link (segment -> outlet segment) the downstream minimum must not exceed "
        "the upstream minimum by more than tolerance_m. Links that rise are the signature of a "
        "reversed network."
    ),
    evidence=(_ROUTES_BY_OUTLET_FIELD, _LEGACY_RUNS, _MUTANTS_RUN),
    mutations=("reverse-network",),
    params={"tolerance_m": 1.0},
)


@register(MODEL, NET_004)
def check_net_004(case: DhsvmCase, params: dict) -> CheckOutput:
    tol = float(params["tolerance_m"])
    cells = case.cells
    findings: list[Finding] = []
    rises: list[float] = []
    not_evaluated = 0
    for sid, did in _down(case).items():
        if did is None:
            continue
        if sid not in cells or did not in cells:
            not_evaluated += 1
            continue
        z_up, z_down = _zmin(case, cells[sid]), _zmin(case, cells[did])
        rise = z_down - z_up
        rises.append(rise)
        if rise > tol:
            findings.append(Finding(
                NET_004.id, Severity.ERROR,
                f"segment {sid} -> {did}: the downstream segment's lowest cell is {rise:.1f} m "
                f"higher ({z_down:.1f} m vs {z_up:.1f} m)",
                where={"segment": sid, "downstream": did},
                values={"z_up_m": z_up, "z_down_m": z_down, "rise_m": rise},
            ))
    findings.sort(key=lambda f: -f.values["rise_m"])
    if not_evaluated:
        findings.append(Finding(NET_004.id, Severity.INFO,
                                f"{not_evaluated} link(s) not evaluated: a segment has no "
                                "stream-map cells inside the grid"))
    n_bad = sum(f.severity == Severity.ERROR for f in findings)
    max_rise = max(rises) if rises else float("nan")
    return CheckOutput(
        summary=f"{n_bad} of {len(rises)} links rise more than {tol:g} m "
                f"(largest rise {max_rise:+.1f} m)",
        findings=findings,
        metrics={"links": len(rises), "uphill": n_bad, "max_rise_m": max_rise,
                 "not_evaluated": not_evaluated},
    )


# ------------------------------------------------------------------ NET-005
NET_005 = RuleInfo(
    id="DHSVM-NET-005",
    title="Linked segments touch in the stream map",
    level="L1",
    severity=Severity.ERROR,
    rationale=(
        "Water leaves a segment where it meets its outlet segment, so the two segments' "
        "stream-map cells must be neighbours: the smallest distance between their cells may "
        "not exceed max_gap_cells (default 1.5, i.e. one diagonal step). Larger gaps move "
        "water across the basin."
    ),
    evidence=(_ROUTES_BY_OUTLET_FIELD, _LEGACY_RUNS, _MUTANTS_RUN),
    mutations=("jump",),
    params={"max_gap_cells": 1.5},
)


@register(MODEL, NET_005)
def check_net_005(case: DhsvmCase, params: dict) -> CheckOutput:
    max_gap = float(params["max_gap_cells"])
    cells = case.cells
    km = case.grid.cellsize / 1000.0
    findings: list[Finding] = []
    gaps: list[float] = []
    not_evaluated = 0
    for sid, did in _down(case).items():
        if did is None:
            continue
        if sid not in cells or did not in cells:
            not_evaluated += 1
            continue
        a, b = cells[sid], cells[did]
        gap = float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).min()))
        gaps.append(gap)
        if gap > max_gap:
            findings.append(Finding(
                NET_005.id, Severity.ERROR,
                f"segment {sid} -> {did}: nearest cells are {gap:.1f} cells "
                f"({gap * km:.2f} km) apart",
                where={"segment": sid, "downstream": did},
                values={"gap_cells": gap, "gap_km": gap * km},
            ))
    findings.sort(key=lambda f: -f.values["gap_cells"])
    if not_evaluated:
        findings.append(Finding(NET_005.id, Severity.INFO,
                                f"{not_evaluated} link(s) not evaluated: a segment has no "
                                "stream-map cells inside the grid"))
    n_bad = sum(f.severity == Severity.ERROR for f in findings)
    gmax = max(gaps) if gaps else float("nan")
    return CheckOutput(
        summary=f"{n_bad} of {len(gaps)} links have a gap over {max_gap:g} cells "
                f"(largest {gmax:.2f} cells = {gmax * km:.2f} km)",
        findings=findings,
        metrics={"links": len(gaps), "jumps": n_bad, "max_gap_cells": gmax,
                 "not_evaluated": not_evaluated},
    )


# ------------------------------------------------------------------ NET-006
NET_006 = RuleInfo(
    id="DHSVM-NET-006",
    title="The outlet sits at the basin low point and few segments drain into it",
    level="L1",
    severity=Severity.ERROR,
    rationale=(
        "With a single outlet, the outlet segment's cells must come within "
        "max_outlet_distance_cells of the lowest DEM cell inside the basin mask (error). "
        "Any outlet that receives more than max_direct_inflows segments directly suggests "
        "orphan segments were wired to a sink (warning). A network without any outlet is "
        "reported by DHSVM-NET-003."
    ),
    evidence=(_ROUTES_BY_OUTLET_FIELD, _LEGACY_RUNS, _MUTANTS_RUN),
    mutations=("reverse-network", "force-wire"),
    params={"max_outlet_distance_cells": 3.0, "max_direct_inflows": 4},
)


@register(MODEL, NET_006)
def check_net_006(case: DhsvmCase, params: dict) -> CheckOutput:
    max_dist = float(params["max_outlet_distance_cells"])
    max_in = int(params["max_direct_inflows"])
    down = _down(case)
    cells = case.cells
    km = case.grid.cellsize / 1000.0
    by_id = case.network.by_id()
    outlets = sorted(s for s, d in down.items() if d is None)
    indeg = Counter(d for d in down.values() if d is not None)
    z_in = np.where(case.mask, case.dem, np.inf)
    if not np.isfinite(z_in).any():
        raise NotApplicable("the basin mask has no in-basin cells")
    z_low = float(z_in.min())
    low = np.argwhere(z_in == z_low)

    if not outlets:
        raise NotApplicable("no segment leaves the network (see DHSVM-NET-003)")
    findings: list[Finding] = []
    info = []
    for o in outlets:
        rec = {"segment": o, "direct_inflows": indeg.get(o, 0), "save": by_id[o].save}
        if o in cells:
            oc = cells[o]
            dist = float(np.sqrt(((oc[:, None, :] - low[None, :, :]) ** 2).sum(-1).min()))
            rec.update(distance_cells=dist, distance_km=dist * km,
                       z_outlet_min_m=_zmin(case, oc))
        info.append(rec)
        if indeg.get(o, 0) > max_in:
            findings.append(Finding(
                NET_006.id, Severity.WARNING,
                f"outlet segment {o} receives {indeg[o]} segments directly (more than "
                f"{max_in}); orphan segments may have been wired to it",
                where={"segment": o}, values={"direct_inflows": indeg[o]},
            ))
    if len(outlets) == 1:
        rec = info[0]
        if "distance_cells" not in rec:
            findings.append(Finding(NET_006.id, Severity.ERROR,
                                    f"outlet segment {outlets[0]} has no stream-map cells "
                                    "inside the grid", where={"segment": outlets[0]}))
        elif rec["distance_cells"] > max_dist:
            findings.append(Finding(
                NET_006.id, Severity.ERROR,
                f"the single outlet (segment {outlets[0]}) is {rec['distance_cells']:.1f} cells "
                f"({rec['distance_km']:.2f} km) from the lowest in-basin cell "
                f"(row {low[0][0]}, col {low[0][1]}, {z_low:.1f} m); its lowest cell is "
                f"{rec['z_outlet_min_m'] - z_low:.1f} m higher",
                where={"segment": outlets[0]},
                values={k: v for k, v in rec.items() if k != "segment"},
            ))
    elif len(outlets) > 1:
        findings.append(Finding(NET_006.id, Severity.INFO,
                                f"{len(outlets)} outlets; the low-point test applies to "
                                "single-outlet networks only"))

    if len(outlets) == 1 and "distance_cells" in info[0]:
        r = info[0]
        summary = (f"1 outlet (segment {r['segment']}): {r['distance_cells']:.1f} cells from the "
                   f"lowest in-basin cell, {r['direct_inflows']} direct inflow(s)")
    else:
        summary = f"{len(outlets)} outlet(s); max direct inflows " \
                  f"{max((r['direct_inflows'] for r in info), default=0)}"
    return CheckOutput(
        summary=summary,
        findings=findings,
        metrics={"outlets": info, "lowest_cell": [int(low[0][0]), int(low[0][1])],
                 "z_lowest_m": z_low},
    )


# ------------------------------------------------------------------ NET-007
NET_007 = RuleInfo(
    id="DHSVM-NET-007",
    title="DHSVM accepts every stream map record and uses it",
    level="L0/L1",
    severity=Severity.ERROR,
    rationale=(
        "A stream map record is 'column row segment length cut-height cut-width ...' with "
        "column and row counted from 0 (row 0 is the northernmost). DHSVM stops if a record "
        "lies outside the grid, names a segment that is not in the network, has a negative "
        "length, cut height or cut width, or sits in a cell with a negative soil depth "
        "(L0). It accepts, without a message: a record outside the basin mask, which it "
        "never uses; a segment with no record in the basin, which gets no water from the "
        "grid; and an in-basin cell whose records have a total length of 0, which turns "
        "the streamflow into NaN (L1)."
    ),
    evidence=(
        Evidence(
            "channel_grid_read_map counts an error for a column outside 0..NX-1 or a row "
            "outside 0..NY-1 ('bad coordinates'), a segment ID that is not in the network, "
            "and a negative length, cut height or cut width. A cut height above the cell's "
            "soil depth is first replaced by 0.95 x soil depth, so a negative soil depth "
            "gives 'bad cut_depth'. A length of 0 is accepted.",
            Status.VERIFIED_SOURCE,
            SRC + "channel_grid.c:231-261, 275-316; DHSVMChannel.c:71",
        ),
        Evidence(
            "The record's row and column index the model grid as [row][column]; the grid is "
            "filled from the binary maps row by row, north to south, each row west to east.",
            Status.VERIFIED_SOURCE,
            SRC + "channel_grid.c:298; InitTerrainMaps.c:262-266; InitFileIO.c:62-66",
        ),
        Evidence(
            "A segment receives water from the grid only through its map records: in every "
            "time step each in-basin cell that holds records gives them its water as mass x "
            "length / total length of the cell's records, which is 0/0 if that total is 0. "
            "Cells outside the basin mask are never visited.",
            Status.VERIFIED_SOURCE,
            SRC + "channel_grid.c:440-450, 498-516; DHSVMChannel.c:233-252; "
                  "RouteSubSurface.c:173-175, 272-300",
        ),
        _TABLE_ERRORS,
        _EXIT_5,
        Evidence(
            "Chiwawa with the column and row fields swapped stops with 'bad coordinates' "
            "and exit status 5; an unknown segment ID and a negative length, cut height or "
            "cut width in one record stop it the same way. With the length of one map "
            "record set to 0, DHSVM runs to the end with exit status 0 and the outlet "
            "hydrograph is NaN.",
            Status.VERIFIED_EXPERIMENT,
            "tests/test_dhsvm_runs.py::test_dhsvm_stops_on_mutant[map-swap-row-col], "
            "::test_dhsvm_rejects_map_field, "
            "::test_zero_length_map_record_gives_nan_streamflow (2026-11-10)",
        ),
        Evidence(
            "A map record added in an out-of-basin cell of Chiwawa stops DHSVM with 'bad "
            "cut_depth' (exit 5), because the soil depth map holds -9999 outside the basin. "
            "With the soil depth outside the basin set to 1 m the record is accepted and "
            "every output file is byte-identical to the run without it. With a stream map "
            "that holds no record at all DHSVM runs to the end without an error and the "
            "outlet only drains the initial channel storage.",
            Status.VERIFIED_EXPERIMENT,
            "tests/test_dhsvm_runs.py::test_map_record_outside_the_basin, "
            "::test_stream_map_without_records_starves_the_network (2026-11-10)",
        ),
        _SHORT_IDS,
    ),
    mutations=("map-swap-row-col", "zero-length-record"),
)


@register(MODEL, NET_007)
def check_net_007(case: DhsvmCase, params: dict) -> CheckOutput:
    sm, g = case.stream_map, case.grid
    ids = set(case.network.by_id())
    mask = case.mask
    notes: list[Finding] = []
    try:
        depth = case.soil_depth
    except InputProblem as exc:
        depth = None
        notes.append(Finding(NET_007.id, Severity.INFO,
                             f"soil depth not checked at the map records: {exc}"))
    findings = _table_error_findings(NET_007.id, sm.notes, "stream map")
    n_table = len(findings)
    ignored = _ignored_field_findings(NET_007.id, sm.notes)

    def record_findings(index: np.ndarray, level: str, severity: Severity, text) -> list:
        return [Finding(NET_007.id, severity, f"line {int(sm.lineno[k])}: {text(k)}",
                        where={"line": int(sm.lineno[k]), "segment": int(sm.segment[k]),
                               "row": int(sm.row[k]), "column": int(sm.col[k])}, level=level)
                for k in index]

    in_grid = (sm.col >= 0) & (sm.col < g.ncols) & (sm.row >= 0) & (sm.row < g.nrows)
    known = np.isin(sm.segment, sorted(ids))
    f32 = np.float32
    negative = {
        "length": sm.length.astype(f32) < 0,
        "cut height": sm.cut_height.astype(f32) < 0,
        "cut width": sm.cut_width.astype(f32) < 0,
    }
    rows = np.where(in_grid, sm.row, 0)
    cols = np.where(in_grid, sm.col, 0)
    in_basin = in_grid & mask[rows, cols]
    if depth is not None:
        cell_depth = depth[rows, cols]
        bad_depth = in_grid & (cell_depth.astype(f32) < 0) & ~negative["cut height"]
    else:
        cell_depth = np.zeros(len(sm))
        bad_depth = np.zeros(len(sm), dtype=bool)
    off_basin = in_grid & ~in_basin & ~bad_depth

    stop = "DHSVM rejects the stream map"
    rejected = record_findings(
        np.flatnonzero(~in_grid), "L0", Severity.ERROR,
        lambda k: f"column {sm.col[k]}, row {sm.row[k]} is outside the grid (columns "
                  f"0..{g.ncols - 1}, rows 0..{g.nrows - 1}); {stop} ('bad coordinates', exit 5)")
    rejected += record_findings(
        np.flatnonzero(~known & (sm.segment <= ID_MAX)), "L0", Severity.ERROR,
        lambda k: f"segment {sm.segment[k]} is not in the stream network; {stop} ('unable to "
                  "locate segment', exit 5)")
    for name, flags in negative.items():
        column = {"length": sm.length, "cut height": sm.cut_height,
                  "cut width": sm.cut_width}[name]
        rejected += record_findings(
            np.flatnonzero(flags), "L0", Severity.ERROR,
            lambda k, name=name, column=column:
                f"segment {sm.segment[k]}: {name} {column[k]:g} is negative; {stop} (exit 5)")
    rejected += record_findings(
        np.flatnonzero(bad_depth), "L0", Severity.ERROR,
        lambda k: f"segment {sm.segment[k]} in cell row {sm.row[k]}, column {sm.col[k]}"
                  f"{'' if in_basin[k] else ' (outside the basin mask)'}: the soil depth "
                  f"there is {cell_depth[k]:g} m; {stop} ('bad cut_depth', exit 5)")
    wrapped = record_findings(
        np.flatnonzero(sm.segment > ID_MAX), "L1", Severity.ERROR,
        lambda k: f"segment ID {sm.segment[k]} exceeds 65535; DHSVM looks up segment "
                  f"{sm.segment[k] % 65536} instead")
    unused = record_findings(
        np.flatnonzero(off_basin), "L1", Severity.WARNING,
        lambda k: f"segment {sm.segment[k]} in cell row {sm.row[k]}, column {sm.col[k]}: the "
                  "cell is outside the basin mask, so DHSVM never uses this record")

    # in-basin cells whose records have a total channel length of 0
    usable = in_basin & ~negative["length"]
    cell_key = sm.row[usable] * g.ncols + sm.col[usable]
    total = np.bincount(cell_key, weights=sm.length[usable].astype(f32),
                        minlength=g.nrows * g.ncols)
    count = np.bincount(cell_key, minlength=g.nrows * g.ncols)
    dead_cells = np.flatnonzero((count > 0) & (total == 0))
    lines_at = defaultdict(list)
    if len(dead_cells):
        dead = set(dead_cells.tolist())
        for k in np.flatnonzero(usable):
            key = int(sm.row[k]) * g.ncols + int(sm.col[k])
            if key in dead:
                lines_at[key].append(int(k))
    zero = [
        Finding(NET_007.id, Severity.ERROR,
                f"cell row {key // g.ncols}, column {key % g.ncols} (line(s) "
                f"{_ids(int(sm.lineno[k]) for k in ks)}; segment(s) "
                f"{_ids(sorted({int(sm.segment[k]) for k in ks}))}): the channel length in the "
                "cell is 0. DHSVM divides by it in every time step, and the flow of these "
                "segments and of everything downstream becomes NaN",
                where={"row": key // g.ncols, "column": key % g.ncols,
                       "lines": [int(sm.lineno[k]) for k in ks]}, level="L1")
        for key, ks in sorted(lines_at.items())
    ]

    fed = set(sm.segment[in_basin].tolist())
    dry = sorted(ids - fed)
    findings += (_capped(rejected, "records that DHSVM rejects")
                 + _capped(wrapped, "segment IDs above 65535")
                 + _capped(zero, "cells with a channel length of 0")
                 + _capped(unused, "records outside the basin mask")
                 + ignored)
    if dry:
        findings.append(Finding(
            NET_007.id, Severity.WARNING,
            f"{len(dry)} of {len(ids)} segments have no stream-map record in an in-basin "
            f"cell (segment(s) {_ids(dry)}): they receive no water from the grid, only "
            "from upstream segments",
            where={"segments": dry[:200]}, values={"count": len(dry)}, level="L1"))
    findings += notes
    n = len(sm)
    problems = (n_table + len(rejected) + len(wrapped) + len(zero) + len(unused) + bool(dry)
                + len(ignored))
    summary = (
        f"all {n} map records are accepted and lie in the basin; every segment has one"
        if not problems else
        f"{n} map records: {n_table + len(rejected)} that DHSVM rejects, {len(unused)} "
        f"outside the basin mask, {len(zero)} cell(s) with a channel length of 0; {len(dry)} "
        f"of {len(ids)} segments without an in-basin record"
    )
    return CheckOutput(
        summary=summary,
        findings=findings,
        metrics={"records": n, "table_errors": n_table, "rejected": len(rejected),
                 "outside_grid": int((~in_grid).sum()),
                 "outside_basin_mask": int((in_grid & ~in_basin).sum()),
                 "segment_ids_above_65535": len(wrapped),
                 "cells_with_zero_channel_length": len(zero),
                 "segments_without_in_basin_record": len(dry)},
    )


# ------------------------------------------------------------------ NET-010
NET_010 = RuleInfo(
    id="DHSVM-NET-010",
    title="Channel classes exist and have usable values",
    level="L0/L1/L2",
    severity=Severity.ERROR,
    rationale=(
        "Every class a segment uses must be defined in the class file, and every class "
        "needs a width, a bank height and a Manning's n above 0; otherwise DHSVM stops "
        "(L0). Class IDs must be unique and at most 65535 (L1). DHSVM routes a segment "
        "with a constant velocity that it computes from the segment's slope, the class's "
        "Manning's n and a flow depth of 0.75 x the class's bank height; it never uses the "
        "class width. A bank height below min_bank_height_m or above max_bank_height_m, or "
        "a Manning's n outside min_manning_n..max_manning_n, in a class that segments use "
        "is therefore reported as implausible (L2, warning). These four bounds are wader "
        "defaults, not DHSVM limits; the lower bank-height bound lies above the 0.03-0.10 m "
        "placeholders of PNNL's class template."
    ),
    evidence=(
        Evidence(
            "channel_read_classes aborts if a class width, bank height or Manning's n is not "
            "above 0. A class ID of 0 only prints 'class id invalid'; the class is kept. "
            "channel_read_network counts an error for a segment whose class is not in the "
            "class list. A class file without records gives no class list.",
            Status.VERIFIED_SOURCE,
            SRC + "channel.c:126-178, 220-227, 425-434",
        ),
        Evidence(
            "The routing constant of a segment is K = sqrt(slope) x (0.75 x bank height)^(2/3)"
            " / (Manning's n x length), i.e. a Manning velocity divided by the length. The "
            "class width is stored and never read.",
            Status.VERIFIED_SOURCE,
            SRC + "channel.c:161-166, 304-319",
        ),
        Evidence(
            "PNNL's stream network scripts assign each class a width and a depth that are "
            "placeholders (the script header says the classification requires manual "
            "modification): depths of 0.03-0.10 m and widths of 0.5-4.5 m in the Python "
            "version; the AML version carries the same numbers with the two labels swapped.",
            Status.VERIFIED_SOURCE,
            "DHSVM-PNNL@e6f333d CreateStreamNetwork_PythonV/channelclass.py:1-14, 35-124; "
            "DHSVM/amlscripts/createstreamnetwork.aml:317-420",
        ),
        _TABLE_ERRORS,
        _EXIT_5,
        Evidence(
            "Chiwawa with one segment given a class that the class file lacks stops with "
            "'channel class 19 not found' and exit status 5; with one class width set to 0 "
            "DHSVM aborts (SIGABRT). An added class with ID 0 prints 'class id invalid' and "
            "leaves the output byte-identical.",
            Status.VERIFIED_EXPERIMENT,
            "tests/test_dhsvm_runs.py::test_dhsvm_stops_on_mutant[missing-class], "
            "::test_class_file_quirks (2026-11-10)",
        ),
        Evidence(
            "Chiwawa with every class width set to 0.03 m gives byte-identical output. With "
            "the widths and bank heights of PNNL's template (bank heights 0.03-0.10 m "
            "instead of 0.4-2.5 m) DHSVM runs to the end and the outlet hydrograph changes.",
            Status.VERIFIED_EXPERIMENT,
            "tests/test_dhsvm_runs.py::test_class_file_quirks, "
            "::test_placeholder_classes_change_the_hydrograph (2026-11-10)",
        ),
        _SHORT_IDS,
    ),
    mutations=("missing-class", "placeholder-classes"),
    params={"min_bank_height_m": 0.15, "max_bank_height_m": 20.0,
            "min_manning_n": 0.01, "max_manning_n": 0.5},
)


@register(MODEL, NET_010)
def check_net_010(case: DhsvmCase, params: dict) -> CheckOutput:
    table, net = case.classes, case.network
    by_id = table.by_id()
    lo_h, hi_h = float(params["min_bank_height_m"]), float(params["max_bank_height_m"])
    lo_n, hi_n = float(params["min_manning_n"]), float(params["max_manning_n"])
    findings = _table_error_findings(NET_010.id, table.notes, "class file")
    findings += _ignored_field_findings(NET_010.id, table.notes)
    if not table.classes and not any(n.kind == "error" for n in table.notes):
        findings.append(Finding(NET_010.id, Severity.ERROR,
                                "the class file defines no class; DHSVM stops (exit 5)",
                                level="L0"))
    lines_by_id: dict[int, list[int]] = defaultdict(list)
    for c in table.classes:
        lines_by_id[c.id].append(c.lineno)
        where = {"class": c.id, "line": c.lineno}
        for name, value in (("width", c.width), ("bank height", c.bank_height),
                            ("Manning's n", c.friction)):
            if as_c_float(value) <= 0:
                findings.append(Finding(
                    NET_010.id, Severity.ERROR,
                    f"class {c.id} (line {c.lineno}): {name} {value:g} is not above 0; DHSVM "
                    "aborts while reading the class file", where=where, level="L0"))
        if c.id > ID_MAX:
            findings.append(Finding(
                NET_010.id, Severity.ERROR,
                f"class {c.id} (line {c.lineno}): the ID exceeds 65535; DHSVM keeps IDs in 16 "
                f"bits and uses {c.id % 65536} instead", where=where, level="L1"))
        if c.id == 0:
            findings.append(Finding(
                NET_010.id, Severity.INFO,
                f"class 0 (line {c.lineno}): DHSVM prints 'class id invalid' but keeps and "
                "uses the class", where=where, level="L1"))
    for cid, lines in sorted(lines_by_id.items()):
        if len(lines) > 1:
            findings.append(Finding(
                NET_010.id, Severity.WARNING,
                f"class {cid} is defined {len(lines)} times (lines {_ids(lines)}); DHSVM uses "
                "the first definition", where={"class": cid, "lines": lines}, level="L1"))

    users: dict[int, list[int]] = defaultdict(list)
    for s in net.segments:
        users[s.cls].append(s.id)
    missing = 0
    used = []
    for cid, segs in sorted(users.items()):
        if cid > ID_MAX:
            findings.append(Finding(
                NET_010.id, Severity.ERROR,
                f"class {cid}, used by segment(s) {_ids(segs)}, exceeds 65535; DHSVM looks "
                f"up class {cid % 65536} instead",
                where={"class": cid, "segments": segs[:200]}, level="L1"))
            continue
        c = by_id.get(cid)
        if c is None:
            missing += 1
            findings.append(Finding(
                NET_010.id, Severity.ERROR,
                f"class {cid} is used by {len(segs)} segment(s) ({_ids(segs)}) but the class "
                f"file does not define it; DHSVM rejects the network ('channel class {cid} "
                "not found', exit 5)",
                where={"class": cid, "segments": segs[:200]}, values={"segments": len(segs)},
                level="L0"))
            continue
        used.append(c)
        where = {"class": cid, "line": c.lineno}
        if 0 < c.bank_height < lo_h or c.bank_height > hi_h:
            side = (f"below {lo_h:g} m; PNNL's class template carries 0.03-0.10 m as "
                    "placeholders" if c.bank_height < lo_h else
                    f"above {hi_h:g} m")
            findings.append(Finding(
                NET_010.id, Severity.WARNING,
                f"class {cid} (line {c.lineno}, used by {len(segs)} segment(s)): bank height "
                f"{c.bank_height:g} m is {side}. DHSVM takes 0.75 x bank height as the flow "
                "depth that sets the routing velocity of these segments",
                where=where, values={"bank_height_m": c.bank_height, "segments": len(segs)},
                level="L2"))
        if 0 < c.friction < lo_n or c.friction > hi_n:
            findings.append(Finding(
                NET_010.id, Severity.WARNING,
                f"class {cid} (line {c.lineno}, used by {len(segs)} segment(s)): Manning's n "
                f"{c.friction:g} is outside {lo_n:g}..{hi_n:g}",
                where=where, values={"manning_n": c.friction, "segments": len(segs)},
                level="L2"))
    n_bad = sum(f.severity >= Severity.WARNING for f in findings)
    if used:
        heights = [c.bank_height for c in used]
        ns = [c.friction for c in used]
        ranges = (f"bank height {min(heights):g}-{max(heights):g} m, Manning's n "
                  f"{min(ns):g}-{max(ns):g}")
    else:
        ranges = "no defined class in use"
    return CheckOutput(
        summary=(f"{len(table.classes)} classes, {len(users)} used by {len(net.segments)} "
                 f"segments: {ranges}" + (f"; {n_bad} problem(s)" if n_bad else "")),
        findings=findings,
        metrics={"classes": len(table.classes), "classes_used": len(users),
                 "classes_missing": missing,
                 "bank_height_m": [min(heights), max(heights)] if used else None,
                 "manning_n": [min(ns), max(ns)] if used else None},
    )


# ------------------------------------------------------------------ NET-013
NET_013 = RuleInfo(
    id="DHSVM-NET-013",
    title="The basin outlet is written so that DHSVM reads it as an outlet",
    level="L1",
    severity=Severity.WARNING,
    rationale=(
        "DHSVM treats outlet ID 0 as 'leaves the network'. Its table scanner has no sign for "
        "integers, so an outlet written as -1 is read as segment 1: DHSVM then routes the "
        "basin outflow into segment 1. If that segment is routed in an earlier or the same "
        "order pass, the water arrives after it was routed and is dropped at the next step, "
        "and the network 'Totals' outflow is reported as 0. If the ID names no segment, "
        "DHSVM stops. Also reported (info): outlets without a SAVE flag, which write no "
        "hydrograph."
    ),
    evidence=(
        Evidence("Table integers are unsigned ([0-9]+); a '-' matches no rule and flex's "
                 "default rule echoes and skips it, so '-1' is read as 1.",
                 Status.VERIFIED_SOURCE, SRC + "tableio.lex:66, tableio.c:1006 (ECHO)"),
        Evidence("Outlet ID 0 means no downstream segment; any other ID must name a segment "
                 "or reading the network fails.",
                 Status.VERIFIED_SOURCE, SRC + "channel.c:458-470"),
        Evidence("Outflow is added to the outlet segment's inflow after routing; inflow is "
                 "reset at the start of the next step; 'Totals' outflow sums segments without "
                 "an outlet only.",
                 Status.VERIFIED_SOURCE,
                 SRC + "channel.c:522-523, channel.c:557-561, channel.c:619-621/630-632"),
        Evidence("PNNL Chiwawa with its outlet written as -1 reports 0 'Totals' outflow at "
                 "every step; written as 0, 'Totals' outflow equals the basin outflow. The "
                 "outlet hydrograph (SAVE) and the mass balance are identical in both runs. "
                 "(First reported in the brief's first-round experiment.)",
                 Status.VERIFIED_EXPERIMENT,
                 "tests/test_dhsvm_runs.py::test_negative_outlet_is_read_as_segment_1 "
                 "(2026-10-30)"),
    ),
    mutations=("negative-outlet",),
)


@register(MODEL, NET_013)
def check_net_013(case: DhsvmCase, params: dict) -> CheckOutput:
    net = case.network
    by_id = net.by_id()
    findings: list[Finding] = []
    for s in net.segments:
        if s.outlet_signed >= 0:
            continue
        target = by_id.get(s.outlet)
        if target is None:
            findings.append(Finding(
                NET_013.id, Severity.ERROR,
                f"segment {s.id} (line {s.lineno}): outlet ID {s.outlet_signed} is read as "
                f"{s.outlet}, which names no segment; DHSVM stops reading the network",
                where={"segment": s.id, "line": s.lineno},
                values={"written": s.outlet_signed, "read_as": s.outlet},
            ))
            continue
        # routed in passes of increasing order, file order within a pass (channel.c:538-547)
        dropped = (target.order, target.lineno) <= (s.order, s.lineno)
        effect = ("the water arrives after that segment was routed and is dropped, so the "
                  "network 'Totals' outflow is 0" if dropped else
                  "the water is routed through the network again")
        findings.append(Finding(
            NET_013.id, Severity.WARNING,
            f"segment {s.id} (line {s.lineno}): outlet ID {s.outlet_signed} is read as "
            f"{s.outlet}, so DHSVM sends the outflow into segment {s.outlet} "
            f"(order {target.order}) instead of out of the basin; {effect}. Write 0.",
            where={"segment": s.id, "line": s.lineno},
            values={"written": s.outlet_signed, "read_as": s.outlet, "dropped": dropped},
        ))
    outlets = [s for s in by_id.values() if s.outlet_signed <= 0]
    for s in outlets:
        if not s.save:
            findings.append(Finding(
                NET_013.id, Severity.INFO,
                f"outlet segment {s.id} has no SAVE flag, so DHSVM writes no hydrograph for it",
                where={"segment": s.id},
            ))
    n_bad = sum(f.severity >= Severity.WARNING for f in findings)
    return CheckOutput(
        summary=(f"{n_bad} outlet field(s) that DHSVM reads differently"
                 if n_bad else "outlet fields read as written"),
        findings=findings,
        metrics={"misread_outlets": n_bad, "outlets_written": [s.id for s in outlets]},
    )
