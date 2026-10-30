"""Stream-network rules (DHSVM-NET-*).

The topology rules (NET-004, NET-005, NET-006) evaluate the network the file
author intended: an outlet ID <= 0 means "leaves the network". NET-013
reports where DHSVM reads the outlet field differently from that intent.

Source references are to pnnl/DHSVM-PNNL@e6f333d (DHSVM/sourcecode/).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from wader.core.rules import (
    CheckOutput,
    Evidence,
    Finding,
    NotApplicable,
    RuleInfo,
    Severity,
    Status,
    register,
)
from wader.models.dhsvm.case import DhsvmCase

MODEL = "dhsvm"
SRC = "DHSVM-PNNL@e6f333d "

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
    "Every wader mutation of the PNNL Chiwawa network runs in DHSVM with exit status 0. "
    "The reverse-network mutant (64 links flipped) gives the same warnings and mass balance "
    "error as the original and a network outflow within 0.2%, while the outlet hydrograph "
    "(segment 494) falls from 10.19 to 0.01 m3/s (October 1970).",
    Status.VERIFIED_EXPERIMENT,
    "tests/test_dhsvm_runs.py; scripts/demo_reverse_network.sh (2026-10-30)",
)


def _down(case: DhsvmCase) -> dict[int, int | None]:
    return case.network.intended_downstream()


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
        "orphan segments were wired to a sink (warning). A network with no outlet at all is "
        "an error."
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

    findings: list[Finding] = []
    if not outlets:
        findings.append(Finding(
            NET_006.id, Severity.ERROR,
            "no segment leaves the network: every outlet ID names another segment, so the "
            "links form a cycle",
        ))
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
