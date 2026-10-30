#!/usr/bin/env python3
"""Fetch the PNNL Chiwawa reference case (TestCase/Chiwawa of pnnl/DHSVM-PNNL).

Downloads only what DHSVM needs, at a pinned commit:

* ``INPUT.Chiwawa.Baseline``, ``input/`` and ``modelstate/`` (about 64 MB);
* the gridded forcing files that DHSVM selects for this basin, out of 320
  (3.4 GB). DHSVM lists the forcing directory and uses a file only if its
  grid point falls in an in-basin cell (InitMetSources.c:313-358, ported in
  ``wader.models.dhsvm.met``). A file that is left out is never considered, so
  the selection must be complete; ``--verify-selection DHSVM`` checks it
  against DHSVM itself.

Forcing files are trimmed to the simulation period (by default the INPUT's
Model Start to Model End plus one day). That is safe because DHSVM scans each
file forward to the current date (ReadMetRecord.c:61-77).

The data go to a cache directory and are never committed: DHSVM-PNNL has no
license file. Layout of the result (``--dest``)::

    <dest>/TestCase/Chiwawa/...   as upstream, forcing reduced and trimmed
    <dest>/MANIFEST.json          commit, files, forcing selection, checks

The INPUT file uses paths like ``../../TestCase/Chiwawa/input/dem.bin``, so
DHSVM must be started two levels below ``<dest>``, e.g. in ``<dest>/run/<name>/``
(``cases/chiwawa/run.py`` does that).

Requires git and an installed ``wader`` (``pip install -e .``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from wader.cache import DHSVM_COMMIT, DHSVM_REPO, chiwawa_dir
from wader.models.dhsvm.case import DhsvmCase
from wader.models.dhsvm.met import dhsvm_file_name, locate_met_files

FETCH_VERSION = 1
CASE = "TestCase/Chiwawa"
INPUT_NAME = "INPUT.Chiwawa.Baseline"
DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})-(\d{1,2})")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True,
                          stdout=subprocess.PIPE).stdout


def parse_date(text: str) -> datetime:
    m = DATE_RE.match(text.strip())
    if not m:
        raise ValueError(f"not a DHSVM date (MM/DD/YYYY-HH): {text!r}")
    mo, d, y, h = (int(g) for g in m.groups())
    return datetime(y, mo, d, h)


def fmt_date(d: datetime) -> str:
    return f"{d:%m/%d/%Y-%H}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def met_settings(case: DhsvmCase) -> dict:
    c = case.config
    if (c.get("OPTIONS", "Gridded Met data") or "")[:4] != "TRUE":
        raise SystemExit("this script handles gridded met data only")
    if (c.get("OPTIONS", "Outside") or "")[:5] != "FALSE":
        raise SystemExit("this script handles Outside = FALSE only")
    return {
        "prefix": c.require("METEOROLOGY", "File Prefix").value,
        "decimals": c.get_int("METEOROLOGY", "GRID_DECIMAL"),
        "zone": c.get_int("METEOROLOGY", "UTM Zone"),
        "n_grids": c.get_int("METEOROLOGY", "Number of Grids"),
    }


def select_forcing(case: DhsvmCase, names: list[str]) -> tuple[list[str], list[dict]]:
    met = met_settings(case)
    g = case.grid
    points = locate_met_files(names, met["prefix"], met["zone"], g.north, g.west, g.cellsize,
                              g.nrows, g.ncols, case.mask)
    chosen = [p for p in points if p.in_basin]
    for p in chosen:
        expected = dhsvm_file_name(met["prefix"], p.lat, p.lon, met["decimals"])
        if expected != p.name:
            raise SystemExit(f"DHSVM would open {expected!r} for directory entry {p.name!r}")
    if len(chosen) >= met["n_grids"]:
        raise SystemExit(f"{len(chosen)} grids selected but Number of Grids is {met['n_grids']}; "
                         "DHSVM would overrun its station array (InitMetSources.c:309, 331)")
    detail = [{"name": p.name, "row": p.row, "col": p.col, "near_cell_edge": p.near_edge}
              for p in chosen]
    return sorted(p.name for p in chosen), detail


def trim(path: Path, start: datetime, end: datetime) -> int:
    """Keep the records from ``start`` to ``end`` inclusive; return how many."""
    keep = []
    with path.open() as f:
        for line in f:
            m = DATE_RE.match(line)
            if m and start <= parse_date(m.group(0)) <= end:
                keep.append(line)
    if not keep or parse_date(keep[0]) != start:
        raise SystemExit(f"{path.name}: no record at {fmt_date(start)}")
    if parse_date(keep[-1]) != end:
        raise SystemExit(f"{path.name}: no record at {fmt_date(end)}")
    path.write_text("".join(keep))
    return len(keep)


def fetch(dest: Path, commit: str, start: str | None, end: str | None, no_trim: bool) -> dict:
    stage = Path(tempfile.mkdtemp(prefix=".fetch-", dir=dest.parent))
    try:
        t0 = time.time()
        log(f"Fetching {CASE} from {DHSVM_REPO} at {commit[:7]} (partial clone) ...")
        git(stage, "init", "-q")
        git(stage, "remote", "add", "origin", DHSVM_REPO)
        git(stage, "config", "core.sparseCheckout", "true")
        sparse = stage / ".git" / "info" / "sparse-checkout"
        sparse.write_text(f"/{CASE}/\n!/{CASE}/LivnehForcing/\n")
        git(stage, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
        git(stage, "checkout", "-q", "--detach", "FETCH_HEAD")
        if git(stage, "rev-parse", "HEAD").strip() != commit:
            raise SystemExit("fetched commit does not match")
        listing = git(stage, "ls-tree", "--name-only", "HEAD", f"{CASE}/LivnehForcing/")
        all_names = sorted(Path(p).name for p in listing.split())
        log(f"  inputs checked out ({time.time() - t0:.0f} s); "
            f"{len(all_names)} forcing files upstream")

        workdir = stage / "run" / "fetch"  # paths like ../../TestCase need a real directory
        workdir.mkdir(parents=True)
        case = DhsvmCase(stage / CASE / INPUT_NAME, workdir=workdir)
        met_dir = case.path("METEOROLOGY", "Met File Path")
        if met_dir.resolve() != (stage / CASE / "LivnehForcing").resolve():
            raise SystemExit(f"unexpected Met File Path {met_dir}")
        selected, detail = select_forcing(case, all_names)
        log(f"  {len(selected)} forcing files have their grid point in an in-basin cell")
        with sparse.open("a") as f:
            for name in selected:
                f.write(f"/{CASE}/LivnehForcing/{name}\n")
        git(stage, "read-tree", "-mu", "HEAD")
        missing = [n for n in selected if not (stage / CASE / "LivnehForcing" / n).is_file()]
        if missing:
            raise SystemExit(f"forcing files not checked out: {missing[:5]}")
        log(f"  forcing fetched ({time.time() - t0:.0f} s)")

        trimmed = None
        if not no_trim:
            t_start = parse_date(start or case.config.require("TIME", "Model Start").value)
            t_end = (parse_date(end) if end else
                     parse_date(case.config.require("TIME", "Model End").value)
                     + timedelta(days=1))
            records = [trim(stage / CASE / "LivnehForcing" / n, t_start, t_end)
                       for n in selected]
            trimmed = {"start": fmt_date(t_start), "end": fmt_date(t_end),
                       "records_per_file": sorted(set(records))}
            log(f"  forcing trimmed to {trimmed['start']} .. {trimmed['end']}")

        shutil.rmtree(stage / ".git")
        shutil.rmtree(stage / "run")
        files = {str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted((stage / CASE).rglob("*")) if p.is_file()}
        manifest = {
            "fetch_version": FETCH_VERSION,
            "repository": DHSVM_REPO,
            "commit": commit,
            "input": f"{CASE}/{INPUT_NAME}",
            "run_from": "two levels below this directory, e.g. run/<name>/",
            "forcing": {
                "upstream_files": all_names,
                "selected": selected,
                "selected_detail": detail,
                "trimmed": trimmed,
            },
            "files": files,
            "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seconds": round(time.time() - t0, 1),
        }
        (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
        if dest.exists():
            shutil.rmtree(dest)
        stage.rename(dest)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def verify_selection(dest: Path, manifest: dict, dhsvm: Path, timeout: int = 900) -> dict:
    """Let DHSVM choose among empty placeholder files for *all* upstream forcing names
    and compare its choice with ours. (DHSVM then stops at its first met read.)"""
    names = manifest["forcing"]["upstream_files"]
    work = dest / "run" / ".verify-met"
    shutil.rmtree(work, ignore_errors=True)
    (work / "out").mkdir(parents=True)
    case = DhsvmCase(dest / manifest["input"], workdir=work)
    met = met_settings(case)
    with tempfile.TemporaryDirectory(prefix="wader-met-") as tmp:
        met_dir = Path(tmp)
        for n in names:
            (met_dir / n).touch()
        cfg = case.config.with_value("METEOROLOGY", "Met File Path", f"{met_dir}/")
        cfg = cfg.with_value("OUTPUT", "Output Directory", "out/")
        cfg.write(work / "INPUT")
        proc = subprocess.run([str(dhsvm), "INPUT"], cwd=work, capture_output=True, text=True,
                              timeout=timeout)
    shutil.rmtree(work, ignore_errors=True)
    # "..... Station %d: %s is selected", where %s is "data_%f_%f\n" (InitMetSources.c:331, 356)
    found = re.findall(r"Station\s+\d+:\s+\w+?_([-0-9.]+)_([-0-9.]+)\s+is selected", proc.stdout)
    picked = sorted(dhsvm_file_name(met["prefix"], float(la), float(lo), met["decimals"])
                    for la, lo in found)
    if not picked:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-20:])
        raise SystemExit(f"DHSVM selected no files; its output ends with:\n{tail}")
    ours = manifest["forcing"]["selected"]
    return {
        "dhsvm_selected": len(picked),
        "fetched": len(ours),
        "missing_from_fetch": sorted(set(picked) - set(ours)),
        "extra_in_fetch": sorted(set(ours) - set(picked)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=Path, default=None,
                    help="target directory (default: $WADER_CACHE/cases/chiwawa-<commit>)")
    ap.add_argument("--commit", default=DHSVM_COMMIT)
    ap.add_argument("--start", help="first forcing record to keep, MM/DD/YYYY-HH "
                                    "(default: Model Start)")
    ap.add_argument("--end", help="last forcing record to keep (default: Model End + 1 day)")
    ap.add_argument("--no-trim", action="store_true", help="keep the full forcing records")
    ap.add_argument("--force", action="store_true", help="fetch again even if up to date")
    ap.add_argument("--verify-selection", type=Path, metavar="DHSVM",
                    help="check the forcing selection against this DHSVM executable")
    a = ap.parse_args(argv)

    dest = (a.dest or chiwawa_dir(a.commit)).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None

    def up_to_date(m: dict | None) -> bool:
        if a.force or m is None or m.get("fetch_version") != FETCH_VERSION:
            return False
        if m.get("commit") != a.commit or not all((dest / f).is_file() for f in m["files"]):
            return False
        t = m["forcing"]["trimmed"]
        if a.no_trim:
            return t is None
        if t is None:
            return False
        return ((not a.start or fmt_date(parse_date(a.start)) == t["start"])
                and (not a.end or fmt_date(parse_date(a.end)) == t["end"]))

    if up_to_date(manifest):
        log(f"Chiwawa case is up to date in {dest}")
    else:
        manifest = fetch(dest, a.commit, a.start, a.end, a.no_trim)
        f = manifest["forcing"]
        total = sum(v["bytes"] for v in manifest["files"].values())
        log(f"Done in {manifest['seconds']} s: {len(manifest['files'])} files, "
            f"{total / 1e6:.1f} MB; {len(f['selected'])} of {len(f['upstream_files'])} "
            f"forcing files")

    if a.verify_selection:
        log(f"Checking the forcing selection with {a.verify_selection} ...")
        res = verify_selection(dest, manifest, a.verify_selection)
        manifest["forcing"]["dhsvm_check"] = res
        manifest_path.write_text(json.dumps(manifest, indent=1))
        if res["missing_from_fetch"]:
            log(f"  DHSVM uses forcing files that were not fetched: "
                f"{res['missing_from_fetch']}")
            return 1
        log(f"  DHSVM selects {res['dhsvm_selected']} files, all fetched "
            f"({len(res['extra_in_fetch'])} fetched files are not used)")
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
