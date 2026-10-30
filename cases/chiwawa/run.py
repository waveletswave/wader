#!/usr/bin/env python3
"""Run DHSVM on the fetched Chiwawa case and check that the run completed.

By default runs one month (Model Start 10/01/1970-00 to 11/01/1970-00).

The run directory is ``<case>/run/<name>/``: two levels below the case root,
as the INPUT file's ``../../TestCase/...`` paths require (DHSVM opens relative
paths against its working directory, Files.c:38-50). A copy of the INPUT is
written there that differs only in ``[TIME] Model End`` and ``[OUTPUT] Output
Directory`` (``output/``: short and relative, see RUN-001).

Checks after the run:

* DHSVM exits with status 0 and prints "END OF MODEL RUN";
* Streamflow.Only has one record per time step, Start..End inclusive, and no NaN;
* the final mass balance error is below 1 mm.

``--input`` runs another INPUT file (for example one written by ``wader mutate``)
from a run directory of the same case. A summary is written to
``<run>/summary.json`` and the last line of stdout is the run directory.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from wader.cache import chiwawa_dir, dhsvm_executable
from wader.models.dhsvm.config import DhsvmConfig

DEFAULT_END = "11/01/1970-00"
DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})-(\d{1,2})")


def parse_date(text: str) -> datetime:
    m = DATE_RE.match(text.strip())
    if not m:
        raise SystemExit(f"not a DHSVM date (MM/DD/YYYY-HH): {text!r}")
    mo, d, y, h = (int(g) for g in m.groups())
    return datetime(y, mo, d, h)


def read_streamflow(path: Path) -> tuple[list[str], list[str], list[list[float]]]:
    """Streamflow.Only: a header 'DATE name...' then one line per time step."""
    lines = path.read_text().splitlines()
    names = lines[0].split()[1:] if lines else []
    dates, rows = [], []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        dates.append(parts[0])
        if len(parts) > 1:
            rows.append([float(x) for x in parts[1:]])
    return names, dates, rows


def mass_error(path: Path) -> float:
    m = re.search(r"Mass Error \(mm\)\.*\s+(-?[0-9.eE+-]+)", path.read_text())
    if not m:
        raise SystemExit(f"no mass error in {path}")
    return float(m.group(1))


def totals_outflow(path: Path) -> float:
    """Sum of the 'Totals' outflow column of Stream.Flow (channel.c:653-656)."""
    total = 0.0
    for line in path.read_text().splitlines():
        if line.endswith('"Totals"'):
            total += float(line.split()[3])
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", type=Path, default=None, help="fetched case (default: cache)")
    ap.add_argument("--input", type=Path, default=None,
                    help="INPUT file to run (default: the case's INPUT.Chiwawa.Baseline)")
    ap.add_argument("--name", default=None, help="run directory name (default: from --input)")
    ap.add_argument("--end", default=DEFAULT_END, help=f"Model End (default {DEFAULT_END})")
    ap.add_argument("--dhsvm", type=Path, default=None,
                    help="DHSVM executable (default: $WADER_DHSVM or the cached build)")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args(argv)

    case = (a.case or chiwawa_dir()).expanduser().resolve()
    manifest_path = case / "MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no fetched case in {case}; run cases/chiwawa/fetch.py first")
    manifest = json.loads(manifest_path.read_text())
    dhsvm = a.dhsvm or dhsvm_executable()
    if dhsvm is None or not Path(dhsvm).is_file():
        raise SystemExit("no DHSVM executable; run scripts/build_dhsvm.sh or pass --dhsvm")
    dhsvm = Path(dhsvm).resolve()

    input_path = (a.input or case / manifest["input"]).resolve()
    name = a.name or ("baseline" if a.input is None else input_path.name.replace("INPUT.", ""))
    work = case / "run" / name
    config = DhsvmConfig.read(input_path)
    start = parse_date(config.require("TIME", "Model Start").value)
    end = parse_date(a.end)
    step = timedelta(hours=config.get_int("TIME", "Time Step"))
    trimmed = manifest["forcing"]["trimmed"]
    if trimmed and end > parse_date(trimmed["end"]):
        raise SystemExit(f"forcing was trimmed to {trimmed['end']}; fetch with a later --end")

    if work.exists():
        shutil.rmtree(work)
    (work / "output").mkdir(parents=True)
    config = config.with_value("TIME", "Model End", a.end)
    config = config.with_value("OUTPUT", "Output Directory", "output/")
    config.write(work / "INPUT")

    print(f"Running DHSVM in {work} ({start:%Y-%m-%d} to {end:%Y-%m-%d}) ...", file=sys.stderr)
    t0 = time.time()
    with (work / "dhsvm.stdout.log").open("w") as out, (work / "dhsvm.stderr.log").open("w") as err:
        proc = subprocess.run([str(dhsvm), "INPUT"], cwd=work, stdout=out, stderr=err,
                              timeout=a.timeout)
    seconds = time.time() - t0

    problems = []
    stdout = (work / "dhsvm.stdout.log").read_text(errors="replace")
    if proc.returncode != 0:
        problems.append(f"DHSVM exited with status {proc.returncode}")
    if "END OF MODEL RUN" not in stdout:
        problems.append("DHSVM did not report END OF MODEL RUN")
    summary: dict = {"run_dir": str(work), "input": str(input_path), "dhsvm": str(dhsvm),
                     "start": f"{start:%m/%d/%Y-%H}", "end": a.end,
                     "exit_code": proc.returncode, "seconds": round(seconds, 1)}
    flow_file = work / "output" / "Streamflow.Only"
    if not problems:
        names, dates, rows = read_streamflow(flow_file)
        expected = int((end - start) / step) + 1
        if len(dates) != expected:
            problems.append(f"Streamflow.Only has {len(dates)} time steps, expected {expected}")
        if any(math.isnan(v) for r in rows for v in r):
            problems.append("Streamflow.Only contains NaN")
        err = mass_error(work / "output" / "Mass.Final.Balance")
        if abs(err) >= 1.0:
            problems.append(f"mass balance error {err} mm")
        summary.update(
            time_steps=len(dates),
            saved_segments=names,
            mean_flow_m3_per_step={n: sum(r[i] for r in rows) / len(rows)
                                   for i, n in enumerate(names)} if rows else {},
            mass_error_mm=err,
            totals_outflow_m3=totals_outflow(work / "output" / "Stream.Flow"),
        )
    summary["problems"] = problems
    (work / "summary.json").write_text(json.dumps(summary, indent=1))

    if problems:
        print("RUN FAILED: " + "; ".join(problems), file=sys.stderr)
        print("\n".join(stdout.splitlines()[-15:]), file=sys.stderr)
        return 1
    flows = ", ".join(f"{n} {v / (step.total_seconds()):.2f} m3/s"
                      for n, v in summary["mean_flow_m3_per_step"].items())
    print(f"OK in {seconds:.0f} s: {summary['time_steps']} time steps, mean flow {flows}, "
          f"mass balance error {summary['mass_error_mm']} mm, network 'Totals' outflow "
          f"{summary['totals_outflow_m3']:.4g} m3", file=sys.stderr)
    print(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
