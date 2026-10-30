#!/usr/bin/env bash
# Demo: a reversed stream network that DHSVM runs without complaint and wader catches.
#
#   scripts/demo_reverse_network.sh            # one month (October 1970), about 1 minute
#   DEMO_END=10/11/1970-00 scripts/demo_reverse_network.sh   # ten days, faster
#
# Needs an installed wader (pip install -e .). Builds DHSVM and fetches the
# Chiwawa case first if they are not cached yet. Exits non-zero if wader flags
# the original network, misses the reversed one, or DHSVM rejects either.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
end="${DEMO_END:-11/01/1970-00}"
py="${PYTHON:-python}"
bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }

dhsvm="${WADER_DHSVM:-$("$repo/scripts/build_dhsvm.sh" | tail -n1)}"
case_dir="$("$py" "$repo/cases/chiwawa/fetch.py" | tail -n1)"
work="$case_dir/run/demo"
rm -rf "$work" && mkdir -p "$work" && cd "$work"
input=../../TestCase/Chiwawa/INPUT.Chiwawa.Baseline   # paths in it are relative to here
rules=NET-004,NET-005,NET-006

bold "1. wader check: the PNNL Chiwawa stream network (494 segments)"
wader check dhsvm "$input" --rules "$rules"

bold "2. Inject a known error: reverse the main stem"
wader mutate dhsvm "$input" reverse-network -o mutated

bold "3. wader check: the mutated network"
set +e
wader check dhsvm mutated/INPUT.reverse-network --rules "$rules" --max-findings 3
status=$?
set -e
if [[ $status -ne 1 ]]; then echo "wader did not flag the reversed network" >&2; exit 1; fi

bold "4. DHSVM runs both networks (to $end)"
"$py" "$repo/cases/chiwawa/run.py" --dhsvm "$dhsvm" --end "$end" --name demo-original
"$py" "$repo/cases/chiwawa/run.py" --dhsvm "$dhsvm" --end "$end" --name demo-reversed \
  --input mutated/INPUT.reverse-network

bold "5. Side by side"
"$py" - "$case_dir/run" <<'EOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
rows = [("PNNL network", "demo-original"), ("main stem reversed", "demo-reversed")]
print(f"{'':20s} {'DHSVM exit':>10s} {'warnings':>9s} {'mass error':>11s} "
      f"{'outlet flow (seg 494)':>22s}")
for label, name in rows:
    s = json.loads((run / name / "summary.json").read_text())
    log = (run / name / "dhsvm.stdout.log").read_text(errors="replace").lower()
    warnings = sum("warning" in line for line in log.splitlines())
    q = s["mean_flow_m3_per_step"]["Chiwawa"] / 10800
    print(f"{label:20s} {s['exit_code']:>10d} {warnings:>9d} {s['mass_error_mm']:>8.3f} mm "
          f"{q:>17.2f} m3/s")
EOF
echo
echo "DHSVM accepts both. Only wader, by checking inside the network, sees that the second is wrong."

bold "6. Bonus: what the full rule set says about the original PNNL network"
wader check dhsvm "$input" --rules NET-013
