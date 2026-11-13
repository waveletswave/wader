# wader

*Don't judge a river from the bridge.*

wader checks whether a distributed hydrologic model setup is correct, whichever
tool generated it: a preprocessing package, a lab script, or code an AI agent
wrote an hour ago. It does not build setups. It is the referee.

A setup that runs, closes its water balance, and matches the outlet hydrograph
can still be wrong inside the basin. On the PNNL Chiwawa test case, a stream
network with reversed links reproduced the outlet hydrograph (NSE >= 0.98)
while interior reaches carried 85-99.7% too little water. DHSVM ran it without
a complaint. wader wades in and checks the inside.

DHSVM is the first supported model. Every rule records the DHSVM behaviour it
relies on, with a source location in
[pnnl/DHSVM-PNNL](https://github.com/pnnl/DHSVM-PNNL) `e6f333d` and whether
that behaviour was verified in the source, verified by running DHSVM, reported
by others, or not yet verified.

Status: pre-alpha (milestone M1 in progress). Twelve rules so far, for the stream network
files, the binary maps, the model state and the output directory; see below.

## Install

```bash
pip install -e '.[dev]'      # Python >= 3.11; numpy is the only runtime dependency
```

## Use

```bash
# Run the rules. Relative paths in the INPUT file are resolved against the
# directory DHSVM would be started from (default: the current directory).
wader check dhsvm path/to/INPUT --workdir path/to/run/dir
wader check dhsvm path/to/INPUT --json          # machine-readable report

wader rules                                     # rules, parameters, evidence
wader mutations                                 # error-injection operators, and what DHSVM does with each
wader mutate dhsvm path/to/INPUT reverse-network -o mutated/
```

Exit codes: 0 no error findings, 1 error findings (`--strict`: warnings too),
2 a rule could not be evaluated.

## Rules

| ID | What it checks | Level | Caught mutations |
|---|---|---|---|
| DHSVM-MAP-001 | Every binary map DHSVM reads exists and holds exactly rows x columns x data sets values of its number type. A short file stops DHSVM; a long one is read without a message, as different numbers | L0/L1 | `truncate-map`, `float64-dem` |
| DHSVM-MAP-003 | In every basin cell the soil is deeper than the total root depth of the cell's vegetation class; otherwise DHSVM stops before it reads the network | L0 | `shallow-soil` |
| DHSVM-NET-001 | DHSVM accepts every stream network record: readable fields; ID and order of at least 1; slope and length above 0; IDs unique and at most 65535 | L0/L1 | `zero-slope` |
| DHSVM-NET-003 | Every positive outlet ID names a segment; the links form no cycle; some segment leaves the network | L0/L1 | `dangling-outlet`, `cycle` |
| DHSVM-NET-004 | No link runs uphill: a segment's outlet segment is not higher (lowest DEM cell, tolerance 1 m) | L1 | `reverse-network` |
| DHSVM-NET-005 | Linked segments touch in `stream.map.dat` (gap <= 1.5 cells) | L1 | `jump` |
| DHSVM-NET-006 | A single outlet lies within 3 cells of the basin's lowest cell; no outlet receives more than 4 segments directly | L1 | `reverse-network`, `force-wire` |
| DHSVM-NET-007 | DHSVM accepts every stream map record (inside the grid, known segment, no negative value, no negative soil depth in its cell) and uses it (inside the basin mask); no in-basin cell has a channel length of 0; every segment has a record | L0/L1 | `map-swap-row-col`, `zero-length-record` |
| DHSVM-NET-010 | Every class a segment uses is defined, with width, bank height and Manning's n above 0; bank height and Manning's n of the classes in use are plausible | L0/L1/L2 | `missing-class`, `placeholder-classes` |
| DHSVM-NET-013 | The outlet ID is written so DHSVM reads it as an outlet (DHSVM reads `-1` as segment `1`) | L1 | `negative-outlet` |
| DHSVM-RUN-002 | The output directory exists before the run (DHSVM does not create it and stops after reading all its input), and `Output Directory` ends with `/` (otherwise the output lands next to the directory) | L0/L1 | `missing-output-dir`, `output-no-slash` |
| DHSVM-STA-001 | The interception, snow and soil state files that DHSVM restores at start-up exist under the names it builds and hold exactly the data sets it reads for the setup's canopy and soil layers. A short file stops DHSVM; a long one is read without a message, with variables taken from the wrong data sets | L0/L1 | `truncate-state`, `state-extra-layer` |

Levels: L0 = DHSVM stops with an error; L1 = DHSVM accepts the input but results
are wrong, the input is silently changed, or the run crashes; L2 = physical
plausibility. A rule that spans levels marks each finding with its own level
(`level` in the JSON report). Run `wader rules` for the evidence behind each rule.

What wader adds lies mostly in L1. Each line below was reproduced by running
DHSVM `e6f333d` on the Chiwawa case with one thing changed
(`tests/test_dhsvm_runs.py`):

| Input | DHSVM | wader |
|---|---|---|
| A segment drains into its own tributary (a cycle) | exit 0, same warnings and mass balance error; the outlet loses over 99% of its flow | NET-003 |
| One stream map record has length 0 | exit 0; the outlet hydrograph is NaN | NET-007 |
| The DEM is written as 8-byte floats | no read error; stops at the first time step with an unrelated error 47 | MAP-001 |
| Class bank heights are PNNL's template placeholders (0.03-0.10 m) | exit 0; the hydrograph changes. Class widths, in contrast, are never used | NET-010 |
| An outlet ID is written as 65539 | exit 0; read as segment 3 | NET-003 |
| The soil state file is written for one more soil layer (two data sets too many) | exit 0, no message; soil temperatures are read as surface water, and the first outlet flows are over 100 times larger | STA-001 |
| `Output Directory` lacks its final `/` | exit 0; the output is written next to the directory, as `outputStreamflow.Only` and so on | RUN-002 |

wader also reports two things in the Chiwawa test case as distributed. Its
outlet is written as `-1`, which DHSVM reads as segment 1 (NET-013). And its
INPUT names an output directory that the repository does not contain, so DHSVM
reads all the input and then stops until someone creates it (RUN-002).

## Reference cases and data

Reference data are not redistributed: the DHSVM-PNNL repository has no license
file. Scripts fetch what is needed from upstream into a local cache
(`${WADER_CACHE:-~/.cache/wader}`):

```bash
scripts/build_dhsvm.sh                 # serial DHSVM at e6f333d -> ~/.cache/wader/dhsvm/e6f333d/bin/DHSVM
python cases/chiwawa/fetch.py          # TestCase/Chiwawa, only the forcing files DHSVM uses
python cases/chiwawa/run.py            # run DHSVM on Chiwawa for one month
```

## Demo: a reversed network that DHSVM runs and wader catches

```bash
scripts/demo_reverse_network.sh        # about 1 minute; builds and fetches on first use
```

The demo checks the PNNL Chiwawa network (it passes NET-004/005/006), reverses
its main stem with `wader mutate ... reverse-network` (64 links, 49.9 km of
channel), checks it again, and runs DHSVM on both networks for October 1970:

```
                     DHSVM exit  warnings  mass error  outlet flow (seg 494)
PNNL network                  0         1   -0.052 mm             10.19 m3/s
main stem reversed            0         1   -0.052 mm              0.01 m3/s
```

DHSVM gives no sign that anything changed: same exit status, same warnings,
same mass balance error, and the network's total outflow differs by 0.2%.
wader reports 47 links that climb (up to 168 m) and an outlet 39.6 km from the
basin's lowest cell.

The full rule set also flags the original PNNL network (DHSVM-NET-013): its
outlet ID is written as `-1`, which DHSVM's table scanner reads as segment `1`,
so the network's "Totals" outflow is 0 at every step. The saved outlet
hydrograph is unaffected. Two one-month runs that differ only in that field
(`-1` vs `0`) confirm it.

## Development

```bash
pytest                                  # unit tests (synthetic cases)
WADER_REQUIRE_CASES=1 pytest            # also require the Chiwawa case and DHSVM
ruff check .
```

See [CLAUDE.md](CLAUDE.md) for the working rules that apply to every change,
whether made by a person or an AI agent.

## License

MIT. See [LICENSE](LICENSE).
