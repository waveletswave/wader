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

Status: pre-alpha (milestone M0). The rule set is small; see below.

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
wader mutations                                 # error-injection operators
wader mutate dhsvm path/to/INPUT reverse-network -o mutated/
```

Exit codes: 0 no error findings, 1 error findings (`--strict`: warnings too),
2 a rule could not be evaluated.

## Rules

| ID | What it checks | Level | Caught mutation |
|---|---|---|---|
| DHSVM-NET-004 | No link runs uphill: a segment's outlet segment is not higher (lowest DEM cell, tolerance 1 m) | L1 | `reverse-network` |
| DHSVM-NET-005 | Linked segments touch in `stream.map.dat` (gap <= 1.5 cells) | L1 | `jump` |
| DHSVM-NET-006 | A single outlet lies within 3 cells of the basin's lowest cell; no outlet receives more than 4 segments directly | L1 | `reverse-network`, `force-wire` |
| DHSVM-NET-013 | The outlet ID is written so DHSVM reads it as an outlet (DHSVM reads `-1` as segment `1`) | L1 | `negative-outlet` |

Levels: L0 = DHSVM stops with an error; L1 = DHSVM accepts the input but results
are wrong, the input is silently changed, or the run crashes; L2 = physical
plausibility. Run `wader rules` for the evidence behind each rule.

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
