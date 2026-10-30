# Reference cases

Reference data are fetched from upstream into a local cache and never committed
(`${WADER_CACHE:-~/.cache/wader}/cases/`). The DHSVM-PNNL repository has no
license file, so its redistribution terms are unclear.

## chiwawa

PNNL's Chiwawa test case (`TestCase/Chiwawa` of pnnl/DHSVM-PNNL `e6f333d`):
446 km², 425 x 300 cells of 90 m, 494 stream segments, Livneh gridded forcing.

| Script | What it does |
|---|---|
| `fetch.py` | Partial clone and sparse checkout of the case. Of the 320 forcing files (3.4 GB), it takes the 13 that DHSVM uses for this basin and trims them to the simulation period. Total download is about 68 MB, in about 10 s. `--verify-selection DHSVM` lets DHSVM choose among empty placeholders for all 320 files and checks that it picks exactly the fetched ones. |
| `run.py` | Runs DHSVM from `<case>/run/<name>/` (default: one month, October 1970). It checks the exit status, the number of time steps (249), NaN values, and the mass balance error, and writes `summary.json`. `--input` runs a mutated setup. |

Layout of the cache:

```
chiwawa-e6f333d/
  MANIFEST.json            commit, files with sha256, forcing selection and its DHSVM check
  TestCase/Chiwawa/        as upstream (INPUT.Chiwawa.Baseline, input/, modelstate/, LivnehForcing/)
  run/<name>/              DHSVM working directories (the INPUT uses ../../TestCase/... paths)
```
