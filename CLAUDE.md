# CLAUDE.md: working rules for wader

These rules apply to every change, whether a person or an AI agent makes it.
They come from section 0 of the project brief.

## What wader is

wader is a referee, not another preprocessor. It checks model setups produced
by any generator: WW-DHSVM, DHSVM_Stream_Toolkit, PNNL scripts, or a script an
AI agent wrote. Do not add setup-generation features, and keep wader
independent of every generator, so that its verdicts can be trusted.

## Rules

1. **Cite every claim about DHSVM behaviour.** Give the source location
   (`file:line` in pnnl/DHSVM-PNNL master `e6f333d`, under
   `DHSVM/sourcecode/`) and one of these statuses:
   - `verified-source`: read in the DHSVM source;
   - `verified-experiment`: reproduced by running DHSVM;
   - `reported`: taken from WW-DHSVM documentation or other people, not re-checked;
   - `unverified`: believed but not yet checked.

   In code, this is the `evidence` of each `RuleInfo`. In docstrings and
   comments, cite `file:line`.
2. **Every rule needs two tests:** a mutation test showing that the rule catches
   its error, and a test on a clean reference case showing that it raises no
   false alarm. Add the mutation operator to
   `src/wader/models/dhsvm/mutate.py` and list it in the rule's `mutations`.
3. **Never commit reference data.** The DHSVM-PNNL repository has no LICENSE
   file, so its redistribution terms are unclear. Fetch data from upstream at
   test time and cache it locally (`${WADER_CACHE:-~/.cache/wader}`).
4. **Do not open pull requests against WW-DHSVM directly.** Write problems up as
   issues first; its maintainer asked for issue-first.
5. **Keep DHSVM output directory paths short** (about 80 characters at most).
   DHSVM builds output file names in fixed 100-character buffers
   (InitDump.c:70, 97; rule RUN-001). Prefer relative output paths.
6. **Write repository documents in English:** README, CLAUDE.md and code comments.

## Conventions

- **Pinned upstream:** DHSVM-PNNL `e6f333d0e70ed72345d52f1ab83e35317fa92458`.
  If the pin changes, every citation must be re-checked.
- **Read inputs the way DHSVM reads them, quirks included:** first match wins
  for sections and keys, `#` starts a comment, integers go through
  `strtol(..., 0)`, and table integers carry no sign. See
  `models/dhsvm/config.py` and `models/dhsvm/io.py`. When DHSVM's reading
  differs from what the file's author meant, report it as a finding. Do not
  silently pick one reading.
- **Relative paths:** paths in a DHSVM INPUT file resolve against the directory
  DHSVM is started from, not against the INPUT file's directory.
- **Rule IDs:** `DHSVM-<GROUP>-<NNN>`, where GROUP is one of CFG, MAP, NET,
  STA, MET or RUN. Keep the numbering of the brief's rule table.
- **Levels and severities:**
  - L0: DHSVM stops with an error.
  - L1: DHSVM accepts the input, but the results are wrong, the input is
    silently changed, or the run crashes.
  - L2: physical plausibility.

  Severities are error, warning and info.
- **Tests that need data:** mark them `case_data` (a fetched reference case)
  or `dhsvm` (a DHSVM executable). They skip when the data are missing, unless
  `WADER_REQUIRE_CASES=1` is set. CI sets it, so these tests can never be
  skipped silently there.

## Commands

```bash
pip install -e '.[dev]'
ruff check .
pytest                                   # synthetic unit tests
scripts/build_dhsvm.sh                   # serial DHSVM, cached
python cases/chiwawa/fetch.py            # Chiwawa reference case, cached
python cases/chiwawa/run.py              # one-month DHSVM run
WADER_REQUIRE_CASES=1 pytest             # everything, as in CI
scripts/demo_reverse_network.sh          # the reversed-network demo
```

## Layout

```
src/wader/core/              rules, findings, registry, reports (model-independent)
src/wader/models/dhsvm/      config.py  INPUT parser (DHSVM semantics)
                             io.py      binary maps, stream network/map tables
                             case.py    a setup as DHSVM sees it (paths, grid, maps)
                             met.py     gridded-met file selection (InitGridMet port)
                             rules/     net.py (NET-*); more groups to come
                             mutate.py  error-injection operators
src/wader/cli.py             wader check | rules | mutate | mutations
cases/chiwawa/               fetch.py (sparse checkout), run.py (DHSVM run)
scripts/                     build_dhsvm.sh, demo_reverse_network.sh
tests/                       unit tests and conformance tests (Chiwawa and mutations)
```
