"""Command-line interface.

    wader check dhsvm INPUT [--workdir DIR] [--rules NET-004,...] [--json] [--strict]
    wader rules [dhsvm] [--json]
    wader mutate dhsvm INPUT MUTATION -o OUT_DIR [--workdir DIR]
    wader mutations [dhsvm]

Exit codes of ``check``: 0 no error findings, 1 error findings (or warnings
with --strict), 2 some rule could not be evaluated or the input is unreadable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from wader import __version__
from wader.core import InputProblem, registered_rules
from wader.core.report import EXIT_NOT_EVALUATED
from wader.models import dhsvm
from wader.models.dhsvm.mutate import MUTATIONS, MutationError, mutate_case

MODELS = ("dhsvm",)


def _params(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--param expects NAME=VALUE, got {item!r}")
        out[name.strip()] = value.strip()
    return out


def _cmd_check(args: argparse.Namespace) -> int:
    rules = [r for r in (args.rules or "").split(",") if r.strip()]
    try:
        report = dhsvm.check(args.input, args.workdir, rules, _params(args.param))
    except (InputProblem, KeyError) as exc:
        print(f"wader: {exc}", file=sys.stderr)
        return EXIT_NOT_EVALUATED
    print(report.to_json() if args.json else report.to_text(args.max_findings))
    return report.exit_code(strict=args.strict)


def _cmd_rules(args: argparse.Namespace) -> int:
    rules = registered_rules(args.model)
    if args.json:
        print(json.dumps([
            {
                "id": r.info.id, "title": r.info.title, "level": r.info.level,
                "severity": r.info.severity.label, "rationale": r.info.rationale,
                "params": r.info.params, "mutations": list(r.info.mutations),
                "evidence": [{"claim": e.claim, "status": e.status.value, "source": e.source}
                             for e in r.info.evidence],
            }
            for r in rules
        ], indent=2))
        return 0
    for r in rules:
        i = r.info
        print(f"{i.id}  [{i.level}, {i.severity.label}]  {i.title}")
        print(f"    {i.rationale}")
        if i.params:
            print("    params: " + ", ".join(f"{k}={v}" for k, v in i.params.items()))
        if i.mutations:
            print("    caught mutations: " + ", ".join(i.mutations))
        for e in i.evidence:
            print(f"    - ({e.status.value}) {e.claim} [{e.source}]")
        print()
    return 0


def _cmd_mutate(args: argparse.Namespace) -> int:
    try:
        case = dhsvm.DhsvmCase(args.input, args.workdir)
        input_path, mutated = mutate_case(case, args.mutation, args.out_dir, _params(args.param))
    except (InputProblem, MutationError) as exc:
        print(f"wader: {exc}", file=sys.stderr)
        return EXIT_NOT_EVALUATED
    print(f"{mutated.info.name}: {mutated.info.title}")
    print(f"  should be caught by: {', '.join(mutated.info.targets)}")
    print(f"  DHSVM: {mutated.info.dhsvm}")
    for line in mutated.changes[:12]:
        print(f"  {line}")
    if len(mutated.changes) > 12:
        print(f"  ... {len(mutated.changes) - 12} more changes")
    print(f"wrote {input_path}")
    return 0


def _cmd_mutations(args: argparse.Namespace) -> int:
    for name, (info, _) in sorted(MUTATIONS.items()):
        print(f"{name}: {info.title}")
        print(f"    caught by: {', '.join(info.targets)}")
        print(f"    DHSVM: {info.dhsvm}")
        print(f"    {info.description}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wader",
        description="Check distributed hydrologic model setups. Don't judge a river from "
                    "the bridge.",
    )
    p.add_argument("--version", action="version", version=f"wader {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="run the rules on a model setup")
    c.add_argument("model", choices=MODELS)
    c.add_argument("input", help="model configuration file (DHSVM: the INPUT file)")
    c.add_argument("--workdir", help="directory the model would be started from; relative "
                                     "paths in the INPUT file are resolved against it "
                                     "(default: current directory, as DHSVM does)")
    c.add_argument("--rules", help="comma-separated rule ids or id fragments, e.g. NET-004")
    c.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                   help="override a rule parameter, e.g. tolerance_m=2")
    c.add_argument("--json", action="store_true", help="print the report as JSON")
    c.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    c.add_argument("--max-findings", type=int, default=10, metavar="N",
                   help="findings to print per rule (default 10)")
    c.set_defaults(func=_cmd_check)

    r = sub.add_parser("rules", help="list rules with their evidence")
    r.add_argument("model", nargs="?", default="dhsvm", choices=MODELS)
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=_cmd_rules)

    m = sub.add_parser("mutate", help="write a copy of a setup with one known error injected")
    m.add_argument("model", choices=MODELS)
    m.add_argument("input")
    m.add_argument("mutation", choices=sorted(MUTATIONS))
    m.add_argument("-o", "--out-dir", required=True)
    m.add_argument("--workdir", help="as for check")
    m.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    m.set_defaults(func=_cmd_mutate)

    ms = sub.add_parser("mutations", help="list mutation operators")
    ms.add_argument("model", nargs="?", default="dhsvm", choices=MODELS)
    ms.set_defaults(func=_cmd_mutations)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
