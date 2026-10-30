"""DHSVM plugin: INPUT parsing, readers, rules, mutations."""

from __future__ import annotations

from pathlib import Path

from wader.core import Report, run_rule, select_rules
from wader.models.dhsvm import rules as _rules  # noqa: F401  (registers the rules)
from wader.models.dhsvm.case import DhsvmCase

MODEL = "dhsvm"


def check(input_path: str | Path, workdir: str | Path | None = None,
          rules: list[str] | None = None, params: dict | None = None) -> Report:
    """Run the selected DHSVM rules on a setup and return the report."""
    case = DhsvmCase(input_path, workdir)
    results = [run_rule(r, case, params) for r in select_rules(MODEL, rules)]
    return Report(MODEL, str(input_path), str(case.workdir.resolve()), case.summary(), results)


__all__ = ["DhsvmCase", "MODEL", "check"]
