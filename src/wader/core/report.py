"""Check reports: human-readable text and JSON (for CI and AI agents)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from wader import __version__
from wader.core.rules import Outcome, RuleResult, Severity

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_NOT_EVALUATED = 2

_TAG = {
    Outcome.PASS: "PASS",
    Outcome.FAIL: "FAIL",
    Outcome.SKIP: "SKIP",
    Outcome.ERROR: "ERROR",
}


@dataclass
class Report:
    model: str
    input: str
    workdir: str
    context: dict[str, Any] = field(default_factory=dict)
    results: list[RuleResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        findings = [f for r in self.results for f in r.findings]
        out = {s.label: sum(f.severity == s for f in findings) for s in Severity}
        for o in Outcome:
            out[f"rules_{o.value}"] = sum(r.outcome == o for r in self.results)
        return out

    def exit_code(self, strict: bool = False) -> int:
        """0: clean; 1: error findings (or warnings with ``strict``); 2: not evaluated."""
        c = self.counts()
        if c["error"] or (strict and c["warning"]):
            return EXIT_FINDINGS
        if c[f"rules_{Outcome.ERROR.value}"]:
            return EXIT_NOT_EVALUATED
        return EXIT_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "wader_version": __version__,
            "model": self.model,
            "input": self.input,
            "workdir": self.workdir,
            "context": self.context,
            "counts": self.counts(),
            "results": [_result_dict(r) for r in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=_json_default)

    def to_text(self, max_findings: int = 10) -> str:
        lines = [f"wader check: {self.model.upper()} setup {self.input}"]
        lines.append(f"  working directory: {self.workdir}")
        for key, value in self.context.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        width = max((len(r.rule.id) for r in self.results), default=0)
        for r in self.results:
            lines.append(f"{_TAG[r.outcome]:<5} {r.rule.id:<{width}}  {r.rule.title}")
            lines.append(f"{'':<5} {'':<{width}}  {r.summary}")
            shown = [f for f in r.findings if f.severity >= Severity.WARNING] or r.findings
            for f in shown[:max_findings]:
                lines.append(f"{'':<5} {'':<{width}}  - [{f.severity.label}] {f.message}")
            if len(shown) > max_findings:
                lines.append(
                    f"{'':<5} {'':<{width}}  - ... {len(shown) - max_findings} more "
                    "(use --json or --max-findings)"
                )
        c = self.counts()
        lines.append("")
        lines.append(
            f"Result: {c['error']} error(s), {c['warning']} warning(s) from "
            f"{len(self.results)} rule(s): {c['rules_pass']} pass, {c['rules_fail']} fail, "
            f"{c['rules_skip']} skipped, {c['rules_error']} not evaluated"
        )
        return "\n".join(lines)


def _result_dict(r: RuleResult) -> dict[str, Any]:
    info = r.rule
    return {
        "rule": {
            "id": info.id,
            "title": info.title,
            "level": info.level,
            "severity": info.severity.label,
            "rationale": info.rationale,
            "evidence": [
                {"claim": e.claim, "status": e.status.value, "source": e.source}
                for e in info.evidence
            ],
            "mutations": list(info.mutations),
        },
        "outcome": r.outcome.value,
        "summary": r.summary,
        "params": r.params,
        "metrics": r.metrics,
        "findings": [
            {
                "severity": f.severity.label,
                "level": f.level or info.level,
                "message": f.message,
                "where": f.where,
                "values": f.values,
            }
            for f in r.findings
        ],
    }


def _json_default(obj: Any) -> Any:
    # numpy scalars and similar
    for attr in ("item", "tolist"):
        if hasattr(obj, attr):
            return getattr(obj, attr)()
    return str(obj)
