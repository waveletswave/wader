"""Rule metadata, findings, results, and the rule registry.

Every rule carries the evidence for the DHSVM (or other model) behaviour it
relies on. Each piece of evidence states where it comes from and how well it
is established, using the four statuses of the project brief:

* ``verified-source``      - read in the model source (file:line given)
* ``verified-experiment``  - reproduced by running the model
* ``reported``             - taken from documentation or other people, not re-checked
* ``unverified``           - believed but not yet checked
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


class Severity(enum.IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2

    @property
    def label(self) -> str:
        return self.name.lower()


class Status(enum.StrEnum):
    VERIFIED_SOURCE = "verified-source"
    VERIFIED_EXPERIMENT = "verified-experiment"
    REPORTED = "reported"
    UNVERIFIED = "unverified"


class Outcome(enum.StrEnum):
    PASS = "pass"  # evaluated, nothing at warning level or above
    FAIL = "fail"  # evaluated, at least one warning or error finding
    SKIP = "skip"  # not applicable to this setup
    ERROR = "error"  # could not be evaluated (unreadable or missing input)


LEVELS = {
    "L0": "DHSVM itself stops with an error",
    "L1": "DHSVM accepts the input but results are wrong, input is silently changed, "
    "or the run crashes",
    "L2": "physical plausibility",
}


@dataclass(frozen=True)
class Evidence:
    claim: str
    status: Status
    source: str


@dataclass(frozen=True)
class RuleInfo:
    id: str
    title: str
    level: str
    severity: Severity
    rationale: str
    evidence: tuple[Evidence, ...]
    mutations: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"{self.id}: unknown level {self.level!r}")


@dataclass
class Finding:
    rule_id: str
    severity: Severity
    message: str
    where: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleResult:
    rule: RuleInfo
    outcome: Outcome
    summary: str
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


class NotApplicable(Exception):
    """Raised by a check when the rule does not apply to this setup."""


class InputProblem(Exception):
    """Raised by a check when the input it needs cannot be read."""


@dataclass
class CheckOutput:
    """What a check function returns; the runner turns it into a RuleResult."""

    summary: str
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


CheckFunction = Callable[[Any, dict[str, Any]], CheckOutput]


@dataclass(frozen=True)
class RegisteredRule:
    model: str
    info: RuleInfo
    check: CheckFunction


_REGISTRY: dict[str, dict[str, RegisteredRule]] = {}


def register(model: str, info: RuleInfo) -> Callable[[CheckFunction], CheckFunction]:
    """Decorator: register ``check(case, params) -> CheckOutput`` under ``info.id``."""

    def deco(fn: CheckFunction) -> CheckFunction:
        rules = _REGISTRY.setdefault(model, {})
        if info.id in rules:
            raise ValueError(f"duplicate rule id {info.id}")
        rules[info.id] = RegisteredRule(model, info, fn)
        return fn

    return deco


def registered_rules(model: str) -> list[RegisteredRule]:
    return [r for _, r in sorted(_REGISTRY.get(model, {}).items())]


def select_rules(model: str, patterns: Iterable[str] | None) -> list[RegisteredRule]:
    """Rules whose id contains any of the patterns (case-insensitive); all if none."""
    rules = registered_rules(model)
    pats = [p.upper() for p in (patterns or []) if p]
    if not pats:
        return rules
    chosen = [r for r in rules if any(p in r.info.id.upper() for p in pats)]
    if not chosen:
        raise KeyError(f"no {model} rule matches {', '.join(pats)}")
    return chosen


def run_rule(rule: RegisteredRule, case: Any,
             overrides: dict[str, Any] | None = None) -> RuleResult:
    """Run one rule. ``overrides`` maps parameter names to new values; unknown names
    are ignored so one override set can be passed to every rule."""
    params = dict(rule.info.params)
    for key, value in (overrides or {}).items():
        if key in params:
            params[key] = type(params[key])(value)
    try:
        out = rule.check(case, params)
    except NotApplicable as exc:
        return RuleResult(rule.info, Outcome.SKIP, str(exc), params=params)
    except InputProblem as exc:
        return RuleResult(rule.info, Outcome.ERROR, str(exc), params=params)
    failed = any(f.severity >= Severity.WARNING for f in out.findings)
    outcome = Outcome.FAIL if failed else Outcome.PASS
    return RuleResult(rule.info, outcome, out.summary, out.findings, out.metrics, params)
