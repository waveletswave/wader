"""Run set-up rules (DHSVM-RUN-*).

Source references are to pnnl/DHSVM-PNNL@e6f333d (DHSVM/sourcecode/).
"""

from __future__ import annotations

import os

from wader.core.rules import (
    CheckOutput,
    Evidence,
    Finding,
    RuleInfo,
    Severity,
    Status,
    register,
)
from wader.models.dhsvm.case import DhsvmCase

MODEL = "dhsvm"
SRC = "DHSVM-PNNL@e6f333d "

# ------------------------------------------------------------------ RUN-002
RUN_002 = RuleInfo(
    id="DHSVM-RUN-002",
    title="The output directory exists before the run",
    level="L0/L1",
    severity=Severity.ERROR,
    rationale=(
        "DHSVM does not create its output directory. It puts the text of Output Directory "
        "directly in front of each output file name, so the directory which that text leads "
        "into must exist and be writable when DHSVM starts. Otherwise DHSVM stops when it "
        "opens its first output file, after it has read the maps and the network and set "
        "up the forcing (L0). If the text does not end with '/', DHSVM does not stop: it "
        "writes files such as outputAggregated.Values next to the directory instead of into "
        "it (L1). A relative path counts from the directory DHSVM is started from."
    ),
    evidence=(
        Evidence(
            "InitDump stops with error 51 if Output Directory is empty. Otherwise it copies "
            "the text to Dump.Path and prints it directly in front of 'Aggregated.Values', "
            "'Mass.Balance' and 'Mass.Final.Balance', which it opens for writing at once, and "
            "later in front of 'Stream.Flow'. OpenFile stops with error 3 if the file cannot "
            "be opened. InitDump runs after the terrain maps, the network and the forcing "
            "sources have been set up.",
            Status.VERIFIED_SOURCE,
            SRC + "InitDump.c:92-94, 137-146, 195-196; Files.c:38-52; GetInit.c:414-421; "
                  "ReportError.c:74; MainDHSVM.c:157-216",
        ),
        Evidence(
            "Chiwawa with Output Directory = no-such-directory/ stops with 'Cannot open file: "
            "no-such-directory/Aggregated.Values' and exit status 3; with an empty Output "
            "Directory it stops with 'Key not found or invalid entry' and exit status 51. The "
            "Chiwawa case as distributed is such a setup: its INPUT names "
            "../../TestCase/Chiwawa/output/, a directory the repository does not contain, and "
            "DHSVM stops in the same way until it is created. With Output Directory = output "
            "(no '/') and an existing directory output/, DHSVM runs to the end and writes "
            "outputAggregated.Values, outputMass.Balance, outputStreamflow.Only and three "
            "more files into the directory it was started from; output/ stays empty.",
            Status.VERIFIED_EXPERIMENT,
            "tests/test_dhsvm_runs.py::test_dhsvm_stops_on_mutant[missing-output-dir], "
            "::test_dhsvm_stops_on_an_empty_output_directory, "
            "::test_chiwawa_as_distributed_has_no_output_directory, "
            "::test_output_directory_without_a_slash (2026-11-11)",
        ),
    ),
    mutations=("missing-output-dir", "output-no-slash"),
)


@register(MODEL, RUN_002)
def check_run_002(case: DhsvmCase, params: dict) -> CheckOutput:
    text = case.output_prefix
    where = {"entry": "[OUTPUT] Output Directory", "value": text}
    if not text:  # IsEmptyStr (GetInit.c:414-421)
        return CheckOutput(
            summary="Output Directory is not set",
            findings=[Finding(
                RUN_002.id, Severity.ERROR,
                "[OUTPUT] Output Directory is empty or missing; DHSVM stops when it sets up "
                "its output ('Key not found or invalid entry', exit 51)",
                where=where, level="L0")],
            metrics={"value": text})

    # DHSVM writes <text>Aggregated.Values and so on: the directory is what precedes the
    # last '/', and whatever follows it becomes the start of every file name.
    head, slash, tail = text.rpartition("/")
    directory = case.start_path(head + slash) if slash else case.workdir
    shown = os.path.normpath(directory)  # for messages: without "run/x/../.."
    where["directory"] = shown
    first = f"{text}Aggregated.Values"
    findings: list[Finding] = []
    if not directory.is_dir():
        what = "is not a directory" if directory.exists() else "does not exist"
        findings.append(Finding(
            RUN_002.id, Severity.ERROR,
            f"the output directory {shown} {what}; DHSVM stops when it opens its first "
            f"output file ('Cannot open file: {first}', exit 3). Create it before the run",
            where=where, level="L0"))
    elif not os.access(directory, os.W_OK | os.X_OK):
        findings.append(Finding(
            RUN_002.id, Severity.ERROR,
            f"no files can be created in the output directory {shown}; DHSVM stops when "
            f"it opens its first output file ('Cannot open file: {first}', exit 3)",
            where=where, level="L0"))
    if tail:
        named = case.start_path(text)
        aside = (f"; {os.path.normpath(named)} is a directory, but nothing is written into it"
                 if named.is_dir() else "")
        findings.append(Finding(
            RUN_002.id, Severity.WARNING,
            f"[OUTPUT] Output Directory = {text!r} does not end with '/': DHSVM puts the text "
            f"directly in front of each file name and writes {tail}Aggregated.Values, "
            f"{tail}Mass.Balance and so on into {shown}{aside}",
            where=where, level="L1"))
    if any(f.level == "L0" for f in findings):
        summary = f"DHSVM cannot write its output to {shown}"
    elif findings:
        summary = f"output files get the prefix {tail!r} in {shown}"
    else:
        summary = f"the output directory {shown} exists"
    return CheckOutput(summary=summary, findings=findings,
                       metrics={"value": text, "directory": shown, "file_name_prefix": tail})
