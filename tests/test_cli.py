from __future__ import annotations

import json

from wader.cli import main


def test_check_clean_and_mutated(synthetic, tmp_path, capsys):
    path, work = synthetic
    assert main(["check", "dhsvm", str(path), "--workdir", str(work)]) == 0
    assert "PASS  DHSVM-NET-004" in capsys.readouterr().out

    out_dir = tmp_path / "mut"
    assert main(["mutate", "dhsvm", str(path), "reverse-network", "-o", str(out_dir),
                 "--workdir", str(work)]) == 0
    capsys.readouterr()
    mutated = out_dir / "INPUT.reverse-network"
    assert main(["check", "dhsvm", str(mutated), "--workdir", str(work), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    fails = {r["rule"]["id"] for r in report["results"] if r["outcome"] == "fail"}
    assert fails == {"DHSVM-NET-004", "DHSVM-NET-006"}
    assert report["counts"]["error"] >= 3


def test_check_select_and_strict(synthetic, tmp_path, capsys):
    path, work = synthetic
    out_dir = tmp_path / "mut"
    main(["mutate", "dhsvm", str(path), "negative-outlet", "-o", str(out_dir),
          "--workdir", str(work)])
    capsys.readouterr()
    mutated = str(out_dir / "INPUT.negative-outlet")
    assert main(["check", "dhsvm", mutated, "--workdir", str(work), "--rules", "NET-013"]) == 0
    assert main(["check", "dhsvm", mutated, "--workdir", str(work), "--rules", "NET-013",
                 "--strict"]) == 1
    text = capsys.readouterr().out
    assert "DHSVM-NET-004" not in text and "read as 1" in text


def test_unreadable_input(tmp_path, capsys):
    assert main(["check", "dhsvm", str(tmp_path / "nope"), "--workdir", str(tmp_path)]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_rules_and_mutations_listing(capsys):
    assert main(["rules", "--json"]) == 0
    ids = [r["id"] for r in json.loads(capsys.readouterr().out)]
    assert {"DHSVM-NET-004", "DHSVM-NET-005", "DHSVM-NET-006", "DHSVM-NET-013"} <= set(ids)
    assert main(["mutations"]) == 0
    assert "reverse-network" in capsys.readouterr().out
