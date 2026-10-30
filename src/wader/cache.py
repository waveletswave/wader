"""Where fetched reference data and the DHSVM build live (never in the repository)."""

from __future__ import annotations

import os
from pathlib import Path

DHSVM_REPO = "https://github.com/pnnl/DHSVM-PNNL.git"
DHSVM_COMMIT = "e6f333d0e70ed72345d52f1ab83e35317fa92458"


def cache_root() -> Path:
    """``$WADER_CACHE`` or ``~/.cache/wader``."""
    env = os.environ.get("WADER_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "wader"


def chiwawa_dir(commit: str = DHSVM_COMMIT) -> Path:
    """Default location of the fetched Chiwawa case (``cases/chiwawa/fetch.py``)."""
    env = os.environ.get("WADER_CHIWAWA")
    return Path(env).expanduser() if env else cache_root() / "cases" / f"chiwawa-{commit[:7]}"


def dhsvm_executable(commit: str = DHSVM_COMMIT) -> Path | None:
    """``$WADER_DHSVM`` or the binary built by ``scripts/build_dhsvm.sh``, if present."""
    env = os.environ.get("WADER_DHSVM")
    if env:
        return Path(env).expanduser()
    default = cache_root() / "dhsvm" / commit[:7] / "bin" / "DHSVM"
    return default if default.is_file() else None
