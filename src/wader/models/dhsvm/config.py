"""Read and edit DHSVM configuration ("INPUT") files the way DHSVM reads them.

The reader mirrors pnnl/DHSVM-PNNL@e6f333d ``GetInit.c`` (verified-source):

* ``ReadInitFile`` (GetInit.c:423-457) calls ``fgets(Buffer, BUFSIZE, ...)``
  with ``BUFSIZE`` = 255 (settings.h:91), so one read returns at most 254
  characters, and it performs exactly as many reads as there are newline
  characters in the file (``CountLines``, GetInit.c:485-496). A last line
  without a trailing newline is therefore never read, and a line longer than
  254 characters is split, which also pushes the file's final line(s) out of
  the read loop.
* ``Strip`` (GetInit.c:235-261) drops everything from the first ``#`` and then
  leading and trailing whitespace. Only section headers and lines containing
  ``=`` are kept (GetInit.c:440).
* Section and key names are compared after ``MakeKeyString``
  (GetInit.c:263-291): letters are upper-cased, a tab is deleted, and any other
  whitespace character starts a run of whitespace that collapses to that one
  character. ``Number of\\tGrids`` therefore becomes ``NUMBER OFGRIDS``.
* The first matching section wins (``LocateSection``, GetInit.c:175-202), and
  within it the first matching key (``LocateKey``, GetInit.c:129-173).
* Integers are parsed with ``strtol(s, &end, 0)`` and must consume the whole
  value (``CopyInt``, GetInit.c:329-345): a leading ``0`` means octal and
  ``0x`` hexadecimal. ``CopyUChar`` (GetInit.c:383-399) keeps the low 8 bits.

Editing (:meth:`DhsvmConfig.with_value`) rewrites only the value text of one
line and keeps everything else byte for byte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

BUFSIZE = 255
_C_SPACE = " \t\n\v\f\r"


class ConfigError(Exception):
    """A required entry is missing or cannot be parsed the way DHSVM parses it."""


@dataclass(frozen=True)
class Entry:
    section: str  # normalized with make_key_string
    key: str  # normalized with make_key_string
    value: str  # stripped value text, as DHSVM sees it
    lineno: int  # 1-based physical line number


@dataclass(frozen=True)
class ReadNote:
    lineno: int
    message: str


def make_key_string(text: str) -> str:
    """Python port of ``MakeKeyString`` (GetInit.c:263-291)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\t":
            i += 1
        elif ch in _C_SPACE:
            out.append(ch)
            while i < n and text[i] in _C_SPACE:
                i += 1
        else:
            out.append(ch.upper() if "a" <= ch <= "z" else ch)
            i += 1
    return "".join(out)


def strip_line(text: str) -> str:
    """Python port of ``Strip`` (GetInit.c:235-261)."""
    cut = text.find("#")
    if cut >= 0:
        text = text[:cut]
    return text.strip(_C_SPACE)


def _fgets_reads(text: str) -> list[tuple[int, str]]:
    """The (line number, chunk) pairs ``ReadInitFile`` actually reads."""
    limit = BUFSIZE - 1
    reads: list[tuple[int, str]] = []
    lineno, i, n = 1, 0, len(text)
    while i < n:
        j = text.find("\n", i, i + limit)
        end = j + 1 if j >= 0 else min(i + limit, n)
        reads.append((lineno, text[i:end]))
        if j >= 0:
            lineno += 1
        i = end
    return reads[: text.count("\n")]


def c_int(text: str) -> int | None:
    """Value of ``CopyInt``/``strtol(text, &end, 0)``, or None if DHSVM rejects it."""
    m = re.fullmatch(r"[ \t\n\v\f\r]*([+-]?)(0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)", text)
    if not m:
        return None
    digits = m.group(2)
    if digits[:2] in ("0x", "0X"):
        value = int(digits, 16)
    elif digits.startswith("0"):
        value = int(digits, 8)
    else:
        value = int(digits)
    return -value if m.group(1) == "-" else value


_FLOAT_RE = re.compile(
    r"[ \t\n\v\f\r]*[+-]?(?:(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|inf(?:inity)?|nan)",
    re.IGNORECASE,
)


def c_float(text: str) -> float | None:
    """Value of ``CopyFloat``/``CopyDouble`` (strtod, whole value consumed), or None."""
    if not _FLOAT_RE.fullmatch(text):
        return None
    return float(text)


class DhsvmConfig:
    """A DHSVM INPUT file, parsed with DHSVM's own lookup rules."""

    def __init__(self, text: str, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.text = text
        self.entries: list[Entry] = []
        self.sections: list[tuple[str, int]] = []  # (normalized name, line number)
        self.notes: list[ReadNote] = []
        self._parse()

    @classmethod
    def read(cls, path: str | Path) -> DhsvmConfig:
        # latin-1 keeps a 1:1 byte/character mapping, like the C code
        return cls(Path(path).read_bytes().decode("latin-1"), path)

    # ----------------------------------------------------------------- parsing
    def _parse(self) -> None:
        text = self.text
        reads = _fgets_reads(text)
        physical = text.split("\n")
        for lineno, line in enumerate(physical, start=1):
            if len(line.rstrip("\r")) > BUFSIZE - 2:
                self.notes.append(
                    ReadNote(lineno, f"line is {len(line)} characters long; DHSVM reads at most "
                             f"{BUFSIZE - 1} per read, so it is split (GetInit.c:438)")
                )
        if not text.endswith("\n") and strip_line(physical[-1]):
            self.notes.append(
                ReadNote(len(physical), "last line has no trailing newline, so DHSVM never "
                         f"reads it (GetInit.c:434-438): {physical[-1].strip()!r}")
            )
        section: str | None = None
        for lineno, chunk in reads:
            buf = strip_line(chunk)
            if buf.startswith("[") and "]" in buf:
                name = make_key_string(strip_line(buf[1 : buf.index("]")]))
                section = name
                self.sections.append((name, lineno))
            elif "=" in buf:
                key, _, value = buf.partition("=")
                if section is None:
                    continue  # never reachable through LocateSection
                self.entries.append(
                    Entry(section, make_key_string(strip_line(key)), strip_line(value), lineno)
                )

    # ------------------------------------------------------------------ lookup
    def _section_index(self, section: str) -> int | None:
        target = make_key_string(section)
        for i, (name, _) in enumerate(self.sections):
            if name == target:
                return i
        return None

    def entry(self, section: str, key: str) -> Entry | None:
        """The entry DHSVM's ``GetInitString`` would return, or None."""
        idx = self._section_index(section)
        if idx is None:
            return None
        start = self.sections[idx][1]
        stop = self.sections[idx + 1][1] if idx + 1 < len(self.sections) else None
        target = make_key_string(key)
        for e in self.entries:
            if e.lineno <= start or (stop is not None and e.lineno >= stop):
                continue
            if e.key == target:
                return e
        return None

    def get(self, section: str, key: str, default: str | None = None) -> str | None:
        e = self.entry(section, key)
        return default if e is None else e.value

    def require(self, section: str, key: str) -> Entry:
        e = self.entry(section, key)
        if e is None or e.value == "":
            raise ConfigError(f"[{section}] {key!r} is missing or empty")
        return e

    def get_int(self, section: str, key: str) -> int:
        e = self.require(section, key)
        value = c_int(e.value)
        if value is None:
            raise ConfigError(f"line {e.lineno}: [{section}] {key} = {e.value!r} is not an "
                              "integer as DHSVM reads it (strtol base 0)")
        return value

    def get_uchar(self, section: str, key: str) -> int:
        return self.get_int(section, key) & 0xFF

    def get_float(self, section: str, key: str) -> float:
        e = self.require(section, key)
        value = c_float(e.value)
        if value is None:
            raise ConfigError(f"line {e.lineno}: [{section}] {key} = {e.value!r} is not a number")
        return value

    # ----------------------------------------------------------------- editing
    def with_value(self, section: str, key: str, value: str) -> DhsvmConfig:
        """A copy with the value of ``[section] key`` replaced (comment kept)."""
        e = self.entry(section, key)
        if e is None:
            raise ConfigError(f"[{section}] {key!r} not found; cannot set it")
        lines = self.text.split("\n")
        line = lines[e.lineno - 1]
        eq = line.index("=")
        rest = line[eq + 1 :]
        cut = rest.find("#")
        body, comment = (rest, "") if cut < 0 else (rest[:cut], rest[cut:])
        cr = "\r" if (not comment and body.endswith("\r")) else ""
        body = body[: len(body) - len(cr)] if cr else body
        lead = body[: len(body) - len(body.lstrip(" \t"))]
        new_body = lead + value
        if comment:
            new_body += " " * max(1, len(body) - len(new_body))
        lines[e.lineno - 1] = line[: eq + 1] + new_body + comment + cr
        return DhsvmConfig("\n".join(lines), self.path)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_bytes(self.text.encode("latin-1"))
        return path
