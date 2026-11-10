"""Readers for DHSVM binary maps and table files (stream network, stream map).

Binary maps (format ``BIN``): ``Read2DMatrixBin`` (FileIOBin.c:77-98) reads
NY*NX values of the variable's number type with ``fread`` in the machine's
native byte order and fails only if the file is too short; extra bytes are
ignored. Values are stored row by row, first row = northernmost
(InitTerrainMaps.c:99-105; ``Yorig`` is the north edge, InitConstants.c:460,
data.h:154). Number types come from VarID.c (DEM: NC_FLOAT, VarID.c:51-53;
basin mask: NC_BYTE, VarID.c:54-55). wader assumes a little-endian machine.

Table files are tokenized like DHSVM's flex scanner (tableio.lex, compiled as
the shipped tableio.c when flex is absent):

* tokens are INTEGER ``[0-9]+``, REAL ``[-+]?[0-9]*\\.[0-9]+([-+]?[EeDd][0-9]+)?``,
  WORD ``[A-Za-z][A-Za-z0-9_]+`` and quoted strings (tableio.lex:65-69);
* ``#`` starts a comment (tableio.lex:134-138);
* any other character is echoed to stdout and skipped by flex's default rule
  (tableio.c:1006). An INTEGER has no sign, so ``-1`` is read as ``1``, and
  ``1.5e-3`` is read as REAL ``1.5`` followed by INTEGER ``3``.

Fields are matched to the field definitions as in ``table_get_fields`` /
``handle_field`` (tableio.lex:201-286, 297-365): token *i* is tried against
definition *i*; an INTEGER is accepted for a REAL field; an optional field whose
type does not match is skipped and the next definition is tried. Every mismatch
or missing required field counts as a table error; a file with table errors is
rejected as a whole (e.g. channel.c:472-483), and DHSVM stops.

The readers keep IDs as written. DHSVM stores segment and class IDs in an
``unsigned short`` (channel.h:16), so an ID above ``ID_MAX`` wraps; the rules
report such IDs instead of emulating the wrap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Number types from VarID.c (verified-source)
DEM_DTYPE = np.dtype("<f4")  # VarID.c:51-53, NC_FLOAT
MASK_DTYPE = np.dtype("u1")  # VarID.c:54-55, NC_BYTE
SOIL_DEPTH_DTYPE = np.dtype("<f4")  # VarID.c:58-60, NC_FLOAT

ID_MAX = 65535  # SegmentID and ClassID are unsigned short (channel.h:16)


class MapReadError(Exception):
    pass


@dataclass(frozen=True)
class BinaryMap:
    path: Path
    data: np.ndarray
    extra_bytes: int  # bytes beyond NY*NX values; DHSVM ignores them silently


def read_bin_map(path: str | Path, nrows: int, ncols: int, dtype: np.dtype) -> BinaryMap:
    """Read the first (and usually only) layer of a DHSVM ``BIN`` map."""
    path = Path(path)
    if not path.is_file():
        raise MapReadError(f"map file not found: {path}")
    expected = nrows * ncols * dtype.itemsize
    size = path.stat().st_size
    if size < expected:
        raise MapReadError(
            f"{path}: {size} bytes, but {nrows} x {ncols} values of {dtype.name} need "
            f"{expected}; DHSVM stops here (FileIOBin.c:92-94)"
        )
    data = np.fromfile(path, dtype=dtype, count=nrows * ncols).reshape(nrows, ncols)
    return BinaryMap(path, data, size - expected)


# --------------------------------------------------------------------- tables
_REAL = re.compile(r"[-+]?[0-9]*\.[0-9]+(?:[-+]?[EeDd][0-9]+)?")
_INT = re.compile(r"[0-9]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
_STRTOD = re.compile(r"[-+]?[0-9]*\.[0-9]+(?:[eE][0-9]+)?")


def strtod_prefix(text: str) -> float:
    """Value C ``strtod`` gives a REAL token (tableio.lex:107): it stops at a 'D'
    exponent or at a sign placed before the 'E', so ``1.5D3`` and ``1.5-E3`` are 1.5."""
    m = _STRTOD.match(text)
    return float(m.group(0)) if m else 0.0


@dataclass(frozen=True)
class Token:
    kind: str  # "int", "real", "word", "string"
    value: int | float | str
    start: int  # span in the line
    end: int
    echoed_before: str  # characters DHSVM skipped (and echoed) right before this token


@dataclass(frozen=True)
class TableNote:
    """Something DHSVM's table reader does with a line: ``error`` notes are table
    errors (DHSVM rejects the file), ``ignored`` notes are fields it drops with only
    a message, ``echo`` notes are characters it skips."""

    lineno: int
    message: str
    kind: str = "error"


def as_c_float(value: float) -> float:
    """A table REAL as DHSVM holds it: a C ``float`` (tableio.h:32-35)."""
    return float(np.float32(value))


def scan_line(line: str) -> tuple[list[Token], str, str]:
    """Tokenize one line like tableio.lex. Returns (tokens, trailing echo, all echoed)."""
    tokens: list[Token] = []
    echo, all_echo = "", ""
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if ch in " \t":
            i += 1
            continue
        if ch == "#":
            break
        if ch == '"':
            j, buf = i + 1, []
            while j < n and line[j] != '"':
                if line[j] == "\\" and j + 1 < n and line[j + 1] == '"':
                    buf.append('"')
                    j += 2
                else:
                    buf.append(line[j])
                    j += 1
            tokens.append(Token("string", "".join(buf), i, min(j + 1, n), echo))
            echo, i = "", j + 1
            continue
        best: tuple[str, re.Match[str]] | None = None
        for kind, rx in (("int", _INT), ("real", _REAL), ("word", _WORD)):
            m = rx.match(line, i)
            if m and (best is None or m.end() > best[1].end()):
                best = (kind, m)
        if best is None:  # flex default rule: ECHO and skip one character
            echo += ch
            all_echo += ch
            i += 1
            continue
        kind, m = best
        text = m.group(0)
        if kind == "int":
            value: int | float | str = int(text)
        elif kind == "real":
            value = strtod_prefix(text)
        else:
            value = text
        tokens.append(Token(kind, value, m.start(), m.end(), echo))
        echo, i = "", m.end()
    return tokens, echo, all_echo


@dataclass(frozen=True)
class FieldDef:
    name: str
    kind: str
    required: bool
    words: tuple[str, ...] = ()


def match_fields(tokens: list[Token],
                 defs: tuple[FieldDef, ...]) -> tuple[dict, list[str], list[str]]:
    """Assign tokens to field definitions like ``table_get_fields``/``handle_field``.

    Returns the field values, the table errors DHSVM counts (any of them makes it
    reject the file), and the fields it only prints 'unable to match field' for:
    that branch does not count an error (tableio.lex:270-283), so the field is
    dropped and the record is kept.
    """
    values: dict[str, tuple[int | float | str, Token]] = {}
    errors: list[str] = []
    ignored: list[str] = []

    def handle(tok: Token, offset: int) -> None:
        d = defs[offset]
        if tok.kind == d.kind:
            if d.kind == "word" and tok.value not in d.words:
                errors.append(f"unknown keyword {tok.value!r} for {d.name}")
            else:
                values[d.name] = (tok.value, tok)
        elif tok.kind == "int" and d.kind == "real":
            values[d.name] = (float(tok.value), tok)
        elif d.required:
            errors.append(f"type mismatch: {tok.kind} {tok.value!r} for {d.name}")
        elif offset + 1 < len(defs):
            handle(tok, offset + 1)
        else:
            ignored.append(f"{tok.kind} {tok.value!r} matches no remaining field")

    for i, tok in enumerate(tokens):
        if i >= len(defs):
            errors.append(f"too many fields, expected {len(defs)}")
            continue
        handle(tok, i)
    for d in defs:
        if d.required and d.name not in values:
            errors.append(f"{d.name} required but not read")
    return values, errors, ignored


def scan_table(path: Path, defs: tuple[FieldDef, ...],
               what: str) -> tuple[list[str], list[tuple[int, dict]], list[TableNote]]:
    """Read a table file: its lines, the (line number, field values) of every
    record, and the notes on what DHSVM's reader does with it."""
    if not path.is_file():
        raise MapReadError(f"{what} file not found: {path}")
    lines = path.read_bytes().decode("latin-1").split("\n")
    records: list[tuple[int, dict]] = []
    notes: list[TableNote] = []
    for lineno, line in enumerate(lines, start=1):
        tokens, _, echoed = scan_line(line)
        if echoed.strip("\r"):
            notes.append(TableNote(lineno, f"DHSVM skips (and echoes) {echoed!r}", "echo"))
        if not tokens:
            continue
        values, errors, ignored = match_fields(tokens, defs)
        for e in errors:
            notes.append(TableNote(lineno, f"DHSVM reports an error: {e}", "error"))
        for e in ignored:
            notes.append(TableNote(lineno, "DHSVM prints 'unable to match field' and drops "
                                           f"the field: {e}", "ignored"))
        records.append((lineno, values))
    return lines, records, notes


def replace_token(line: str, index: int, text: str) -> str:
    """``line`` with its ``index``-th table token replaced by ``text`` (right-aligned
    in the old token's width when shorter)."""
    tok = scan_line(line)[0][index]
    if len(text) < tok.end - tok.start:
        text = text.rjust(tok.end - tok.start)
    return line[:tok.start] + text + line[tok.end:]


# ------------------------------------------------------------- stream network
NETWORK_FIELDS = (  # channel.c:333-342
    FieldDef("id", "int", True),
    FieldDef("order", "int", True),
    FieldDef("slope", "real", True),
    FieldDef("length", "real", True),
    FieldDef("class", "int", True),
    FieldDef("outlet", "int", False),
    FieldDef("save", "word", False, ("SAVE",)),
    FieldDef("save_name", "string", False),
)


@dataclass
class Segment:
    id: int
    order: int
    slope: float
    length: float
    cls: int
    outlet: int  # outlet ID as DHSVM reads it; 0 = no downstream segment (channel.c:458-461)
    outlet_signed: int  # outlet ID with a skipped minus sign put back (what the author wrote)
    save: bool
    save_name: str | None
    lineno: int
    order_span: tuple[int, int]
    outlet_span: tuple[int, int] | None  # includes a skipped minus sign
    class_end: int


@dataclass
class StreamNetwork:
    path: Path
    lines: list[str]
    segments: list[Segment] = field(default_factory=list)
    notes: list[TableNote] = field(default_factory=list)

    def by_id(self) -> dict[int, Segment]:
        """First segment per ID, like ``channel_find_segment`` (channel.c:284-299)."""
        out: dict[int, Segment] = {}
        for s in self.segments:
            out.setdefault(s.id, s)
        return out

    def intended_downstream(self) -> dict[int, int | None]:
        """Downstream segment per ID as the file author meant it: outlet IDs <= 0 mean
        'leaves the network'. IDs that name no segment are mapped to None as well."""
        ids = self.by_id()
        return {
            s.id: (s.outlet_signed if s.outlet_signed > 0 and s.outlet_signed in ids else None)
            for s in ids.values()
        }


def read_stream_network(path: str | Path) -> StreamNetwork:
    path = Path(path)
    lines, records, notes = scan_table(path, NETWORK_FIELDS, "stream network")
    net = StreamNetwork(path, lines, notes=notes)
    for lineno, values in records:
        line = lines[lineno - 1]
        if any(k not in values for k in ("id", "order", "slope", "length", "class")):
            continue
        out_val, out_tok = values.get("outlet", (0, None))
        signed = int(out_val)
        span = None
        if out_tok is not None:
            start = out_tok.start
            if start > 0 and line[start - 1] == "-":  # e.g. "-1", read by DHSVM as 1
                signed = -signed
                start -= 1
            span = (start, out_tok.end)
        order_tok = values["order"][1]
        net.segments.append(
            Segment(
                id=int(values["id"][0]),
                order=int(values["order"][0]),
                slope=float(values["slope"][0]),
                length=float(values["length"][0]),
                cls=int(values["class"][0]),
                outlet=int(out_val),
                outlet_signed=signed,
                save="save" in values,
                save_name=str(values["save_name"][0]) if "save_name" in values else None,
                lineno=lineno,
                order_span=(order_tok.start, order_tok.end),
                outlet_span=span,
                class_end=values["class"][1].end,
            )
        )
    return net


def rewrite_network(net: StreamNetwork, downstream: dict[int, int | None],
                    order: dict[int, int], outlet_sentinel: str | None = None) -> str:
    """Text of ``net`` with new outlet IDs and orders; everything else is kept.

    Only fields whose meaning changes are rewritten. A segment that becomes an
    outlet (``downstream[id] is None``) gets outlet ID 0, which DHSVM reads as
    "no downstream segment" (channel.c:458-461). With ``outlet_sentinel`` every
    outlet's field is rewritten to that text instead.
    """
    lines = list(net.lines)
    for s in net.segments:
        if s.id not in downstream:
            continue
        line = lines[s.lineno - 1]
        d = downstream[s.id]
        edits: list[tuple[int, int, str]] = []
        new_order = order.get(s.id, s.order)
        if new_order != s.order:
            edits.append((*s.order_span, str(new_order)))
        if d is None:
            new_out = outlet_sentinel if outlet_sentinel is not None else (
                None if s.outlet_signed <= 0 else "0")
        else:
            new_out = None if s.outlet_signed == d else str(d)
        if new_out is not None:
            if s.outlet_span is None:
                edits.append((s.class_end, s.class_end, " " + new_out))
            elif line[slice(*s.outlet_span)] != new_out:
                edits.append((*s.outlet_span, new_out))
        for start, end, text in sorted(edits, reverse=True):
            if len(text) < end - start:
                text = text.rjust(end - start)
            line = line[:start] + text + line[end:]
        lines[s.lineno - 1] = line
    return "\n".join(lines)


# ----------------------------------------------------------------- stream map
MAP_FIELDS = (  # channel_grid.c:185-194
    FieldDef("col", "int", True),
    FieldDef("row", "int", True),
    FieldDef("segment", "int", True),
    FieldDef("length", "real", True),
    FieldDef("cut_height", "real", True),
    FieldDef("cut_width", "real", True),
    FieldDef("azimuth", "real", False),
    FieldDef("sink", "word", False, ("SINK",)),
)


@dataclass
class StreamMap:
    path: Path
    col: np.ndarray
    row: np.ndarray
    segment: np.ndarray
    length: np.ndarray
    cut_height: np.ndarray
    cut_width: np.ndarray
    lineno: np.ndarray
    notes: list[TableNote] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.segment)

    def cells_by_segment(self, nrows: int, ncols: int) -> dict[int, np.ndarray]:
        """(row, col) cells of each segment, keeping only cells inside the grid
        (DHSVM rejects the others, channel_grid.c:230-257)."""
        ok = (self.row >= 0) & (self.row < nrows) & (self.col >= 0) & (self.col < ncols)
        rc = np.column_stack([self.row[ok], self.col[ok]])
        seg = self.segment[ok]
        order = np.argsort(seg, kind="stable")
        seg, rc = seg[order], rc[order]
        bounds = np.flatnonzero(np.diff(seg)) + 1
        return {
            int(s[0]): cells
            for s, cells in zip(np.split(seg, bounds), np.split(rc, bounds), strict=True)
            if len(s)
        }


def read_stream_map(path: str | Path) -> StreamMap:
    path = Path(path)
    lines, records, notes = scan_table(path, MAP_FIELDS, "stream map")
    rows: list[tuple] = []
    for lineno, values in records:
        if any(k not in values for k in ("col", "row", "segment", "length", "cut_height",
                                         "cut_width")):
            continue
        rows.append((
            int(values["col"][0]), int(values["row"][0]), int(values["segment"][0]),
            float(values["length"][0]), float(values["cut_height"][0]),
            float(values["cut_width"][0]), lineno,
        ))
    arr = np.array(rows, dtype=float).reshape(-1, 7)
    return StreamMap(
        path=path,
        col=arr[:, 0].astype(int),
        row=arr[:, 1].astype(int),
        segment=arr[:, 2].astype(int),
        length=arr[:, 3],
        cut_height=arr[:, 4],
        cut_width=arr[:, 5],
        lineno=arr[:, 6].astype(int),
        notes=notes,
        lines=lines,
    )


# -------------------------------------------------------------- stream classes
CLASS_FIELDS = (  # channel.c:97-107; the last two are required for road classes only
    FieldDef("id", "int", True),
    FieldDef("width", "real", True),
    FieldDef("bank_height", "real", True),
    FieldDef("friction", "real", True),
    FieldDef("infiltration", "real", False),
    FieldDef("crown", "word", False, ("OUTSLOPED", "CROWNED", "INSLOPED")),
)


@dataclass
class ChannelClass:
    id: int
    width: float  # read and required to be > 0, but not used by DHSVM (channel.c:161-166)
    bank_height: float  # 0.75 x bank height is the flow depth in the routing constant
    friction: float  # Manning's n (channel.c:311-314)
    lineno: int


@dataclass
class StreamClasses:
    path: Path
    lines: list[str]
    classes: list[ChannelClass] = field(default_factory=list)
    notes: list[TableNote] = field(default_factory=list)

    def by_id(self) -> dict[int, ChannelClass]:
        """First class per ID, like ``find_channel_class`` (channel.c:71-79)."""
        out: dict[int, ChannelClass] = {}
        for c in self.classes:
            out.setdefault(c.id, c)
        return out


def read_stream_classes(path: str | Path) -> StreamClasses:
    path = Path(path)
    lines, records, notes = scan_table(path, CLASS_FIELDS, "stream class")
    out = StreamClasses(path, lines, notes=notes)
    for lineno, values in records:
        if any(k not in values for k in ("id", "width", "bank_height", "friction")):
            continue
        out.classes.append(ChannelClass(
            id=int(values["id"][0]),
            width=float(values["width"][0]),
            bank_height=float(values["bank_height"][0]),
            friction=float(values["friction"][0]),
            lineno=lineno,
        ))
    return out
