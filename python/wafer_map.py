"""
Wafer-row batch vocabulary: the wafer map, the VK4 filename grammar and the run plan.

A wafer carries a GRID of samples. Each sample is one uniform-cell dataset (one pin geometry, one
laser dose) imaged as one or more disjoint snapshots, i.e. exactly what ``run_sample.py
--snapshots`` analyses today. ``run_row.py`` runs a whole wafer ROW in one go; this module holds
everything that decision needs and is PURE: it parses names and CSV text, and composes a plan.
Nothing here reads a VK4, parses a DXF, or writes to disk (``read_wafer_map`` reads one CSV).

That purity is deliberate and load-bearing twice over:
  * ``ui_shared.py`` imports it at module scope, so it must not drag in numpy/pandas/matplotlib, and
  * the whole plan -- name parsing, grouping, DXF pairing, skips -- is unit-testable in selftest.py
    without a single Keyence file.

Two conventions come from the user's wafer:

VK4 filenames carry a compact ``_{col}{row}_`` token before the snapshot label::

    072230_PFLMTIM_D50_11_Center.vk4     -> col 1, row 1, snapshot 'Center'
    072230_PFLMTIM_D50_13_TopLeft.vk4    -> col 1, row 3, snapshot 'TopLeft'

FIRST digit is the column, SECOND is the row. An explicit ``C{col}R{row}`` token is also accepted
so the scheme survives a wafer wider than nine columns.

``csv/wafer_map.csv`` declares, per (row, col): the laser dose, the pin geometry and the lattice.
Geometry is declared PER LINE and never inferred -- on this wafer the column->geometry pairing
REVERSES between wafer rows 1-2 and row 4, so any inference rule would be wrong half the time.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

MAX_INDEX = 9                       # the compact 2-digit CR token can only address 1..9
DEFAULT_MAP_NAME = "wafer_map.csv"

# Snapshot label ordering, so snapshot_id 1..N means the same thing in every sample of the row and
# the montage panels line up column to column. Unknown labels sort after these, alphabetically.
SNAPSHOT_ORDER = ("Center", "TopLeft", "TopRight", "BottomLeft", "BottomRight")

MAP_REQUIRED_HEADERS = ("row", "col")
MAP_KNOWN_HEADERS = ("row", "col", "laser", "geometry", "lattice", "skip", "note", "dxf")
MAP_META_KEYS = ("date", "dxf_dir", "vk4_dir")


# ----------------------------------------------------------- filesystem-safe names #
# The one sanitiser for every results-folder name, so the UI and the CLI cannot drift apart on what
# a sample folder is called; '[' and ']' are added because _prepare_output_transaction globs
# f".{final_dir.name}.staging-*" and brackets are glob metacharacters (legal on NTFS, so the old
# sanitizer let them through).
_SLASH = re.compile(r"[/\\]")
_INVALID_NAME = re.compile(r'[<>:"|?*\[\]\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_name(name):
    """Turn a UI sample name into a filesystem-safe folder name (Windows-safe): slashes become
    spaces; any other character illegal in a Windows path -- or special to ``glob`` -- becomes an
    underscore; a Windows reserved device name (CON, PRN, COM1, ...) gets a trailing underscore so
    it can be a real folder.

    NOTE: '/' -> space is intentional (per the sample-naming convention) and is not one-to-one, so
    e.g. 'D100/D50' and 'D100 D50' still map to the same folder -- that collision is inherent to the
    space convention, not a sanitizer bug."""
    name = _SLASH.sub(" ", str(name))
    name = _INVALID_NAME.sub("_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if name.upper().split(".")[0] in _RESERVED:
        name += "_"
    return name or "unnamed"


# ------------------------------------------------------------------ scalar parsers #
_NUM = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_DOSE_PS = re.compile(rf"P\s*(\d+)\s*_?\s*S\s*({_NUM})", re.IGNORECASE)
_DOSE_SP = re.compile(rf"S\s*({_NUM})\s*_?\s*P\s*(\d+)", re.IGNORECASE)
_GEOM_RE = re.compile(rf"D\s*({_NUM})\s*[_\-\s]*P\s*({_NUM})", re.IGNORECASE)

_LATTICE_ALIASES = {
    "hex": "hex", "hexagonal": "hex", "triangular": "hex", "tri": "hex", "hexagon": "hex",
    "square": "square", "sq": "square", "rect": "square", "rectangular": "square",
    "grid": "square",
    # A 'staggered' (brick / centered-rectangular) lattice: rows at spacing p, offset by p/2. Hex is
    # the SPECIAL CASE of that where the offset makes every neighbour distance equal, so the two are
    # different lattices and must never be aliased together.
    "stagger": "stagger", "staggered": "stagger", "brick": "stagger",
    "centered": "stagger", "centred": "stagger",
}
_TRUTHY = {"1", "true", "yes", "y", "x", "skip", "t"}
_FALSY = {"", "0", "false", "no", "n", "f"}


def dose_label(passes, speed):
    """Canonical dose token, byte-identical to run_sample's ``_dose_tag`` (``f"P{p}_S{s:g}"``)."""
    return f"P{int(passes)}_S{float(speed):g}"


def parse_dose(text):
    """Parse a laser dose token in EITHER order -> ``(passes, speed, canonical_label)`` or None.

    ``'S400_P25'`` (the rows 1-2 convention on this wafer) and ``'P26_S800'`` (the row 4
    convention) both parse; both canonicalise to the P-first ``'P{passes}_S{speed:g}'`` form that
    the rest of the pipeline speaks. Anchored with ``fullmatch``: a token that merely CONTAINS a
    dose is rejected rather than guessed at, because the failure mode is silent -- run_sample's
    ``_parse_ps_label('D50_P100_S400_P25')`` happily returns ``(100, 400.0)``, reading the geometry
    pitch as the pass count."""
    s = str(text or "").strip()
    if not s:
        return None
    m = _DOSE_PS.fullmatch(s)
    if m:
        passes, speed = int(m.group(1)), float(m.group(2))
    else:
        m = _DOSE_SP.fullmatch(s)
        if not m:
            return None
        passes, speed = int(m.group(2)), float(m.group(1))
    if passes <= 0 or not math.isfinite(speed) or speed <= 0:
        return None
    return passes, speed, dose_label(passes, speed)


def parse_geometry(text):
    """Parse a pin geometry -> the canonical ``'D50 P100'`` form (µm), or None.

    Accepts ``'D50 P100'``, ``'D50_P100'``, ``'D50-P100'``. The space form is canonical because it
    is what ``porosity_table.FAMILIES`` uses, so the label can go straight onto a figure."""
    s = str(text or "").strip()
    if not s:
        return None
    m = _GEOM_RE.fullmatch(s)
    if not m:
        return None
    d, p = float(m.group(1)), float(m.group(2))
    if not (math.isfinite(d) and math.isfinite(p)) or d <= 0 or p <= 0:
        return None
    return f"D{d:g} P{p:g}"


def geometry_tokens(geometry):
    """``'D50 P100'`` -> ``(50.0, 100.0)``; ``(nan, nan)`` when unparseable."""
    m = _GEOM_RE.fullmatch(str(geometry or "").strip())
    if not m:
        return float("nan"), float("nan")
    return float(m.group(1)), float(m.group(2))


def parse_lattice(text):
    """``'hex'`` | ``'square'`` | ``'stagger'`` from a user token, or None. 'triangular' is the
    DXF's word for the same thing the wafer table calls 'Hex'."""
    return _LATTICE_ALIASES.get(str(text or "").strip().casefold()) or None


def parse_skip(text):
    """Truthy/falsy skip flag. Raises ValueError on anything ambiguous -- a misread skip either
    silently drops a good sample or analyses one the user called invalid."""
    s = str(text or "").strip().casefold()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    raise ValueError(f"skip={text!r} — expected blank/0/1/true/false/yes/no")


# ------------------------------------------------------------ VK4 filename grammar #
_RASTER_RE = re.compile(r"_Y\d+_X\d+$", re.IGNORECASE)      # mirrors run_sample._RASTER_RE
_CR_EXPLICIT = re.compile(r"^C(\d+)R(\d+)$", re.IGNORECASE)
_CR_COMPACT = re.compile(r"^(\d)(\d)$")
_D_TOKEN = re.compile(r"^D(\d+)$", re.IGNORECASE)


def parse_sample_id(stem):
    """``(col, row, snapshot_label)`` from a VK4 filename stem, or None -- never a guess.

    ``'072230_PFLMTIM_D50_11_Center'`` -> ``(1, 1, 'Center')``: the first digit of the ``11`` token
    is the column, the second is the row, and everything after it is the snapshot label.

    A candidate token must sit at index > 0 (so a leading date is never one) and must NOT be the
    last token (the last token is always the snapshot label -- ``'..._D50_11'`` is genuinely
    ambiguous, since run_sample's own ``_label_from_name`` would call that ``11`` the label).
    An explicit ``C{col}R{row}`` token wins outright. Zero candidates, or more than one, -> None.
    """
    stem = Path(str(stem)).stem if str(stem).lower().endswith(".vk4") else str(stem)
    if _RASTER_RE.search(stem):
        return None
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    explicit, compact = [], []
    for i, tok in enumerate(parts[1:-1], start=1):           # never index 0, never the last token
        m = _CR_EXPLICIT.match(tok)
        if m:
            explicit.append((i, int(m.group(1)), int(m.group(2))))
            continue
        m = _CR_COMPACT.match(tok)
        if m:
            compact.append((i, int(m.group(1)), int(m.group(2))))
    hits = explicit or compact                                # explicit form wins if present
    if len(hits) != 1:
        return None                                           # zero or ambiguous -> refuse
    i, col, row = hits[0]
    if col < 1 or row < 1:
        return None
    label = "_".join(parts[i + 1:])
    return (col, row, label) if label else None


def vk4_d_token(stem):
    """The ``D{n}`` geometry token in a VK4 stem (``'D50'``), or None when absent or contradictory.

    This is the only INDEPENDENT witness to a sample's geometry: the wafer map could be transposed
    and a right-pitch/wrong-diameter DXF still registers cleanly, so run_row cross-checks this."""
    toks = [t for t in str(stem).split("_") if _D_TOKEN.match(t)]
    uniq = {t.upper() for t in toks}
    return uniq.pop() if len(uniq) == 1 else None


def _snapshot_sort_key(label):
    order = SNAPSHOT_ORDER.index(label) if label in SNAPSHOT_ORDER else len(SNAPSHOT_ORDER)
    return (order, label.casefold())


def group_snapshots(names, row):
    """Group VK4 file NAMES into the samples of one wafer row. PURE -- never touches the disk.

    ``names`` may be strings or Paths (only ``.name``/``.stem`` is read). Returns
    ``({col: [(name, label), ...]}, other_row_names, unparsed)`` where ``unparsed`` holds
    ``(name, reason)`` pairs for every file that carries no unambiguous ``_{col}{row}_`` token.

    Within one column, colliding labels fall back to full stems (mirroring
    ``run_sample.snapshots_from_dir``) so two snapshots can never silently merge into one. Labels
    repeating ACROSS samples is normal and must NOT trigger that fallback -- every sample has a
    ``Center``."""
    by_col, other, unparsed = {}, [], []
    for n in names:
        name = Path(str(n)).name
        stem = Path(name).stem
        if _RASTER_RE.search(stem):
            unparsed.append((name, "looks like a _Y{n}_X{m} raster tile, not a wafer snapshot"))
            continue
        parsed = parse_sample_id(stem)
        if parsed is None:
            unparsed.append((name, "no unambiguous _{col}{row}_ (or C{col}R{row}) token"))
            continue
        col, r, label = parsed
        if r != int(row):
            other.append(name)
            continue
        by_col.setdefault(col, []).append((name, label, stem))
    out = {}
    for col, items in sorted(by_col.items()):
        labels = [lab for _n, lab, _s in items]
        if len(set(labels)) != len(labels):                  # collision WITHIN a sample
            items = [(n, s, s) for n, _lab, s in items]      # -> full stems, never merge
        pairs = [(n, lab) for n, lab, _s in items]
        out[col] = sorted(pairs, key=lambda t: _snapshot_sort_key(t[1]))
    return out, sorted(other), unparsed


# --------------------------------------------------------------------- the map #
@dataclass(frozen=True)
class MapEntry:
    """One (row, col) line of the wafer map, fully parsed."""

    row: int
    col: int
    line: int                                   # 1-based source line, for error messages
    skip: bool = False
    note: str = ""
    laser_raw: str = ""                         # exactly as authored ('S400_P25')
    laser: str = ""                             # canonical ('P25_S400')
    passes: int = 0
    speed: float = float("nan")
    geometry: str = ""                          # canonical ('D50 P100')
    nominal_d_um: float = float("nan")
    nominal_p_um: float = float("nan")
    lattice: str = ""                           # 'hex' | 'square'
    dxf: str = ""                               # explicit override, '' = resolve by content


def parse_wafer_map_rows(dict_rows, *, base_dir=None, dxf_dir=None):
    """Parse ``[(line_no, {header: value})]`` -> ``(entries, problems)``. PURE.

    Every problem is collected and reported together -- a first-error-only parser makes fixing a
    24-line map a 24-run chore. Entries are returned sorted by (row, col)."""
    entries, problems, seen = [], [], {}
    for line_no, raw in dict_rows:
        rec = {str(k or "").strip().casefold(): (v if v is not None else "")
               for k, v in raw.items()}
        rc = []
        for key in ("row", "col"):
            val = str(rec.get(key, "")).strip()
            try:
                i = int(val)
            except (TypeError, ValueError):
                problems.append(f"line {line_no}: {key}={val!r} is not an integer in 1..{MAX_INDEX}")
                rc.append(None)
                continue
            if not (1 <= i <= MAX_INDEX):
                problems.append(f"line {line_no}: {key}={val!r} is outside 1..{MAX_INDEX}")
                rc.append(None)
                continue
            rc.append(i)
        row, col = rc
        if row is None or col is None:
            continue
        if (row, col) in seen:
            problems.append(f"lines {seen[(row, col)]} and {line_no}: duplicate entry for "
                            f"row {row} col {col}")
            continue
        seen[(row, col)] = line_no

        note = str(rec.get("note", "")).strip()
        try:
            skip = parse_skip(rec.get("skip", ""))
        except ValueError as e:
            problems.append(f"line {line_no}: {e}")
            continue
        dxf = str(rec.get("dxf", "")).strip()
        if dxf and not Path(dxf).is_absolute():
            # A bare filename in the dxf= column almost always means "the one in the DXF folder",
            # not "next to this CSV" -- try the declared dxf_dir first, then the map's own folder,
            # and fall back to the map folder so the error names a definite path.
            for cand in [Path(d) / dxf for d in (dxf_dir, base_dir) if d]:
                if cand.is_file():
                    dxf = str(cand.resolve())
                    break
            else:
                dxf = str((Path(base_dir) / dxf).resolve()) if base_dir else dxf

        if skip:
            if not note:
                problems.append(f"WARNING: line {line_no}: skipped row {row} col {col} has "
                                f"no note — add one so the gap is self-explanatory")
            entries.append(MapEntry(row=row, col=col, line=line_no, skip=True, note=note, dxf=dxf))
            continue

        laser_raw = str(rec.get("laser", "")).strip()
        dose = parse_dose(laser_raw)
        if dose is None:
            problems.append(f"line {line_no}: laser={laser_raw!r} — expected P{{n}}_S{{v}} or "
                            f"S{{v}}_P{{n}} (e.g. P25_S400 or S400_P25)")
            continue
        passes, speed, laser = dose

        geom_raw = str(rec.get("geometry", "")).strip()
        geometry = parse_geometry(geom_raw)
        if geometry is None:
            problems.append(f"line {line_no}: geometry={geom_raw!r} — expected "
                            f"'D{{diameter}} P{{pitch}}' in µm (e.g. 'D50 P100')")
            continue
        d_um, p_um = geometry_tokens(geometry)

        lat_raw = str(rec.get("lattice", "")).strip()
        lattice = parse_lattice(lat_raw)
        if lattice is None:
            problems.append(f"line {line_no}: lattice={lat_raw!r} — expected hex, square "
                            f"or stagger")
            continue

        entries.append(MapEntry(
            row=row, col=col, line=line_no, skip=False, note=note,
            laser_raw=laser_raw, laser=laser, passes=passes, speed=speed,
            geometry=geometry, nominal_d_um=d_um, nominal_p_um=p_um,
            lattice=lattice, dxf=dxf))
    entries.sort(key=lambda e: (e.row, e.col))
    return entries, problems


def read_wafer_map(path):
    """Read a wafer map CSV -> ``(entries, meta, problems)``.

    ``#`` lines are comments; ``# key: value`` sets metadata for the three recognised keys
    (``date``, ``dxf_dir``, ``vk4_dir``). Headers are case-insensitive and order-insensitive."""
    p = Path(path)
    meta, problems = {}, []
    if not p.is_file():
        return [], meta, [f"wafer map not found: {p}"]
    # Parse with csv.reader FIRST, then drop comment records. Stripping '#' lines textually before
    # the CSV parser would corrupt any quoted field containing a newline (the 'note' column is
    # quoted): a continuation line starting with '#' would be deleted, silently swallowing whole
    # records and shifting every reported line number. csv.reader consumes such fields correctly,
    # and its line_num gives the true source line for error messages.
    body = []
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for rec in reader:
            line_no = reader.line_num
            if not rec or all(not str(c).strip() for c in rec):
                continue
            if str(rec[0]).lstrip().startswith("#"):
                m = re.match(r"#\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$", ",".join(rec).strip())
                if m and m.group(1).strip().casefold() in MAP_META_KEYS:
                    meta[m.group(1).strip().casefold()] = m.group(2).strip()
                continue
            body.append((line_no, rec))
    if not body:
        return [], meta, [f"{p.name}: no data lines (only comments/blanks)"]

    headers = [str(h or "").strip().casefold() for h in body[0][1]]
    missing = [h for h in MAP_REQUIRED_HEADERS if h not in headers]
    if missing:
        return [], meta, [f"{p.name}: missing required column(s) {missing}; found {headers}"]
    unknown = [h for h in headers if h and h not in MAP_KNOWN_HEADERS]
    if unknown:
        problems.append(f"WARNING: {p.name}: ignoring unknown column(s) {unknown}")
    dict_rows = [(line_no, dict(zip(headers, rec))) for line_no, rec in body[1:]]
    entries, more = parse_wafer_map_rows(dict_rows, base_dir=p.parent,
                                        dxf_dir=meta.get("dxf_dir"))
    return entries, meta, problems + more


def rows_present(entries):
    """Sorted distinct wafer rows in the map."""
    return sorted({e.row for e in entries})


# ------------------------------------------------------------------------ naming #
def sample_out_name(entry):
    """Results subfolder name for one sample: ``'c1 D50 P100 P25_S400'``.

    Column first so the folder listing reads in wafer order, then the geometry and the canonical
    dose -- enough to identify the sample without opening it."""
    if entry.skip or not entry.geometry:
        return safe_name(f"c{entry.col}")
    return safe_name(f"c{entry.col} {entry.geometry} {entry.laser}")


def row_out_name(date_tag, row):
    """Results container name for a whole wafer row: ``'072426 Row 1'``."""
    tag = safe_name(str(date_tag or "")).strip()
    return safe_name(f"{tag} Row {int(row)}" if tag else f"Row {int(row)}")


def date_tag_from_names(names):
    """The modal leading 6-digit token across VK4 names (``'072230'``), or None.

    Only a default: the user's results folder is named for the analysis date, which need not be the
    scan date, so the UI and ``--out-name`` can override it."""
    counts = {}
    for n in names:
        first = Path(str(n)).stem.split("_")[0]
        if re.fullmatch(r"\d{6}", first):
            counts[first] = counts.get(first, 0) + 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda k: counts[k])


# ---------------------------------------------------------------------- planning #
#: sample statuses that mean "do not run, and the row must not start"
BLOCKING_STATUSES = ("no-dxf", "dup-label", "dup-out-name", "name-mismatch")


@dataclass(frozen=True)
class PlannedSample:
    entry: MapEntry
    snapshots: tuple = ()                   # ((vk4 name, label), ...) in SNAPSHOT_ORDER
    dxf: str = ""                           # resolved path; '' when unresolved
    out_name: str = ""
    status: str = "ready"                   # ready|skipped|no-vk4|no-dxf|dup-label|
    #                                         dup-out-name|name-mismatch
    reason: str = ""


@dataclass(frozen=True)
class RowPlan:
    row: int
    date_tag: str = ""
    samples: tuple = ()
    unparsed: tuple = ()                    # ((name, reason), ...)
    other_rows: tuple = ()
    problems: tuple = ()
    warnings: tuple = ()

    @property
    def blocking(self):
        """True when the row must not start: a map problem, or any sample whose failure means the
        user's intent is unclear. A merely missing sample ('no-vk4') is not blocking -- the rest of
        the row is still worth running -- but a mis-paired or ambiguous one is."""
        return bool(self.problems) or any(s.status in BLOCKING_STATUSES for s in self.samples)

    @property
    def ready(self):
        return tuple(s for s in self.samples if s.status == "ready")


def plan_row(entries, vk4_names, dxf_for_geometry, row, *, strict_names=True, date_tag=""):
    """Compose the run plan for one wafer row. PURE -- and the main unit under test.

    ``dxf_for_geometry`` maps ``(geometry, lattice) -> path`` and is produced impurely by
    ``run_row.resolve_dxfs``; injecting it keeps this function testable with a dict literal.
    Never raises: every failure becomes a sample ``status`` or a ``problems`` string."""
    row = int(row)
    by_col, other, unparsed = group_snapshots(vk4_names, row)
    problems, warnings = [], []

    row_entries = [e for e in entries if e.row == row]
    if not row_entries:
        problems.append(f"wafer map has no entries for row {row} "
                        f"(rows present: {', '.join(str(r) for r in rows_present(entries)) or 'none'})")

    samples, used_out_names = [], {}
    for e in sorted(row_entries, key=lambda x: x.col):
        out_name = sample_out_name(e)
        snaps = tuple(by_col.get(e.col, ()))
        if e.skip:
            samples.append(PlannedSample(e, snaps, "", out_name, "skipped",
                                         e.note or "marked skip in the wafer map"))
            continue
        if not snaps:
            samples.append(PlannedSample(e, (), "", out_name, "no-vk4",
                                         f"no VK4 file carries a _{e.col}{row}_ token"))
            continue

        status, reason = "ready", ""
        # The VK4 filename's D token is an independent witness to the geometry: the column ->
        # geometry pairing reverses between wafer rows on this wafer, and a right-pitch /
        # wrong-diameter DXF registers cleanly, so a transposed map is otherwise silent.
        d_tokens = {vk4_d_token(Path(n).stem) for n, _lab in snaps}
        d_tokens.discard(None)
        want_d = f"D{e.nominal_d_um:g}".upper()
        # EVERY snapshot must agree, not merely one of them: a mixed set means a file from a
        # different geometry has been grouped into this sample by its col/row token.
        if d_tokens and not d_tokens <= {want_d}:
            msg = (f"c{e.col}: VK4 files declare {'/'.join(sorted(d_tokens))} but the wafer map "
                   f"says {e.geometry}. The geometry/column pairing REVERSES between wafer rows "
                   f"1-2 and row 4 — check the map.")
            if strict_names:
                status, reason = "name-mismatch", msg
            else:
                warnings.append(msg + "  (allowed by --allow-name-mismatch)")

        dxf = str(dxf_for_geometry.get((e.geometry, e.lattice), "") or "")
        if status == "ready" and not dxf:
            status, reason = "no-dxf", (f"no DXF matches geometry {e.geometry} on a "
                                        f"{e.lattice} lattice")

        if status == "ready":
            labels = [lab for _n, lab in snaps]
            if len(set(labels)) != len(labels):
                status, reason = "dup-label", (f"c{e.col}: snapshot labels {labels} are not unique "
                                               f"— analyze_multi_snapshot requires unique labels")

        prev = used_out_names.get(out_name.casefold())
        if prev is not None:
            status = "dup-out-name"
            reason = (f"output folder {out_name!r} collides with wafer column {prev} — "
                      f"two map lines describe the same sample")
        else:
            used_out_names[out_name.casefold()] = e.col

        samples.append(PlannedSample(e, snaps, dxf, out_name, status, reason))

    # A wafer column present in the FILES but absent from the map would otherwise be dropped in
    # total silence -- the commonest way to lose a sample is a typo'd or missing map line.
    mapped_cols = {e.col for e in row_entries}
    orphan_cols = sorted(c for c in by_col if c not in mapped_cols)
    if orphan_cols:
        problems.append(
            f"wafer row {row}: {len(orphan_cols)} column(s) {orphan_cols} have VK4 files but no "
            f"line in the wafer map — add them (or they are silently not analysed)")

    tag = str(date_tag or "") or (date_tag_from_names(vk4_names) or "")
    return RowPlan(row=row, date_tag=tag, samples=tuple(samples),
                   unparsed=tuple(unparsed), other_rows=tuple(other),
                   problems=tuple(problems), warnings=tuple(warnings))


def format_plan(plan, *, out_dir=None):
    """The preflight / ``--dry-run`` table: everything the run will do, before it does any of it."""
    lats = sorted({s.entry.lattice for s in plan.samples if s.entry.lattice})
    head = f"Row {plan.row}"
    if lats:
        head += f"  ({'/'.join(lats)})"
    if out_dir is not None:
        head += f"  ->  {out_dir}"
    lines = [head,
             f"{'col':>4}  {'laser':<10} {'geometry':<11} {'dxf':<34} "
             f"{'snapshots':<22} status"]
    for s in plan.samples:
        e = s.entry
        dxf = Path(s.dxf).name if s.dxf else "—"
        snaps = ", ".join(lab for _n, lab in s.snapshots) or "—"
        lines.append(f"{e.col:>4}  {e.laser or '—':<10} {e.geometry or '—':<11} "
                     f"{dxf[:34]:<34} {snaps[:22]:<22} {s.status}")
        if s.reason:
            lines.append(f"        {s.reason}")
    n_dxf = len({s.dxf for s in plan.ready if s.dxf})
    tail = (f" {n_dxf} distinct DXF(s) over {len(plan.ready)} sample(s);  "
            f"{len(plan.unparsed)} unparsed VK4")
    if plan.other_rows:
        tail += f";  {len(plan.other_rows)} file(s) belong to other wafer rows"
    lines.append(tail)
    for name, reason in plan.unparsed:
        lines.append(f"   unparsed: {name}  ({reason})")
    for w in plan.warnings:
        lines.append(f"   WARNING: {w}")
    for p in plan.problems:
        lines.append(f"   PROBLEM: {p}")
    return "\n".join(lines)
