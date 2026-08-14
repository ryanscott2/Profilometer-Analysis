"""
Wafer-row driver: analyze every sample of one wafer row in a single command, then roll them up.

A wafer holds a GRID of samples. Each sample is one uniform-cell dataset -- one pin geometry, one
laser dose, imaged as one or more disjoint snapshots -- i.e. exactly what ``run_sample.py
--snapshots`` analyses. This module runs a whole ROW of them:

1. read ``csv/wafer_map.csv`` (``wafer_map.py``) for each (row, col)'s dose, geometry and lattice,
2. group the VK4 files by their ``_{col}{row}_`` token into one sample per wafer column,
3. resolve which DXF each sample needs BY CONTENT (pitch + lattice read out of the drawing, not out
   of its filename), cross-checked against the diameter and the VK4 filenames,
4. print the whole plan and refuse to start if anything is ambiguous,
5. call ``run_sample.analyze_multi_snapshot`` once per column into its own subfolder,
6. write a row-level rollup: one combined CSV plus cross-sample comparison figures.

Output layout::

    results/072426 Row 1/            <- a plain CONTAINER, never a transaction target
        .pflm-row.json
        c1 D50 P100 P25_S400/        <- a normal, fully transactional per-sample dataset
        c2 D50 P100 P20_S400/
        ...
        row_measurements.csv  row_units.csv  row_summary.txt  row_manifest.json
        row_figures/

The container must never hold a ``figures/`` directory or a ``legacy/measurements.csv``:
``run_sample._looks_like_legacy_output`` ANDs exactly those two conditions, and a transaction
committed on this folder would rename the ENTIRE subtree into the system temp dir and delete it --
destroying every per-sample result inside while printing success. ``write_rollup`` asserts this,
and ``run_sample._contains_owned_datasets`` refuses the folder as an output target from the other
side. That is why the rollup files are named ``row_*`` and written with plain atomic replaces.

Usage::

    python python/run_row.py --row 1
    python python/run_row.py --row 1 --vk4 <dir> [<dir> ...] --dxf-dir <dir-or-file> [...] --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wafer_map as wm
from dxf_geometry import read_design
import run_sample as rs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repo root: modules live in python/, data sits beside it
DEF_DXF_DIR = ROOT / "dxf"
DEF_VK4_DIR = ROOT / "vk4"
DEF_CSV_DIR = ROOT / "csv"
DEF_OUT_DIR = ROOT / "results"

ROW_SENTINEL = rs.ROW_SENTINEL                 # ".pflm-row.json"
ROW_FIGURES_DIR = "row_figures"
ROW_MEAS_CSV = "row_measurements.csv"
ROW_UNITS_CSV = "row_units.csv"
ROW_SUMMARY_TXT = "row_summary.txt"
ROW_MANIFEST = "row_manifest.json"
ROW_MONTAGE_PNG = "row_montage.png"

#: fractional tolerance on |drawn Ø - nominal Ø| when verifying a resolved DXF. Generous on purpose:
#: the drawn diameter is NOT the label (the D300 square DXF draws 295 µm, D100 triangular draws 95),
#: so diameter is a sanity tolerance, never a matching key.
DIAMETER_TOL_FRAC = 0.15
PITCH_TOL_UM = 0.5
#: measured/drawn Ø band outside which a sample is flagged 'suspect' -- the only guard against a
#: same-pitch, same-lattice, wrong-diameter DXF, which registers perfectly and is otherwise silent.
SUSPECT_D_RATIO = (0.7, 1.6)
MIN_REG_SCORE = 0.5                            # mirrors the run_sample.analyze_sample warning


# ================================================================ DXF resolution == #
def lattice_kind(array, *, tol=0.03, ang_tol=0.02):
    """``'square'|'rectangular'|'triangular'|'staggered'|'oblique'`` from a primitive basis.

    The design carries no lattice field -- ``pitch_x_um``/``pitch_y_um`` are both the
    nearest-neighbour pitch for a hex array too, so they cannot tell hex from square. The primitive
    translation vectors can: 90° between equal-length vectors is a square lattice, 60° (equivalently
    120°) between equal-length vectors is triangular/hex.

    Judging by ``|cos|`` makes this invariant to the reduced-basis choice, which matters: the three
    real TRIANGULAR drawings come back with bases at 60°, 60° and 120° for the same lattice.
    Consistent with ``extract._is_oblique_lattice`` / ``register._lattice_is_oblique`` (same
    ``tol=0.03``), extended with the hex test."""
    lv = getattr(array, "lattice_vectors", None)
    if lv is None:
        px, py = float(array.pitch_x_um), float(array.pitch_y_um)
        return "square" if abs(px - py) <= tol * max(px, py, 1e-9) else "rectangular"
    a1, a2 = np.asarray(lv, dtype=float)
    n1, n2 = float(np.linalg.norm(a1)), float(np.linalg.norm(a2))
    if n1 <= 0 or n2 <= 0:
        return "oblique"
    c = abs(float(a1 @ a2) / (n1 * n2))
    equal = abs(n1 - n2) <= tol * max(n1, n2)
    if c <= ang_tol:
        return "square" if equal else "rectangular"
    if equal and abs(c - 0.5) <= ang_tol:
        return "triangular"
    # Centered-rectangular ("staggered"/brick): rows at spacing |a| offset by half a period, i.e.
    # one primitive vector is half the other plus a PERPENDICULAR component. Tested after the
    # triangular case on purpose -- hex satisfies this too, as the special case where the offset
    # makes every neighbour distance equal, and it must keep its own name.
    for u, v in ((a1, a2), (a2, a1)):
        nu = float(np.linalg.norm(u))
        if nu <= 0:
            continue
        for off in (v - 0.5 * u, v + 0.5 * u):
            if abs(float(off @ u)) <= ang_tol * nu * max(float(np.linalg.norm(off)), 1e-9):
                return "staggered"
    return "oblique"


def lattice_pitch_um(array):
    """The nearest-neighbour period of a ``PinArray``'s lattice, in µm.

    NOT ``PinArray.pitch_um``, which is the MEAN of the two primitive vector lengths. Those are
    equal for a square or triangular lattice (so the mean is the period), but not for a staggered
    one: a 150 µm staggered lattice has |a1| = 150 and |a2| = 167.7, averaging to a meaningless
    158.9 that matches no declared pitch. The shortest lattice translation is the period in every
    case."""
    lv = getattr(array, "lattice_vectors", None)
    if lv is None:
        return float(array.pitch_um)
    a1, a2 = np.asarray(lv, dtype=float)
    cands = [a1, a2, a1 + a2, a1 - a2]
    norms = [float(np.linalg.norm(v)) for v in cands]
    good = [n for n in norms if n > 1e-9]
    return min(good) if good else float(array.pitch_um)


@dataclass(frozen=True)
class DxfFact:
    """What one DXF actually contains, read out of the drawing rather than out of its name."""

    path: str
    name: str
    n_cells: int
    n_arrays: int
    is_unit_cell: bool
    marker_shape: str
    lattice: str
    pitch_um: float
    diameter_um: float
    n_pins: int
    name_d: float | None = None                # tokens parsed from the FILENAME, for cross-check
    name_p: float | None = None
    name_tri: bool = False

    @property
    def is_wafer_candidate(self):
        """A single uniform markerless array -- the shape every wafer-row sample is drawn as.

        This predicate is what makes content matching unambiguous: it excludes the legacy tiled
        multi-cell / multi-array designs that share a pitch with a wafer DXF (e.g. the markered
        ``071826_UVPFLM_D300.dxf``, also 350 µm square), leaving exactly one candidate per
        ``(pitch, lattice)``."""
        return self.n_cells == 1 and self.n_arrays == 1 and not self.is_unit_cell

    @property
    def key(self):
        return (round(self.pitch_um, 1), self.lattice)


# Anchored tokens only: a naive "D300" in name also matches D3000, and "D50" is a prefix of "D500".
_NAME_D = re.compile(r"(?<![0-9A-Za-z])D(\d+(?:\.\d+)?)(?![0-9])", re.IGNORECASE)
_NAME_P = re.compile(r"(?<![0-9A-Za-z])P(\d+(?:\.\d+)?)(?![0-9])", re.IGNORECASE)


def _name_tokens(name):
    """``(diameter, pitch, is_triangular)`` parsed from a DXF FILENAME -- cross-check only."""
    toks = re.split(r"[_\s\-]+", Path(name).stem)
    d = next((float(m.group(1)) for t in toks if (m := _NAME_D.fullmatch(t))), None)
    p = next((float(m.group(1)) for t in toks if (m := _NAME_P.fullmatch(t))), None)
    tri = any(t.upper() in ("TRIANGULAR", "TRI", "HEX", "HEXAGONAL") for t in toks)
    return d, p, tri


_DESIGN_CACHE = {}


def read_design_cached(path):
    """``read_design`` memoised by resolved path. Six files parse in ~0.7 s, and the driver needs
    each design twice (resolution, then the montage template)."""
    key = str(Path(path).resolve())
    if key not in _DESIGN_CACHE:
        _DESIGN_CACHE[key] = read_design(path)
    return _DESIGN_CACHE[key]


def _collect_dxf_files(inputs):
    """Expand DXF inputs (each a FILE or a DIRECTORY) to a deduped, sorted list of ``.dxf`` Paths.

    A directory contributes its top-level ``*.dxf`` (the original single-folder behaviour); an
    individual file is taken as given. Deduped by resolved path so the same drawing listed twice --
    or one also reachable through a listed folder -- is indexed once. Returns ``(files, problems)``;
    an empty folder is silent (as before), a missing path or a non-``.dxf`` file is a problem."""
    files, problems, seen = [], [], set()
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            cands = sorted(p.glob("*.dxf"))
        elif p.is_file():
            if p.suffix.lower() != ".dxf":
                problems.append(f"not a .dxf file: {p}")
                continue
            cands = [p]
        else:
            problems.append(f"DXF path not found: {p}")
            continue
        for f in cands:
            r = f.resolve()
            if r in seen:
                continue
            seen.add(r)
            files.append(f)
    return files, problems


def index_dxf_dir(dxf_inputs, *, read=None):
    """Parse candidate ``*.dxf`` from one or more inputs -> ``(facts, problems)``.

    ``dxf_inputs`` is a single path or a list of them; each may be a DIRECTORY (its top-level
    ``*.dxf`` are indexed, as before) or an individual ``.dxf`` FILE. Naming the exact drawings a
    wafer row uses -- rather than a whole folder -- keeps stray or unreadable drawings from other
    wafer generations out of the candidate pool, which is the surest way to avoid a same-pitch,
    same-lattice ambiguity. Unreadable files are reported, never silently skipped."""
    read = read or read_design_cached
    if isinstance(dxf_inputs, (str, Path)):
        dxf_inputs = [dxf_inputs]
    files, problems = _collect_dxf_files(dxf_inputs)
    facts = []
    for p in files:
        try:
            design = read(p)
        except Exception as e:                              # a malformed DXF must not kill the plan
            problems.append(f"could not parse {p.name}: {type(e).__name__}: {e}")
            continue
        if not design.cells or not design.cells[0].arrays:
            problems.append(f"{p.name}: no pin arrays found; ignored")
            continue
        cell = design.cells[0]
        arr = cell.arrays[0]
        nd, np_, ntri = _name_tokens(p.name)
        facts.append(DxfFact(
            path=str(p), name=p.name, n_cells=len(design.cells), n_arrays=len(cell.arrays),
            is_unit_cell=bool(design.is_unit_cell), marker_shape=cell.marker_shape or "",
            lattice=lattice_kind(arr), pitch_um=lattice_pitch_um(arr),
            diameter_um=float(arr.diameter_um), n_pins=int(cell.n_pins),
            name_d=nd, name_p=np_, name_tri=ntri))
    return facts, problems


#: wafer-map lattice word -> the word ``lattice_kind`` derives from the drawing's primitive basis
_LATTICE_DXF_WORD = {"hex": "triangular", "square": "square", "stagger": "staggered"}


def _lattice_to_dxf_word(lattice):
    """The wafer map says 'hex'; a drawing's primitive basis says 'triangular'. Same lattice."""
    return _LATTICE_DXF_WORD.get(lattice, "square")


def _verify_fact(fact, geometry, lattice):
    """Reasons ``fact`` cannot be the drawing for ``(geometry, lattice)``; empty list = accepted."""
    want_d, want_p = wm.geometry_tokens(geometry)
    bad = []
    if not fact.is_wafer_candidate:
        bad.append(f"{fact.name} is not a single markerless uniform array "
                   f"({fact.n_cells} cell(s), {fact.n_arrays} array(s), marker "
                   f"{fact.marker_shape or 'none'!r})")
    if fact.lattice != _lattice_to_dxf_word(lattice):
        bad.append(f"{fact.name} draws a {fact.lattice} lattice, the map declares {lattice}")
    if math.isfinite(want_p) and abs(fact.pitch_um - want_p) > PITCH_TOL_UM:
        bad.append(f"{fact.name} pitch {fact.pitch_um:g} µm != declared {want_p:g} µm")
    if math.isfinite(want_d) and want_d > 0:
        if abs(fact.diameter_um - want_d) / want_d > DIAMETER_TOL_FRAC:
            bad.append(f"{fact.name} draws Ø {fact.diameter_um:g} µm, the map declares "
                       f"Ø {want_d:g} µm (>{DIAMETER_TOL_FRAC:.0%} apart)")
    return bad


def resolve_dxfs(facts, needed, *, explicit=None):
    """Map each needed ``(geometry, lattice)`` to exactly one DXF -> ``(mapping, problems, warns)``.

    Content is authoritative: the key is ``(pitch, lattice)`` read out of the drawing. Filenames are
    only ever a cross-check -- ``TRIANGULAR`` is unenforced free text, so a re-export that drops it
    would otherwise hand a square drawing to a hex row. Zero candidates or more than one is a
    blocking problem naming every file considered; nothing is ever guessed."""
    explicit = explicit or {}
    candidates = [f for f in facts if f.is_wafer_candidate]
    by_key = {}
    for f in candidates:
        by_key.setdefault(f.key, []).append(f)
    mapping, problems, warns = {}, [], []

    # Filename-vs-content cross-check on the EXACT tokens only -- lattice and pitch. Deliberately
    # NOT on the diameter: a drawing is routinely undersized against its nominal label to
    # compensate for laser widening (D300_P350 draws 295 µm, D100 P150 TRIANGULAR draws 95), so a
    # diameter warning would fire on most correct files and train the user to ignore warnings.
    # Diameter is still checked against the MAP in _verify_fact, where the tolerance is meaningful.
    for f in facts:
        if f.name_tri and f.lattice != "triangular":
            warns.append(f"{f.name}: filename says TRIANGULAR but the drawing is {f.lattice}")
        if f.name_p is not None and abs(f.name_p - f.pitch_um) > PITCH_TOL_UM:
            warns.append(f"{f.name}: filename says P{f.name_p:g} but the drawing pitches at "
                         f"{f.pitch_um:g} µm")

    for geometry, lattice in sorted(set(needed)):
        override = explicit.get((geometry, lattice))
        if override:
            fact = next((f for f in facts
                         if Path(f.path).resolve() == Path(override).resolve()), None)
            if fact is None:
                try:
                    design = read_design_cached(override)
                    cell = design.cells[0]
                    arr = cell.arrays[0]
                    nd, np_, ntri = _name_tokens(Path(override).name)
                    fact = DxfFact(path=str(override), name=Path(override).name,
                                   n_cells=len(design.cells), n_arrays=len(cell.arrays),
                                   is_unit_cell=bool(design.is_unit_cell),
                                   marker_shape=cell.marker_shape or "",
                                   lattice=lattice_kind(arr), pitch_um=lattice_pitch_um(arr),
                                   diameter_um=float(arr.diameter_um), n_pins=int(cell.n_pins),
                                   name_d=nd, name_p=np_, name_tri=ntri)
                except Exception as e:
                    problems.append(f"{geometry} / {lattice}: dxf= override {override!r} could not "
                                    f"be read: {type(e).__name__}: {e}")
                    continue
            bad = _verify_fact(fact, geometry, lattice)     # an override is verified, not trusted
            if bad:
                problems.append(f"{geometry} / {lattice}: dxf= override rejected — " + "; ".join(bad))
                continue
            mapping[(geometry, lattice)] = fact.path
            continue

        _d, want_p = wm.geometry_tokens(geometry)
        # Scan by TOLERANCE, not by an exact rounded dict key: a drawing pitched at 350.3 um is
        # within PITCH_TOL_UM of a declared 350 but lands in a different bucket, so a key lookup
        # would both miss it and hide it from the >=2-candidate ambiguity guard.
        want_lat = _lattice_to_dxf_word(lattice)
        hits = [f for f in candidates
                if f.lattice == want_lat and abs(f.pitch_um - want_p) <= PITCH_TOL_UM]
        if len(hits) > 1:
            # Two drawings can share a pitch AND a lattice: on this wafer the D50 and D100 STAGGERED
            # cells were both drawn on the same 150 µm period. Diameter is normally only a
            # tolerance, but here it is the sole discriminator, so fall back to it before declaring
            # ambiguity -- and only if it leaves exactly one survivor.
            narrowed = [f for f in hits if not _verify_fact(f, geometry, lattice)]
            if len(narrowed) == 1:
                warns.append(f"{geometry} / {lattice}: {len(hits)} drawings share that pitch and "
                             f"lattice ({', '.join(f.name for f in hits)}); selected "
                             f"{Path(narrowed[0].path).name} on drawn diameter")
                hits = narrowed
        if not hits:
            seen = "; ".join(f"{f.name} (pitch {f.pitch_um:g}, {f.lattice}, Ø {f.diameter_um:g}, "
                             f"{f.n_arrays} array(s), marker {f.marker_shape or 'none'})"
                             for f in facts) or "no DXF files at all"
            problems.append(f"no DXF matches {geometry} on a {lattice} lattice. Parsed: {seen}")
            continue
        if len(hits) > 1:
            problems.append(
                f"{geometry} / {lattice} is ambiguous — {len(hits)} drawings match on pitch and "
                f"lattice: {', '.join(f.name for f in hits)}. Add a dxf= column to the wafer map "
                f"naming the right one.")
            continue
        bad = _verify_fact(hits[0], geometry, lattice)
        if bad:
            problems.append(f"{geometry} / {lattice}: " + "; ".join(bad))
            continue
        mapping[(geometry, lattice)] = hits[0].path
    return mapping, problems, warns


# ==================================================================== VK4 intake == #
def collect_vk4(vk4_dirs):
    """Gather ``*.vk4`` from one or more directories -> ``({filename: Path}, problems)``.

    Top-level files first; a directory holding none is searched recursively, so pointing at the
    PARENT of several per-geometry folders just works (this wafer's scans are filed by geometry, so
    one wafer ROW is spread across several folders). Two different files with the same name are a
    hard error -- the plan keys samples by filename."""
    out, problems, seen = {}, [], {}
    for d in vk4_dirs:
        p = Path(d)
        if not p.is_dir():
            problems.append(f"VK4 directory not found: {p}")
            continue
        # Recursive ALWAYS (rglob includes the top level). An `or` short-circuit on the top-level
        # glob would let one stray .vk4 beside the per-geometry folders hide the entire tree, and
        # the run would still exit 0 having analysed almost nothing.
        files = sorted(p.rglob("*.vk4"))
        if not files:
            problems.append(f"no .vk4 files under {p}")
        for f in files:
            r = f.resolve()
            if f.name in out and out[f.name].resolve() != r:
                problems.append(f"two different files are both named {f.name!r}: "
                                f"{seen[f.name]} and {f}")
                continue
            out[f.name] = f
            seen[f.name] = f
    return out, problems


# ====================================================================== the run == #
@dataclass
class RowRecord:
    """Outcome of one wafer column: what was planned, what happened, and the data if any."""

    planned: "wm.PlannedSample"
    status: str                               # ok|partial|suspect|failed|skipped|no-vk4|
    #                                           no-dxf|stale|cancelled|...
    reason: str = ""
    df: object = None                         # pandas DataFrame, or None
    n_registered: int = 0
    seconds: float = 0.0
    out_dir: object = None
    traceback_tail: str = ""

    @property
    def produced_data(self):
        return self.df is not None and len(self.df) > 0


def _atomic_write_text(path, text, encoding="utf-8"):
    """Overwrite one rollup file atomically (temp in the same directory, then ``os.replace``).

    The row directory is a plain CONTAINER of per-sample datasets and must NEVER host a run_sample
    output transaction: ``_commit_output_transaction`` -> ``_discard_dir`` renames the whole target
    tree into %TEMP% and deletes it, destroying every per-sample result inside. The rollup is cheap
    and fully derived, so per-file atomic replacement is the right durability level -- no staging
    directory, no directory swap."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def say(*parts):
    """``print`` that cannot abort a run. ``main()`` reconfigures stdout to UTF-8, but this module
    is also imported and driven directly (the selftest, ``--rollup-only`` from a notebook), and the
    row summary is full of µ, Ø and Φ. Losing a written result to a console encoding error would be
    absurd, so fall back to a lossy encode."""
    text = " ".join(str(p) for p in parts)
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(HERE),
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def write_row_sentinel(out_dir, plan, meta):
    """Identify the folder as a wafer-row CONTAINER before the first sample runs, so an interrupted
    row is still recognisable (and still refused as a transaction target)."""
    payload = {"format": "PFLM_ROW", "version": 1, "row": plan.row, "date": plan.date_tag,
               "created": _now(), "git_commit": _git_commit(),
               "map": str(meta.get("map", "")),
               "samples": [s.out_name for s in plan.ready]}
    _atomic_write_text(Path(out_dir) / ROW_SENTINEL, json.dumps(payload, indent=2) + "\n")


def _sample_registered_count(df):
    try:
        return int(df["snapshot_id"].nunique())
    except Exception:
        return 0


def _grade_sample(record, planned, df):
    """Post-run assertions on one sample -> ``(status, notes)``.

    ``analyze_multi_snapshot`` WARNS and skips a snapshot that fails registration and only dies if
    every one fails, so a nominally successful call can quietly cover fewer snapshots than asked."""
    notes = []
    n_want, n_got = len(planned.snapshots), _sample_registered_count(df)
    status = "ok"
    if n_got < n_want:
        status = "partial"
        notes.append(f"{n_got} of {n_want} snapshots registered")
    try:
        score = float(df["reg_score"].median())
        if math.isfinite(score) and score < MIN_REG_SCORE:
            notes.append(f"low median registration score {score:.2f}")
    except Exception:
        pass
    try:
        meas = float(df["diameter_um"].median())
        drawn = float(df["drawn_diameter_um"].median())
        if math.isfinite(meas) and math.isfinite(drawn) and drawn > 0:
            ratio = meas / drawn
            if not (SUSPECT_D_RATIO[0] <= ratio <= SUSPECT_D_RATIO[1]):
                status = "suspect"
                notes.append(f"measured/drawn Ø = {ratio:.2f} — is this the right DXF?")
    except Exception:
        pass
    return status, notes


def _manifest_created(sample_dir):
    try:
        data = json.loads((Path(sample_dir) / "figures" / "run_manifest.json")
                          .read_text(encoding="utf-8"))
        return data.get("created") or data.get("timestamp") or ""
    except Exception:
        return ""


def run_row(plan, out_dir, *, paths=None, results_root=None, jobs=None, make_qc=False,
            analyze=None, capture_panels=True, started_at=None, meta=None):
    """Run every ready sample of ``plan`` into ``out_dir``/<sample>. Returns ``(records, panels)``.

    One process, sequentially, calling ``run_sample.analyze_multi_snapshot`` per wafer column.
    Sequential is not laziness: ``_prepare_output_transaction`` unconditionally ``rmtree``s every
    ``%TEMP%\\pflm-trash-*`` on the machine, which would race a sibling process's ``_discard_dir``
    holder. Intra-sample parallelism already comes from ``jobs``.

    ``paths`` maps a VK4 filename (what the plan carries) to its real Path; the plan itself stays
    name-only and therefore pure. ``analyze`` is injectable so the selftest can drive the whole
    driver -- transactions, ordering, error containment, rollup -- without a Keyence file."""
    analyze = analyze or rs.analyze_multi_snapshot
    paths = paths or {}
    out_dir = Path(out_dir)
    started_at = started_at or _now()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_row_sentinel(out_dir, plan, meta or {})

    records, panels = [], []
    ready = plan.ready
    for i, ps in enumerate(plan.samples, start=1):
        e = ps.entry
        if ps.status != "ready":
            records.append(RowRecord(ps, ps.status, ps.reason))
            say(f"[{i}/{len(plan.samples)}] c{e.col}: {ps.status} — {ps.reason}")
            continue
        sample_dir = out_dir / ps.out_name
        say(f"\n{'='*78}\n[{i}/{len(plan.samples)}] c{e.col}  {e.geometry}  {e.laser}  "
              f"({len(ps.snapshots)} snapshot(s))  ->  {ps.out_name}\n{'='*78}")
        t0 = time.perf_counter()
        try:
            df, results, tiles = analyze(
                [(paths.get(n, Path(n)), lab) for n, lab in ps.snapshots],
                sample_dir, ps.dxf,
                passes=e.passes, speed=e.speed, cell_label=e.laser,
                make_qc=make_qc, jobs=jobs, results_root=results_root)
        except KeyboardInterrupt:
            records.append(RowRecord(ps, "cancelled", "interrupted by the user"))
            raise
        except (Exception, SystemExit) as exc:
            # SystemExit inherits BaseException, NOT Exception, and it is how run_sample reports the
            # two most likely per-sample failures ("No snapshots could be registered against the
            # DXF" and every _validate_output_target refusal). A bare `except Exception` here would
            # let one bad sample abort the entire row.
            secs = time.perf_counter() - t0
            stale = ""
            if sample_dir.is_dir():
                when = _manifest_created(sample_dir)
                if when and when < started_at:
                    stale = when
            rec = RowRecord(ps, "stale" if stale else "failed",
                            f"{type(exc).__name__}: {exc}", seconds=secs,
                            out_dir=sample_dir if sample_dir.is_dir() else None,
                            traceback_tail="".join(traceback.format_exc().splitlines(True)[-6:]))
            records.append(rec)
            say(f"FAILED  c{e.col}: {type(exc).__name__}: {exc}")
            if stale:
                say(f"  c{e.col}: a previous result from {stale} remains in that folder and was "
                      f"NOT used; nothing was deleted.")
            say("  -> continuing with the rest of the row.")
            continue

        secs = time.perf_counter() - t0
        status, notes = _grade_sample(None, ps, df)
        rec = RowRecord(ps, status, "; ".join(notes), df=df,
                        n_registered=_sample_registered_count(df), seconds=secs,
                        out_dir=sample_dir)
        records.append(rec)
        say(f"  c{e.col}: {status}"
              + (f" — {rec.reason}" if rec.reason else "")
              + f"  [{secs:.1f} s, {len(df)} measurement rows]")

        if capture_panels:
            try:
                import row_report
                template = read_design_cached(ps.dxf).cells[0]
                panel = row_report.capture_row_panel(tiles, template)
                if panel is not None:
                    panels.append((e.col, ps.out_name, e.geometry, e.laser, panel))
            except Exception as exc:                       # a montage is never worth failing a row
                say(f"  c{e.col}: montage panel unavailable ({type(exc).__name__}: {exc})")
        # The returned tiles hold live VK4 scans (height + intensity, ~10 MB each); across a
        # 12-snapshot row that is hundreds of MB of pure retention.
        del tiles, results
        gc.collect()
    if not ready:
        say("\nNo sample was ready to run.")
    return records, panels


def write_rollup(records, plan, out_dir, meta, panels, *, transparent=False):
    """Write the row-level rollup: combined CSV, per-sample units, summary, manifest and figures."""
    import row_report

    out_dir = Path(out_dir)
    assert not (out_dir / "legacy").exists() and not (out_dir / "figures").exists(), (
        "row container must never contain legacy/ or figures/ — run_sample._looks_like_legacy_output "
        "would then accept it as a transaction target and a commit would delete every sample in it")

    written = []
    rollup = row_report.build_rollup(records, row=plan.row, date_tag=plan.date_tag)
    if rollup.empty:
        text = row_report.render_row_summary(rollup, None, records, plan, empty=True)
        _atomic_write_text(out_dir / ROW_SUMMARY_TXT, text)
        say(text)
        written.append(out_dir / ROW_SUMMARY_TXT)
    else:
        units = row_report.build_units(rollup)
        _atomic_write_text(out_dir / ROW_MEAS_CSV, rollup.to_csv(index=False))
        _atomic_write_text(out_dir / ROW_UNITS_CSV, units.to_csv(index=False))
        text = row_report.render_row_summary(rollup, units, records, plan)
        _atomic_write_text(out_dir / ROW_SUMMARY_TXT, text)
        say("\n" + text)
        written += [out_dir / ROW_MEAS_CSV, out_dir / ROW_UNITS_CSV, out_dir / ROW_SUMMARY_TXT]

        fig_dir = out_dir / ROW_FIGURES_DIR
        if fig_dir.is_dir():
            # Only ever delete figures we generated: refuse to touch anything that is not a .png we
            # are about to rewrite. row_montage.png is KEPT when no panels were captured
            # (--rollup-only cannot rebuild it from the committed CSVs, so deleting it would be a
            # permanent loss of something this run did not produce).
            _keep = {ROW_MONTAGE_PNG} if not panels else set()
            for _f in sorted(fig_dir.iterdir()):
                if _f.is_file() and _f.suffix.lower() == ".png" and _f.name not in _keep:
                    _f.unlink(missing_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        written += row_report.make_row_figures(rollup, units, fig_dir, panels=panels,
                                               plan=plan, records=records,
                                               transparent=transparent)

    manifest = {
        "format": "PFLM_ROW_MANIFEST", "version": 1,
        "row": plan.row, "date": plan.date_tag, "created": _now(), "git_commit": _git_commit(),
        "map": str(meta.get("map", "")), "dxf_dir": str(meta.get("dxf_dir", "")),
        "vk4_dirs": [str(v) for v in meta.get("vk4_dirs", [])],
        "unparsed_vk4": [{"file": n, "reason": r} for n, r in plan.unparsed],
        "other_row_files": list(plan.other_rows),
        "warnings": list(plan.warnings),
        "samples": [{
            "col": r.planned.entry.col, "out_name": r.planned.out_name,
            "geometry": r.planned.entry.geometry, "lattice": r.planned.entry.lattice,
            "laser": r.planned.entry.laser, "laser_as_written": r.planned.entry.laser_raw,
            "dxf": r.planned.dxf, "status": r.status, "reason": r.reason,
            "note": r.planned.entry.note,
            "snapshots": [lab for _n, lab in r.planned.snapshots],
            "n_registered": r.n_registered, "seconds": round(r.seconds, 2),
            "rows": (0 if r.df is None else int(len(r.df))),
            "traceback": r.traceback_tail,
        } for r in records],
    }
    _atomic_write_text(out_dir / ROW_MANIFEST, json.dumps(manifest, indent=2) + "\n")
    written.append(out_dir / ROW_MANIFEST)
    return written


# ========================================================================= CLI == #
def validate_row_container(out_dir):
    """Refuse an output folder that is not safe to use as a wafer-row CONTAINER.

    Checked BEFORE the first sample runs, because by the time ``write_rollup``'s assert would fire
    the per-sample transactions have already committed inside the folder -- and if that folder was
    somebody's single-sample dataset, its own ``legacy/``+``figures/`` are still there, so it now
    satisfies ``_looks_like_legacy_output`` AND contains datasets: the next ordinary run on it
    deletes everything. Raises SystemExit with the reason."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    if rs._looks_like_legacy_output(out_dir) or (out_dir / rs.RESULTS_SENTINEL).is_file():
        raise SystemExit(
            f"refusing to use {out_dir} as a wafer-row container: it is already a SINGLE-sample "
            f"result (it has legacy/ + figures/ or a {rs.RESULTS_SENTINEL}). Writing per-sample "
            f"folders inside it would leave a directory that is both a dataset and a container, "
            f"and the next ordinary run on it would delete every result inside. "
            f"Choose a different --out / --out-name.")
    for name in ("legacy", "figures"):
        if (out_dir / name).exists():
            raise SystemExit(
                f"refusing to use {out_dir} as a wafer-row container: it contains a {name}/ "
                f"directory, which is what marks an ordinary dataset. Choose a different --out.")


def _find_map(explicit, vk4_dirs):
    if explicit:
        return Path(explicit)
    for d in vk4_dirs:
        for cand in (Path(d) / wm.DEFAULT_MAP_NAME, Path(d).parent / wm.DEFAULT_MAP_NAME):
            if cand.is_file():
                return cand
    return DEF_CSV_DIR / wm.DEFAULT_MAP_NAME


def build_parser():
    ap = argparse.ArgumentParser(
        prog="run_row.py",
        description="Analyze every sample of one wafer row, then roll them up.")
    ap.add_argument("--row", type=int, required=True,
                    help="wafer row number (the ROW of the VK4 names' _{col}{row}_ / C{col}R{row} / "
                         "R{row}C{col} token)")
    ap.add_argument("--map", default="", help=f"wafer map CSV (default: search the VK4 folder, its "
                                              f"parent, then csv/{wm.DEFAULT_MAP_NAME})")
    ap.add_argument("--vk4", nargs="+", default=None,
                    help="one or more VK4 folders; a folder with no top-level .vk4 is searched "
                         "recursively, so the PARENT of several per-geometry folders works")
    ap.add_argument("--dxf-dir", nargs="+", default=None, metavar="PATH",
                    help="candidate DXF drawings: one or more folders AND/OR individual .dxf files. "
                         "Naming the exact drawings a row needs keeps unrelated or unreadable DXFs "
                         "out of the pool (default: the map's dxf_dir, else dxf/)")
    ap.add_argument("--out", default="", help="output folder (default: results/<date> Row <n>)")
    ap.add_argument("--out-name", default="", help="name of the Results subfolder (overrides the date)")
    ap.add_argument("--only", type=int, nargs="+", default=None, help="run only these wafer columns")
    ap.add_argument("--jobs", type=int, default=None, help="per-array extraction workers")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; write nothing")
    ap.add_argument("--rollup-only", action="store_true",
                    help="rebuild the rollup from already-committed per-sample results")
    ap.add_argument("--allow-name-mismatch", action="store_true",
                    help="downgrade a VK4-filename-vs-map geometry disagreement to a warning")
    ap.add_argument("--allow-unparsed", action="store_true",
                    help="proceed even though some .vk4 carry no _{col}{row}_ token")
    ap.add_argument("--transparent", action="store_true",
                    help="transparent figure backgrounds (for a dark slide deck)")
    ap.add_argument("--make-qc", action="store_true", help="write per-array QC thumbnails")
    return ap


def main(argv=None):
    warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
    warnings.filterwarnings("ignore", message=".*divide by zero encountered.*")
    for stream in (sys.stdout, sys.stderr):                # µ and Ø through a cp1252 console pipe
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = build_parser().parse_args(argv)
    vk4_dirs = [Path(v) for v in (args.vk4 or [DEF_VK4_DIR])]
    map_path = _find_map(args.map, vk4_dirs)

    entries, meta, problems = wm.read_wafer_map(map_path)
    if args.dxf_dir:                                 # one or more folders and/or .dxf files
        dxf_inputs = [Path(x) for x in args.dxf_dir]
    elif meta.get("dxf_dir"):
        dxf_inputs = [Path(meta["dxf_dir"])]
    else:
        dxf_inputs = [DEF_DXF_DIR]

    vk4_files, vk4_problems = collect_vk4(vk4_dirs)
    facts, dxf_problems = index_dxf_dir(dxf_inputs)

    row_entries = [e for e in entries if e.row == args.row and not e.skip]
    if args.only:
        keep = set(args.only)
        row_entries = [e for e in row_entries if e.col in keep]
    needed = {(e.geometry, e.lattice) for e in row_entries if e.geometry and e.lattice}
    explicit = {(e.geometry, e.lattice): e.dxf for e in row_entries if e.dxf}
    mapping, map_problems, dxf_warns = resolve_dxfs(facts, needed, explicit=explicit)

    plan = wm.plan_row(entries, list(vk4_files), mapping, args.row,
                       strict_names=not args.allow_name_mismatch,
                       date_tag=meta.get("date", ""))
    if args.only:
        keep = set(args.only)
        plan = wm.RowPlan(
            row=plan.row, date_tag=plan.date_tag,
            samples=tuple(s for s in plan.samples if s.entry.col in keep),
            unparsed=plan.unparsed, other_rows=plan.other_rows,
            problems=plan.problems, warnings=plan.warnings)

    all_problems = tuple(problems) + tuple(vk4_problems) + tuple(dxf_problems) + tuple(map_problems)
    # Classify on a LEADING marker, never a substring: problem text quotes the user's own cells, so
    # a cell reading "WARNING" would otherwise downgrade a blocking error to a warning and let a
    # broken map run. Every real problem starts with "line N:" or "<file>:".
    blocking = tuple(p for p in all_problems if not p.startswith("WARNING"))
    soft = tuple(p for p in all_problems if p.startswith("WARNING"))
    if plan.unparsed and not args.allow_unparsed:
        blocking += (f"{len(plan.unparsed)} .vk4 file(s) carry no usable _{{col}}{{row}}_ token; "
                     f"rename them or pass --allow-unparsed",)
    plan = wm.RowPlan(row=plan.row, date_tag=plan.date_tag, samples=plan.samples,
                      unparsed=plan.unparsed, other_rows=plan.other_rows,
                      problems=plan.problems + blocking,
                      warnings=plan.warnings + soft + tuple(dxf_warns))

    out_dir = (Path(args.out) if args.out
               else DEF_OUT_DIR / (wm.safe_name(args.out_name) if args.out_name
                                   else wm.row_out_name(plan.date_tag, plan.row)))
    dxf_show = ", ".join(str(x) for x in dxf_inputs)
    say(f"wafer map : {map_path}")
    say(f"DXF       : {dxf_show}   ({len(facts)} drawing(s) parsed)")
    say(f"VK4       : {', '.join(str(v) for v in vk4_dirs)}   ({len(vk4_files)} file(s))\n")
    say(wm.format_plan(plan, out_dir=out_dir))

    if plan.blocking:
        say("\nRefusing to start: fix the problems above (nothing was written).")
        return 3
    if args.dry_run:
        say("\n--dry-run: nothing was written.")
        return 0

    validate_row_container(out_dir)          # BEFORE anything is written (raises SystemExit)
    # The row folder's PARENT is the results root every per-sample transaction is validated against,
    # so an --out outside results/ works instead of failing every sample.
    results_root = out_dir.parent

    run_meta = {"map": str(map_path), "dxf_dir": dxf_show, "vk4_dirs": vk4_dirs}
    if args.rollup_only:
        records = _records_from_disk(plan, out_dir)
        panels = []
    else:
        records, panels = run_row(plan, out_dir, paths=vk4_files, jobs=args.jobs,
                                  make_qc=args.make_qc, meta=run_meta,
                                  results_root=results_root)
    if args.only:
        # A rollup built from a subset would REPLACE the whole row's rollup with a partial one, and
        # a reader cannot tell the difference. The per-sample results are committed and complete;
        # re-run without --only to rebuild the rollup.
        say(f"\n--only was given, so the row-level rollup was NOT rewritten "
            f"(it would replace the whole row's CSVs and figures with just columns "
            f"{sorted(set(args.only))}). Re-run with --rollup-only to rebuild it from all samples.")
    else:
        write_rollup(records, plan, out_dir, run_meta, panels, transparent=args.transparent)

    n_ready = len(plan.ready)
    n_ok = sum(1 for r in records if r.produced_data)
    say(f"\nRow {plan.row}: {n_ok}/{n_ready} sample(s) produced data  ->  {out_dir}")
    if n_ok == 0:
        return 2
    return 0 if n_ok == n_ready else 1


def _records_from_disk(plan, out_dir):
    """``--rollup-only``: rebuild records from committed per-sample ``legacy/measurements.csv``."""
    import pandas as pd
    records = []
    for ps in plan.samples:
        if ps.status != "ready":
            records.append(RowRecord(ps, ps.status, ps.reason))
            continue
        csv = Path(out_dir) / ps.out_name / "legacy" / "measurements.csv"
        if not csv.is_file():
            records.append(RowRecord(ps, "failed", f"no committed result at {csv}"))
            continue
        df = pd.read_csv(csv)
        status, notes = _grade_sample(None, ps, df)
        records.append(RowRecord(ps, status, "; ".join(notes), df=df,
                                 n_registered=_sample_registered_count(df),
                                 out_dir=Path(out_dir) / ps.out_name))
    return records


if __name__ == "__main__":
    raise SystemExit(main())
