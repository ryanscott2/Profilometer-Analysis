"""Shared launch helpers for the PFLM desktop UI.

Constants and pure helpers the desktop front end (``pflm_app.py``) needs to launch a run the same
way every time: where the results tree and the sample library live, how a VK4 folder is classified
(tiled raster / wafer row / snapshot montage), the depth-calibration output contract, and the
folder-name guards that stop two samples from overwriting each other.

These lived in ``pflm_ui.py`` (the retired Tkinter front end) and were split out so the maintained
PySide6/QML app no longer depends on it. Kept stdlib-only -- its one import, ``wafer_map``, is
itself stdlib-only -- so pulling these in never drags numpy/pandas/matplotlib into UI startup.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# The wafer-row vocabulary (filename grammar, wafer map, run plan) lives in wafer_map.py, which is
# deliberately stdlib-only so importing it here keeps the UI's startup free of numpy/pandas/
# matplotlib. safe_name is shared from there so the UI and run_row.py cannot drift apart on what a
# results folder is called.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wafer_map import parse_sample_id, safe_name as _safe_name  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repo root: modules live in python/, data sits beside it
SAMPLES_JSON = ROOT / ".ui_samples.json"
WORKSPACE = ROOT / ".ui_workspace"
DEF_OUT = ROOT / "results"

# Depth-calibration tool (calibrate_depth.py). Keep these in sync with that module's OUT_NAME /
# MEAS_REL: it pools the per-sample legacy CSVs and writes the cross-sample analysis here.
CAL_SCRIPT = "calibrate_depth.py"
ROW_SCRIPT = "run_row.py"        # wafer-row batch driver (run_row.py --row N)
ROW_FIGURES_DIR = "row_figures"  # keep in sync with run_row.ROW_FIGURES_DIR
CAL_OUT_NAME = "etch depth"
MEAS_REL = Path("legacy") / "measurements.csv"

# Prefilled band definitions (one band per line: min_Ø, max_Ø, pitch in µm) — tight tolerance bands
# around the three pin families on the test wafers: Ø ~50 µm @100 µm pitch, Ø ~100 µm @150 µm pitch
# and Ø ~300 µm @350 µm pitch. Editable; blank (or comments only) tells calibrate_depth.py to fall
# back to the measurements' own 'band' column.
DEFAULT_BAND_DEFS = ("47.5, 52.5, 100\n"
                     "95, 105, 150\n"
                     "290, 310, 350\n")


def _sample_name_collision(name, existing_names):
    """Return the existing sample whose Windows output folder collides with ``name``."""
    target = _safe_name(name).casefold()
    return next((other for other in existing_names
                 if other != name and _safe_name(other).casefold() == target), None)


# Multi-snapshot mode: a dataset folder can hold either a ``_Y{n}_X{m}`` tile raster (the tiled
# full-sample workflow) OR a set of DISJOINT snapshots -- independent crops of ONE uniform cell
# (e.g. a "Center" and a "TopLeft"). These mirror ``run_sample._RASTER_RE`` / ``snapshots_from_dir``
# so the UI can show which mode a folder will run in and route the run command accordingly.
_RASTER_TILE_RE = re.compile(r"_Y\d+_X\d+$", re.IGNORECASE)


def _snapshot_label(stem):
    """Snapshot label from a filename stem: the trailing ``_``-token (e.g. 'Center', 'TopLeft')."""
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def _wafer_cell_id(stem):
    """``(col, row)`` from a VK4 stem's ``_{col}{row}_`` token, or None. Thin wrapper over
    ``wafer_map.parse_sample_id`` so the UI and ``run_row.py`` agree on the grammar exactly."""
    got = parse_sample_id(stem)
    return None if got is None else (got[0], got[1])


def _classify_vk4_folder(vk4_dir):
    """``(mode, labels)`` for a VK4 folder.

    ``'raster'``   -- holds a ``_Y{n}_X{m}`` tile grid (the assembled full-sample workflow).
    ``'row'``      -- a FLAT MULTI-SAMPLE wafer folder: the plain ``*.vk4`` carry ``_{col}{row}_``
                      tokens for MORE THAN ONE distinct (col, row), i.e. several different samples
                      live here (-> run_row.py). Also detected one level down, so pointing at the
                      PARENT of several per-geometry folders works.
    ``'snapshots'``-- disjoint crops of ONE uniform cell (-> run_sample --snapshots).

    The ``row`` test deliberately runs BEFORE the label-collision fallback below: on a flat wafer
    folder every sample has a ``Center``, so that fallback would otherwise fire and route every
    sample of every row into a single ``analyze_multi_snapshot`` call, which averages them as if
    they were replicate views of one cell -- silently, with no error.

    ``len(cells) > 1``, not ``>= 1``: a single-sample folder that merely carries a CR token is
    today's ``snapshots`` mode and must keep behaving identically."""
    vks = sorted(Path(vk4_dir).glob("*.vk4"))
    if vks and not any(_RASTER_TILE_RE.search(p.stem) for p in vks):
        cells = {c for c in (_wafer_cell_id(p.stem) for p in vks) if c is not None}
        if len(cells) > 1:
            return "row", [f"c{c}r{r}" for c, r in sorted(cells)]
        labels = [_snapshot_label(p.stem) for p in vks]
        if len(set(labels)) != len(labels):                 # collision -> disambiguate, never merge
            labels = [p.stem for p in vks]
        return "snapshots", labels
    if not vks:                                  # maybe a PARENT of per-geometry sample folders
        deep = sorted(Path(vk4_dir).rglob("*.vk4"))
        cells = {c for c in (_wafer_cell_id(p.stem) for p in deep) if c is not None}
        if len(cells) > 1:
            return "row", [f"c{c}r{r}" for c, r in sorted(cells)]
    return "raster", []


def _first_ps_label(text):
    """First ``P{passes}_S{speed}`` dose token in the params text (the shared dose for a
    multi-snapshot dataset), or ``''`` if none."""
    m = re.search(r"P\d+_S\d+(?:\.\d+)?", text or "")
    return m.group(0) if m else ""
