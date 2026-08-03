"""
Full-sample driver: assemble a tiled VK4 raster, register every unit cell, measure all pins.

For a sample scanned as a continuous ``_Y{n}_X{m}`` tile grid that spans MANY unit cells:

1. stitch the tiles into one scan (``assemble.py``),
2. locate every unit cell by its 200 um alignment marker and map the DXF onto each -- the scan
   is X-mirrored vs the DXF, handled per cell (the array is never flipped; ``register.py``),
3. measure every array of every cell (``extract.py``), tagged with that cell's laser
   parameters from ``CSV/cell_params.csv`` keyed by design (row, col) with (1,1) = top-left,
4. write ``Results/measurements.csv`` + the v1 plot set + a labeled cell overview and a
   full-sample height map, both rendered in DESIGN orientation (un-mirrored) for presentation.

Usage:
    python run_sample.py                      # DXF/ VK4/ CSV/ Results/ under this folder
    python run_sample.py <vk4_dir> <out_dir> [<dxf>] [<cell_csv>]
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs      # house type sizes -- one knob for every figure in the project
from dxf_geometry import read_design, validate_equivalent_cells
from assemble import assemble_tiles
from register import RegistrationAmbiguityError, register_sample
from extract import ArraySample, extract_array
from laser_params import load_cell_params, CELL_CSV_NAME
from vk4 import read_vk4     # per-snapshot ingest for the disjoint multi-snapshot mode
import report as ra          # shared measurement-row builder + legacy plot suite
import parallel              # optional CPU process fan-out for the per-array extraction loop


def _extract_worker(scan, item):
    """Top-level (picklable) unit of work for :func:`parallel.pmap_shared`: measure one array.

    Pure w.r.t. the shared read-only ``scan`` and its own picklable ``item``; returns one
    PinFinResult.  Kept module-level so process workers can import it under Windows 'spawn'.
    """
    pl, a, sample, qc_path, make_qc = item
    return extract_array(scan, pl, a, sample, make_qc=make_qc, qc_path=qc_path)

HERE = Path(__file__).parent
DEF_DXF_DIR = HERE / "DXF"
DEF_VK4_DIR = HERE / "VK4"
DEF_CSV_DIR = HERE / "CSV"
DEF_OUT_DIR = HERE / "Results"
RESULTS_SENTINEL = ".pflm-results.json"
# Marks a wafer-row CONTAINER (written by run_row.py): a plain folder holding several per-sample
# datasets plus a rollup. It is never itself a transaction target -- see _contains_owned_datasets.
ROW_SENTINEL = ".pflm-row.json"

# Radial-average overlay figures are configured by CSV/radial_sets.csv: each row is one
# comparison set -- a list of P{passes}_S{speed} labels whose mean-pin radial profiles are
# overlaid (one figure per nominal geometry). If that file is absent or empty, every laser
# parameter present in the sample is overlaid instead (a single "all" set per geometry).
RADIAL_CSV_NAME = "radial_sets.csv"


def save_source_docs(dxf_path, vk4_dir, figures_dir):
    """Copy the design DXF and archive the source VK4 tiles into figures/ for provenance.

    The VK4 archive is rebuilt only when missing or older than the newest VK4 file, so repeated
    runs on the same scan don't re-zip a large dataset."""
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)
    dxf_path = Path(dxf_path)
    if dxf_path.exists():
        shutil.copy2(dxf_path, figures_dir / dxf_path.name)
        print(f"Copied DXF -> {figures_dir / dxf_path.name}")
    vk4_dir = Path(vk4_dir)
    vk4s = (sorted(p for p in vk4_dir.iterdir() if p.suffix.lower() == ".vk4")
            if vk4_dir.is_dir() else [])
    if not vk4s:
        print(f"No .vk4 files in {vk4_dir} -> no source archive.")
        return
    zpath = figures_dir / "vk4_source.zip"
    newest = max(p.stat().st_mtime for p in vk4s)
    if zpath.exists() and zpath.stat().st_mtime >= newest:
        print(f"VK4 source archive already current -> {zpath}")
        return
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in vk4s:
            zf.write(p, arcname=p.name)
    print(f"Zipped {len(vk4s)} VK4 files -> {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")


def save_provenance(figures_dir, vk4_dir, dxf_path, cell_csv):
    """Make a Results/ folder self-describing: copy the run's laser-parameter grid
    (``cell_params.csv``) and its sibling ``radial_sets.csv`` into figures/, and write a
    ``run_manifest.json`` recording which code + inputs produced this run (git commit, input paths,
    the VK4 file list). Cheap, and re-created every run alongside the DXF/VK4 provenance."""
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)
    cell_csv = Path(cell_csv)
    for src in (cell_csv, cell_csv.parent / RADIAL_CSV_NAME):
        if src.exists():
            shutil.copy2(src, figures_dir / src.name)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(HERE),
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = "unknown"
    vk4_dir = Path(vk4_dir)
    vk4s = sorted(p.name for p in vk4_dir.glob("*.vk4")) if vk4_dir.is_dir() else []
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "dxf": str(dxf_path),
        "cell_csv": str(cell_csv),
        "vk4_dir": str(vk4_dir),
        "n_vk4": len(vk4s),
        "vk4_files": vk4s,
    }
    (figures_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote provenance (cell_params/radial_sets + run_manifest.json) -> {figures_dir}")


def _sentinel_valid(path, final_dir=None, states=None):
    try:
        data = json.loads((Path(path) / RESULTS_SENTINEL).read_text(encoding="utf-8"))
        if data.get("format") != "PFLM_RESULTS" or data.get("version") != 1:
            return False
        if final_dir is not None:
            recorded = Path(data["final_dir"]).resolve()
            if recorded != Path(final_dir).resolve():
                return False
        if states is not None and data.get("state") not in set(states):
            return False
        return True
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        return False


def _looks_like_legacy_output(path):
    """Recognise pre-sentinel PFLM results so existing datasets can be migrated safely once."""
    path = Path(path)
    return ((path / "legacy" / "measurements.csv").is_file()
            and (path / "figures").is_dir())


def _contains_owned_datasets(path):
    """True if ``path`` is a CONTAINER of other PFLM datasets (a wafer-row folder), not a dataset.

    A transaction REPLACES its target: :func:`_commit_output_transaction` renames the entire
    existing tree into the system temp dir and deletes it (:func:`_discard_dir`), because
    ``os.replace`` cannot rename onto a non-empty directory. So a directory that CONTAINS other
    results must never be a transaction target -- committing there would destroy every dataset
    inside it while printing success. One-level scan; dot-directories are our own staging orphans
    and are skipped."""
    path = Path(path)
    if (path / ROW_SENTINEL).is_file():
        return True
    try:
        kids = [c for c in path.iterdir() if c.is_dir() and not c.name.startswith(".")]
    except OSError:
        return False
    return any(_sentinel_valid(c) or _looks_like_legacy_output(c) for c in kids)


def _validate_output_target(out_dir, protect=(), results_root=DEF_OUT_DIR):
    """Validate a dedicated owned dataset directory and return its resolved path.

    Output is restricted to a strict descendant of ``results_root``. Existing non-empty folders
    must carry our sentinel or match the pre-sentinel PFLM output structure; arbitrary directories
    are never recursively cleared or replaced.
    """
    resolved = Path(out_dir).resolve()
    root = Path(results_root).resolve()
    if resolved == root or root not in resolved.parents:
        raise SystemExit(
            f"refusing unsafe output dir {resolved}: choose a dedicated dataset subfolder under "
            f"{root} (the Results root itself and unrelated directories are never replaced)")
    for src in protect:
        if not src:
            continue
        sp = Path(src).resolve()
        if resolved == sp or resolved in sp.parents or sp in resolved.parents:
            raise SystemExit(
                f"refusing output dir {resolved}: it overlaps input path {sp}; keep Results and "
                "raw VK4/DXF/CSV inputs in separate directories")
    if resolved.exists() and _contains_owned_datasets(resolved):
        raise SystemExit(
            f"refusing to use {resolved} as an output dataset: it CONTAINS other PFLM results "
            f"(it is a wafer-row container). A run replaces its whole output directory, which "
            f"would destroy them. Target one of its subfolders instead.")
    if resolved.exists() and any(resolved.iterdir()):
        if not (_sentinel_valid(resolved, resolved, ("complete",))
                or _looks_like_legacy_output(resolved)):
            raise SystemExit(
                f"refusing to replace unowned non-empty directory {resolved}: missing "
                f"{RESULTS_SENTINEL} and not recognisable as an earlier PFLM result")
    return resolved


def _write_results_sentinel(path, final_dir, state):
    payload = {"format": "PFLM_RESULTS", "version": 1, "state": state,
               "final_dir": str(Path(final_dir).resolve())}
    (Path(path) / RESULTS_SENTINEL).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resilient_rmtree(path, *, attempts=30, delay=0.5):
    """``shutil.rmtree`` with retries (~15 s). On Windows a cloud-sync client (OneDrive), the Search
    indexer, or an antivirus scanner can briefly hold a handle on a just-renamed directory, so
    ``rmtree`` transiently raises ``PermissionError``/``OSError`` even though the caller owns the
    tree. Retry until the handle is released, then give up. Returns True on success (or if already
    gone), False if it never succeeded -- the caller decides whether that is fatal."""
    import time
    path = Path(path)
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except (PermissionError, OSError):
            if i == attempts - 1:
                return False
            time.sleep(delay)
    return False


def _discard_dir(path):
    """Remove a directory even while a cloud-sync client (OneDrive) holds handles on files inside it.
    A directory RENAME succeeds where an in-place ``rmtree`` is blocked, so move the tree into the
    system temp dir (not OneDrive/Results) and delete it there, where nothing locks it. Falls back to
    an in-place resilient rmtree only if the temp dir is on a different volume (rename impossible).
    Returns True on success."""
    path = Path(path)
    if not path.exists():
        return True
    try:
        holder = Path(tempfile.mkdtemp(prefix="pflm-trash-"))
        os.replace(path, holder / path.name)             # dir rename works with locked contents
    except OSError:
        return _resilient_rmtree(path)
    _resilient_rmtree(holder)                            # lock-free temp -> succeeds; best-effort
    return True


def _prepare_output_transaction(out_dir, protect=(), results_root=DEF_OUT_DIR):
    """Create an owned hidden staging sibling while leaving the last good result untouched.

    No ``.previous`` backup is kept -- at commit the completed staging dir is renamed straight onto
    the final path (after the old result is removed), so no backup pile can accumulate. A stale
    staging dir left by an interrupted run is inert and GC'd here."""
    final_dir = _validate_output_target(out_dir, protect=protect, results_root=results_root)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    for _held in Path(tempfile.gettempdir()).glob("pflm-trash-*"):
        shutil.rmtree(_held, ignore_errors=True)         # opportunistic GC of settled discard-holders
    # Recover / GC any staging orphan from an interrupted run. A 'complete' orphan is a result that
    # was built but not swapped in: promote it if the final is missing, otherwise it is superseded.
    for orphan in sorted(final_dir.parent.glob(f".{final_dir.name}.staging-*")):
        if not orphan.is_dir():
            continue
        if _sentinel_valid(orphan, final_dir, ("complete",)) and not final_dir.exists():
            os.replace(orphan, final_dir)
            print(f"Recovered a completed-but-unswapped result -> {final_dir}")
        elif _sentinel_valid(orphan, final_dir, ("staging", "complete")):
            _discard_dir(orphan)                         # incomplete or superseded -> inert junk
    stage = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.staging-", dir=final_dir.parent))
    _write_results_sentinel(stage, final_dir, "staging")
    return stage, final_dir


def _commit_output_transaction(stage, final_dir):
    """Validate the staged run and rename staging onto the final path.

    No ``.previous`` backup pile is kept. ``os.replace`` cannot rename onto a non-empty directory on
    Windows, so the old result is first moved aside by a fast rename into the SYSTEM temp dir (not
    OneDrive, not Documents, not Results) -- a rename succeeds even while a cloud-sync client holds
    handles, and deleting it there is never blocked by a sync lock, so nothing accumulates where the
    user looks. On any failure the fully-built staging dir remains (``.<name>.staging-*``) so a
    completed prior result is never left half-swapped."""
    stage, final_dir = Path(stage).resolve(), Path(final_dir).resolve()
    if (stage.parent != final_dir.parent
            or not stage.name.startswith(f".{final_dir.name}.staging-")
            or not _sentinel_valid(stage, final_dir, ("staging",))):
        raise RuntimeError(f"refusing to commit unowned or mismatched staging directory {stage}")
    required = (stage / "legacy" / "measurements.csv", stage / "figures" / "run_manifest.json")
    missing = [str(p.relative_to(stage)) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"staged analysis is incomplete; missing required outputs: {missing}")
    _write_results_sentinel(stage, final_dir, "complete")
    if final_dir.exists() and not _discard_dir(final_dir):   # move old result out + delete in temp
        raise RuntimeError(
            f"could not clear the previous result {final_dir} (a cloud-sync client or another "
            f"process is holding it); close anything using it and re-run. The new result is fully "
            f"staged at {stage}.")
    os.replace(stage, final_dir)                             # atomic same-dir rename of new result
    print(f"Committed complete analysis transaction -> {final_dir}")


def _band_targets(template):
    out = {}
    for b in sorted(set(a.band for a in template.arrays)):
        out[b] = float(np.median([a.diameter_um for a in template.arrays if a.band == b]))
    return out


def analyze_sample(vk4_dir, out_dir, dxf_path, cell_csv, *, make_qc=False, jobs=None):
    # Build the new run beside the last good result. Nothing below writes to the final
    # destination until every required artifact has been produced successfully.
    stage_dir, final_out_dir = _prepare_output_transaction(
        out_dir,
        protect=(vk4_dir, dxf_path, cell_csv),
        results_root=DEF_OUT_DIR,
    )
    out_dir = stage_dir
    design = read_design(dxf_path)
    validate_equivalent_cells(design)
    template = design.cells[0]
    band_target = _band_targets(template)
    print(design.summary())

    scan = assemble_tiles(vk4_dir)
    try:
        placements = register_sample(scan, template)
    except RegistrationAmbiguityError as e:
        raise SystemExit(str(e)) from None
    if not placements:
        raise SystemExit("No unit cells could be registered in the assembled sample.")
    params = load_cell_params(cell_csv)
    nrow = max(p.cell_row for p in placements)
    ncol = max(p.cell_col for p in placements)
    print(f"\nRegistered {len(placements)} unit cells in a {nrow}x{ncol} grid.")
    if params:
        print(f"Loaded laser params for {len(params)} grid cells from {cell_csv}.")
    else:
        print(f"No cell params at {cell_csv} (geometry still measured; fill the P_S grid).")
    # cell_params.csv is read in DESIGN/DXF orientation: line r -> design row r (1 = top), column
    # c -> design col c (1 = left). The Keyence scan is X-mirrored vs the design, so design col 1
    # sits at the RIGHT edge of the raw scan (largest marker x). This table lets you confirm each
    # CSV grid cell maps to the physical cell you intend; a rot near +-180 would flag a wafer
    # placed/scanned re-oriented (the grid frame would then need flipping), vs rot~0 = as-designed.
    print("\nCSV grid (design orientation) -> physical cell — verify this maps as you intend:")
    print(f"  {'design(r,c)':>11} {'CSV label':>10} {'mark x_mm':>10} {'y_mm':>7} {'rot deg':>8} {'reg':>5}")
    for p in sorted(placements, key=lambda q: (q.cell_row, q.cell_col)):
        pr = params.get((p.cell_row, p.cell_col))
        note = "  <-- low reg" if p.score < 0.5 else ""    # (no cell ever has method 'grid-infill')
        if not p.absolute_origin:
            note += f"  <-- PHASE ONLY; absolute {p.ambiguous_axes.upper()} index unresolved"
        print(f"  {f'({p.cell_row},{p.cell_col})':>11} {(pr.label if pr else '-'):>10} "
              f"{p.origin_col * scan.x_um_per_px / 1000:>10.2f} "
              f"{p.origin_row * scan.y_um_per_px / 1000:>7.2f} "
              f"{p.rotation_deg:>+8.2f} {p.score:>5.2f}{note}")

    out_dir = Path(out_dir)
    qc_dir = out_dir / "legacy" / "qc"       # created on demand by extract only when make_qc=True
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "legacy").mkdir(parents=True, exist_ok=True)
    save_source_docs(dxf_path, vk4_dir, out_dir / "figures")   # DXF copy + VK4 archive (provenance)
    save_provenance(out_dir / "figures", vk4_dir, dxf_path, cell_csv)   # cell_params/radial + manifest

    rows, results = [], []
    res_by_cell = {pl.cell_id: {} for pl in placements}
    # Build the flat list of independent (cell x array) extraction jobs, then run them -- serially by
    # default (byte-identical reference) or across processes when jobs>1 / PFLM_JOBS is set. Every
    # extract_array call is pure w.r.t. the shared read-only scan, so parallelism only reorders
    # execution; results come back in submission order, so the CSV / bookkeeping is unchanged.
    work, book = [], []                          # work: extract inputs; book: bookkeeping (in order)
    for pl in placements:
        pr = params.get((pl.cell_row, pl.cell_col))
        passes = pr.passes if pr else 0
        speed = pr.speed if pr else float("nan")
        label = pr.label if pr else ""
        reliable_cell = pl.score >= 0.5
        for a in template.arrays:
            sample = ArraySample(
                filename=f"r{pl.cell_row}c{pl.cell_col}_b{a.band}c{a.col}_D{a.diameter_um:g}",
                vk4_stem=f"r{pl.cell_row}c{pl.cell_col}", cell_id=pl.cell_id,
                array_id=a.array_id, band=a.band, col=a.col, passes=passes, speed=speed,
                nominal_diameter_um=a.diameter_um, target_diameter_um=band_target[a.band],
                nominal_pitch_um=a.pitch_um, nominal_pitch_x_um=a.pitch_x_um,
                nominal_pitch_y_um=a.pitch_y_um, nx=a.nx, ny=a.ny,
                cx_um=a.cx_um, cy_um=a.cy_um, cell_label=label)
            work.append((pl, a, sample, qc_dir / (sample.filename + ".png"), make_qc))
            book.append((pl, a, sample, passes, label, reliable_cell))

    extracted = parallel.pmap_shared(_extract_worker, work, scan, jobs=jobs)

    for (pl, a, sample, passes, label, reliable_cell), res in zip(book, extracted):
        reliable = (reliable_cell and passes > 0
                    and not any(k in res.flags for k in ra.CRITICAL_FLAGS))
        row = ra.result_to_row(res, reliable)
        row["cell_row"], row["cell_col"] = pl.cell_row, pl.cell_col
        row["reg_score"], row["cell_label"] = pl.score, label
        rows.append(row)
        results.append((sample, res, reliable))
        res_by_cell[pl.cell_id][a.array_id] = res

    df = pd.DataFrame(rows).sort_values(["cell_row", "cell_col", "band", "col"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meas_csv = out_dir / "legacy" / "measurements.csv"
    df.to_csv(meas_csv, index=False)
    print(f"\nWrote {meas_csv} ({len(df)} rows)")

    missing = [(p.cell_row, p.cell_col) for p in placements
               if (p.cell_row, p.cell_col) not in params]
    if missing:                                          # only warn; the CSV is user-authored
        print(f"WARNING: {len(missing)} registered cell(s) have no entry in {cell_csv} "
              f"(row,col): {sorted(missing)}. Geometry is still measured; add their "
              f"P{{passes}}_S{{speed}} to the grid at that row/column to tag them.")
    # Reverse check: a cell_params entry with NO registered cell means a design row/column dropped
    # out of registration. Interior gaps are preserved by the absolute (pitch-gap) cell indexing in
    # register._assign_grid_indices, but a missing EDGE row/column cannot be recovered from geometry
    # alone and would shift the surviving cells' (passes,speed) mapping -- so flag it loudly.
    reg_rc = {(p.cell_row, p.cell_col) for p in placements}
    unregistered = [rc for rc in params if rc not in reg_rc]
    if unregistered:
        print(f"WARNING: {len(unregistered)} cell_params entry(ies) have NO registered cell "
              f"(row,col): {sorted(unregistered)}. If this is a dropped edge row/column the "
              f"surviving cell->parameter (passes/speed) mapping may be SHIFTED -- verify the "
              f"design(r,c) table above before trusting laser-parameter assignments.")

    # per-unit-cell report figures
    cells_dir = out_dir / "figures" / "cells"; cells_dir.mkdir(parents=True, exist_ok=True)
    for pl in placements:
        # x = design column (left->right), y = design row (top->bottom)
        render_cell_report(scan, pl, template, res_by_cell[pl.cell_id],
                           params.get((pl.cell_row, pl.cell_col)),
                           cells_dir / f"cell_x{pl.cell_col}_y{pl.cell_row}.png")
    print(f"Wrote {len(placements)} per-cell reports -> {cells_dir}")

    # Clean outputs live in figures/ (intensity_map, sample_heightmap, param_depth_scatter,
    # radial_overlays/, cells/, diameter_model). Legacy outputs (v1 figure set,
    # measurements.csv, qc/) are regenerated under legacy/ ahead of the planned refactor.
    save_sample_overview(scan, template, placements, out_dir / "figures" / "intensity_map.png")
    save_sample_heightmap(scan, out_dir / "figures" / "sample_heightmap.png")
    # 3D centre-5x5 height map per array; repeating cells get per-cell subfolders (all cells).
    _multi = len({(p.cell_row, p.cell_col) for p in placements}) > 1
    def _plabel3d(p):
        pr = params.get((p.cell_row, p.cell_col))
        return f"Passes: {pr.passes}\nSpeed: {pr.speed:g} mm/s" if (pr and pr.valid) else ""
    write_3d_pin_maps(out_dir / "figures",
                      [((f"cell_x{p.cell_col}_y{p.cell_row}" if _multi else None), scan, p, _plabel3d(p))
                       for p in placements], template)
    make_param_summary(df, out_dir)          # per-cell median depth & Ø-oversizing vs params
    make_param_depth_scatter(df, out_dir)    # same, with every array scattered around the median
    make_radial_overlays(template, placements, params, res_by_cell, out_dir,
                         sets_csv=Path(cell_csv).parent / RADIAL_CSV_NAME)
    ra.make_diameter_model(df, out_dir / "figures")     # process-conditional model (drawn,P,S) + R²/CI
    ra.make_plots(df, results, out_dir / "legacy")
    _commit_output_transaction(out_dir, final_out_dir)
    return df, results, placements


# ---------------------------------------------------- per-unit-cell report #
def _cell_content_box(template, margin_um=40.0):
    """Marker-relative design bounds (x0, x1, y0, y1) covering the alignment marker AND every pin
    disk, plus a margin. ``template.size_um`` is only the pin-cluster SIZE and implicitly assumes
    the pins start at the marker origin; when the marker is offset from the pin block (e.g. the
    D50/D100 cell, whose arrays sit ~325 um above an isolated bottom-left marker) that assumption
    crops the render. Deriving the true content extent here keeps every cell fully in frame."""
    m = template.marker_size_um if np.isfinite(template.marker_size_um) else 0.0
    xs, ys = [0.0, m], [0.0, m]
    for a in template.arrays:
        r = a.diameter_um / 2.0
        xs += [a.centers_um[:, 0].min() - r, a.centers_um[:, 0].max() + r]
        ys += [a.centers_um[:, 1].min() - r, a.centers_um[:, 1].max() + r]
    return (min(xs) - margin_um, max(xs) + margin_um,
            min(ys) - margin_um, max(ys) + margin_um)


def _resample_cell(field, placement, template, valid, res_um=2.0, box=None):
    """Resample a cell into DESIGN orientation (un-mirrored, y up), sampling ``field`` at each
    design pixel via the registration transform. ``box`` = (x0, x1, y0, y1) marker-relative design
    bounds to render (defaults to the full cell content box). Returns (image, extent_um)."""
    x0, x1, y0, y1 = box if box is not None else _cell_content_box(template)
    cw, ch = x1 - x0, y1 - y0
    nx, ny = max(1, int(round(cw / res_um))), max(1, int(round(ch / res_um)))
    gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * res_um,
                         y0 + (np.arange(ny) + 0.5) * res_um)
    cols, rows = placement.dxf_to_px(gx.ravel(), gy.ravel())
    ci = np.round(cols).astype(int); ri = np.round(rows).astype(int)
    H, W = field.shape
    m = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    out = np.full(gx.size, np.nan)
    idx = np.nonzero(m)[0]
    vv = valid[ri[m], ci[m]]
    out[idx[vv]] = field[ri[m][vv], ci[m][vv]]
    return out.reshape(ny, nx), (x0, x1, y0, y1)


def _cb(cb, label, fontsize=fs.LABEL):
    """Label a colorbar at the figure's presentation type size (label + tick labels together)."""
    cb.set_label(label, fontsize=fontsize); cb.ax.tick_params(labelsize=fontsize)
    return cb


def _image_cb(im, ax, label, *, size="4%", pad=0.16):
    """Colorbar matched to the IMAGE height, for an ``aspect='equal'`` map.

    ``plt.colorbar(ax=ax)`` sizes its axes from ax's box as it stands at creation time, but an
    aspect-equal image is afterwards shrunk to fit inside that box -- leaving a bar visibly taller
    than the picture it describes. An axes_grid1 divider is re-evaluated at draw time from the
    final position, so the two match exactly (measured 1.000 vs 1.056 for the shrink= form)."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size=size, pad=pad)
    return _cb(ax.figure.colorbar(im, cax=cax), label)


def _overlay_design(ax, template, res_by_array, box=None):
    """Red pin + alignment-marker outlines, dashed array dividers, per-array D/P + discrepancy."""
    if np.isfinite(template.marker_size_um):                             # markerless cells have none
        ax.add_patch(Rectangle((0, 0), template.marker_size_um, template.marker_size_um,
                               ec="red", fill=False, lw=1.6))            # alignment marker
    for a in template.arrays:
        ax.add_patch(Rectangle((a.x0_um - 18, a.y0_um - 18), a.width_um + 36, a.height_um + 36,
                               ec="red", ls="--", fill=False, lw=0.8))   # array divider
        for (x, y) in a.centers_um:
            ax.add_patch(Circle((x, y), a.diameter_um / 2, ec="red", fill=False, lw=0.35))
        r = res_by_array.get(a.array_id)
        txt = f"D{a.diameter_um:g} P{a.pitch_um:g}"
        if r is not None and np.isfinite(r.diameter_um) and a.diameter_um:
            txt += f" {100*(r.diameter_um-a.diameter_um)/a.diameter_um:+.0f}%"
        ax.text(a.x0_um - 10, a.y1_um + 26, txt, color="red", fontsize=fs.ANNOT_SM,
                va="bottom", ha="left")
    x0, x1, y0, y1 = box if box is not None else _cell_content_box(template)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("x (µm, design)", fontsize=fs.LABEL)
    ax.set_ylabel("y (µm, design)", fontsize=fs.LABEL)


def render_cell_report(scan, placement, template, res_by_array, params, path, box=None):
    """One report per unit cell: height (floor=0) + intensity with pin/marker/array overlays,
    a title with laser params + cell position, and a measured-vs-expected table per array.

    ``box`` = (x0, x1, y0, y1) marker-relative design bounds to render; defaults to the full cell
    content box. A partial snapshot passes its visible-pin box so the report frames the data, not
    5 mm of empty design space."""
    valid = scan.height_raw != 0
    box = box if box is not None else _cell_content_box(template)
    z_cell, ext = _resample_cell(scan.height_um, placement, template, valid, box=box)
    floor = np.nanpercentile(z_cell, 20) if np.isfinite(z_cell).any() else 0.0
    zdisp = z_cell - floor                                # local zero = trench floor
    inten = (_resample_cell(scan.intensity.astype(float), placement, template, valid, box=box)[0]
             if scan.intensity is not None else None)

    fig = plt.figure(figsize=(21, 12))
    gs = GridSpec(2, 2, width_ratios=[1.3, 1.0], figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)   # aligned x/y with height panel

    vmax = np.nanpercentile(zdisp, 98) if np.isfinite(zdisp).any() else 1.0
    im = ax0.imshow(zdisp, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax,
                    aspect="equal")
    _overlay_design(ax0, template, res_by_array, box=box)
    _image_cb(im, ax0, "height above floor (µm)")
    ax0.set_title("Height above floor", fontsize=fs.TITLE)
    if inten is not None:
        im1 = ax1.imshow(inten, origin="lower", extent=ext, cmap="gray", aspect="equal",
                         vmin=np.nanpercentile(inten, 2), vmax=np.nanpercentile(inten, 98))
        _image_cb(im1, ax1, "intensity")            # matched height keeps the x-axes aligned
    _overlay_design(ax1, template, res_by_array, box=box)
    ax1.set_title("Intensity", fontsize=fs.TITLE)
    ax0.tick_params(labelsize=fs.TICK); ax1.tick_params(labelsize=fs.TICK)  # ax1 shares ax0

    axr = fig.add_subplot(gs[:, 1]); axr.axis("off")

    cx = placement.origin_col * scan.x_um_per_px / 1000
    cy = placement.origin_row * scan.y_um_per_px / 1000
    lp = (f"P{params.passes}_S{params.speed:g}" if params and params.valid   # P{p}_S{s}, as in
          else "laser params: (fill CSV/cell_params.csv)")                   # the averaged report
    title = (f"Unit cell (row {placement.cell_row}, col {placement.cell_col})   —   {lp}\n"
             f"cell marker in sample: x = {cx:.2f} mm, y = {cy:.2f} mm   ·   "
             f"registration {placement.score:.2f}")
    axr.text(0.0, 1.0, title, transform=axr.transAxes, va="top", fontsize=fs.HEADLINE,
             weight="bold")

    def _f(v):
        return f"{v:.1f}" if np.isfinite(v) else "—"

    rows_t = []
    for a in sorted(template.arrays, key=lambda a: (a.band, a.col)):
        r = res_by_array.get(a.array_id)
        if r is None:
            continue
        rows_t.append([f"D{a.diameter_um:g} P{a.pitch_um:g}", _f(r.depth_um),
                       _f(r.top_diameter_um), _f(r.diameter_um), _f(r.base_diameter_um),
                       f"{r.meas_pitch_um:.0f}/{a.pitch_um:g}", f"{100*r.debris_fraction:.0f}"])
    col_labels = ["array (D/P)", "depth µm", "Ø top", "Ø nom", "Ø floor",
                  "pitch m/e", "debris%"]
    _th = min(0.82, 0.09 * (len(rows_t) + 1))            # scale height with #rows (no tall stretch)
    tbl = axr.table(cellText=rows_t, colLabels=col_labels, cellLoc="center",
                    bbox=[0.0, 0.80 - _th, 1.0, _th])
    tbl.auto_set_font_size(False); tbl.set_fontsize(fs.TABLE)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))   # headers need it at this size
    for c in range(len(col_labels)):
        tbl[0, c].set_facecolor("#dddddd"); tbl[0, c].set_text_props(weight="bold")

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


# ------------------------------------------------- laser-parameter summary #
# Presentation feedback on this pair of figures (2026-08 review): the viridis pass-count ramp has
# poor line-to-line contrast (its pale-yellow end all but vanishes on white), the lines were too
# thin, the design-target rule was unreadable, and the log speed axis was confusing because
# matplotlib labels the minor decade subdivisions ("3 x 10^2") alongside the plain 200/400/800
# majors. The helpers below fix each of those once, for both figures.

# Okabe-Ito, ordered dark -> warm so the ramp still reads as "more passes". High contrast on white
# AND separable under the common colour-vision deficiencies, which a sequential map is not.
_ORDINAL_COLORS = ("#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00", "#56B4E9", "#333333")
_ORDINAL_DASHES = ((0, ()), (0, (6, 2)), (0, (2, 1.6)), (0, (8, 2, 1.5, 2)))
PARAM_LINE_LW = 3.2                                  # was matplotlib's 1.5 default: "way too thin"


def _passes_style(passes_vals):
    """passes value -> (colour, dash). Redundant encoding: the dash still carries the series in
    greyscale, or for a reader who cannot separate two of the hues."""
    return {p: (_ORDINAL_COLORS[i % len(_ORDINAL_COLORS)],
                _ORDINAL_DASHES[i % len(_ORDINAL_DASHES)])
            for i, p in enumerate(passes_vals)}


def _speed_log_axis(ax, speeds):
    """Log speed axis labelled with the measured speeds ONLY.

    matplotlib also labels a log axis's minor ticks, so '3 x 10^2' / '6 x 10^2' appear between the
    explicit 200/400/800 majors and the axis reads as a muddle of two notations -- which is exactly
    what made the log scaling easy to miss. Minor ticks stay (they show the compression), their
    labels go."""
    ax.set_xscale("log")
    ax.set_xticks(list(speeds))
    ax.set_xticklabels([f"{s:g}" for s in speeds])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=3)


def _target_line(ax, depth_um=55.0):
    """The design-target rule drawn to be READ: a dark dashed line plus a caption in the RIGHT
    MARGIN, instead of the thin grey line and grey caption that got lost in the data.

    The caption sits outside the axes because this rule crosses the middle of both figures -- at
    presentation type size an in-axes caption lands on a data point wherever it is put. Both
    figures save with bbox_inches='tight', so the margin text is never cropped."""
    ax.axhline(depth_um, color="0.15", ls=(0, (7, 4)), lw=2.0, zorder=1.5)
    ax.text(1.008, depth_um, f"design\ntarget\n{depth_um:g} µm",
            transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=fs.NOTE,
            color="0.1", clip_on=False)


def _halo(lw=3.0):
    """White outline so a point callout stays readable where it lands on the data."""
    return [pe.withStroke(linewidth=lw, foreground="white")]


def _cell_callout(ax, label, x, y, *, at_right):
    """P{passes}_S{speed} callout on a cell marker. Points at the highest speed are labelled to the
    LEFT: at the right-hand end of the axis an outward label runs off the figure."""
    ax.annotate(label, (x, y), fontsize=fs.ANNOT, path_effects=_halo(), color="0.15", zorder=5,
                textcoords="offset points",
                xytext=(-8, 9) if at_right else (7, 9), ha="right" if at_right else "left")


def make_param_summary(df, out_dir):
    """Single figure: how laser parameters (passes × speed) drive etch depth and mid-diameter
    oversizing. One marker per unit cell, each labelled with its P{passes}_S{speed} id, one
    line per passes value across speed."""
    d = df[df["reliable"] & (df["passes"] > 0) & (df["speed"] > 0) & df["depth_um"].notna()].copy()
    if not len(d):
        print("No cells with laser params -> skipping param summary.")
        return
    d["overs"] = (d["diameter_um"] - d["drawn_diameter_um"]) / d["drawn_diameter_um"] * 100.0
    g = d.groupby(["cell_row", "cell_col"]).agg(
        passes=("passes", "first"), speed=("speed", "first"), label=("cell_label", "first"),
        depth=("depth_um", "median"), overs=("overs", "median")).reset_index()

    passes_vals = sorted(g["passes"].unique())
    style = _passes_style(passes_vals)
    speeds = sorted(g["speed"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(11, 11), sharex=True)
    s_max = max(speeds) if speeds else float("nan")
    for metric, ax, ylab, target in [
            ("depth", axes[0], "median etch depth (µm)", 55),
            ("overs", axes[1], "mid-Ø oversizing  (meas − drawn)/drawn  (%)", None)]:
        for pas in passes_vals:
            c, dash = style[pas]
            gg = g[g["passes"] == pas].sort_values("speed")
            ax.plot(gg["speed"], gg[metric], marker="o", color=c, ls=dash, lw=PARAM_LINE_LW,
                    ms=11, mew=1.2, mec="0.15", label=f"{pas} passes", zorder=3)
            # Only the depth panel is labelled: the callout is (passes, speed), which colour and x
            # position already give, and at this type size a second set collides in the lower panel.
            if metric == "depth":
                for _, r in gg.iterrows():
                    _cell_callout(ax, r["label"], r["speed"], r[metric],
                                  at_right=r["speed"] >= s_max)
        if target is not None:
            _target_line(ax, target)
        else:
            ax.axhline(0, color="0.35", ls="--", lw=1.2)
        ax.set_ylabel(ylab, fontsize=fs.LABEL); ax.tick_params(labelsize=fs.TICK)
        ax.margins(y=0.10)                               # headroom: callouts sit above their marker
        _speed_log_axis(ax, speeds); ax.grid(alpha=0.3)
    axes[1].set_xlabel("scan speed (mm/s) — log scale", fontsize=fs.LABEL)
    axes[0].legend(title="passes", fontsize=fs.LEGEND_SM, title_fontsize=fs.LEGEND_SM)
    axes[0].set_title("Laser-parameter effect on depth and diameter", fontsize=fs.TITLE)
    fig.tight_layout()
    p = Path(out_dir) / "figures" / "param_summary.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p}")


def make_param_depth_scatter(df, out_dir):
    """Companion to the depth panel of make_param_summary: instead of a single median per unit
    cell, scatter EVERY pin array's etch depth around that cell's median. Same axes (depth vs
    scan speed, one colour/line per passes value). Marker shape encodes the pin family --
    squares = D100 arrays (drawn Ø >= 75 um), triangles = D50 arrays (drawn Ø < 75 um)."""
    d = df[df["reliable"] & (df["passes"] > 0) & (df["speed"] > 0) & df["depth_um"].notna()].copy()
    if not len(d):
        print("No cells with laser params -> skipping depth scatter.")
        return
    d["family"] = np.select([d["drawn_diameter_um"] >= 200, d["drawn_diameter_um"] >= 75],
                            ["D300", "D100"], default="D50")

    # per-cell median depth == the 'average' the individual points scatter around (identical to
    # the value make_param_summary plots for the depth panel)
    med = d.groupby(["cell_row", "cell_col"]).agg(
        passes=("passes", "first"), speed=("speed", "first"),
        label=("cell_label", "first"), depth=("depth_um", "median")).reset_index()

    passes_vals = sorted(d["passes"].unique())
    style = _passes_style(passes_vals)
    speeds = sorted(d["speed"].unique())
    s_max = max(speeds) if speeds else float("nan")

    # diamonds = D300, squares = D100, triangles = D50 family (by drawn Ø). Each family is
    # sub-dodged and every point jittered in log10(speed) space so the cloud reads symmetrically on
    # the log x-axis (seeded so the figure is reproducible run-to-run).
    fam_style = {"D300": ("D", 0.0), "D100": ("s", +0.022), "D50": ("^", -0.022)}
    rng = np.random.default_rng(0)
    jit = 0.014

    fig, ax = plt.subplots(figsize=(12, 8))
    for fam, (marker, dodge) in fam_style.items():
        for pas in passes_vals:
            sub = d[(d["family"] == fam) & (d["passes"] == pas)]
            if not len(sub):
                continue
            x = sub["speed"].to_numpy() * 10.0 ** (dodge + rng.uniform(-jit, jit, len(sub)))
            ax.plot(x, sub["depth_um"], marker=marker, ls="none", ms=6.5, mew=0.5,
                    mfc=style[pas][0], mec="white", alpha=0.5, zorder=2)

    # per-passes line through the cell medians, drawn on top of its scatter
    for pas in passes_vals:
        c, dash = style[pas]
        gg = med[med["passes"] == pas].sort_values("speed")
        ax.plot(gg["speed"], gg["depth"], marker="o", color=c, ls=dash, lw=PARAM_LINE_LW,
                ms=12, mew=1.2, mec="0.15", label=f"{pas} passes", zorder=3)
        for _, r in gg.iterrows():
            _cell_callout(ax, r["label"], r["speed"], r["depth"], at_right=r["speed"] >= s_max)

    _target_line(ax, 55)
    _speed_log_axis(ax, speeds)
    ax.tick_params(labelsize=fs.TICK)
    ax.set_xlabel("scan speed (mm/s) — log scale", fontsize=fs.LABEL)
    ax.set_ylabel("etch depth (µm)", fontsize=fs.LABEL)
    ax.grid(alpha=0.3)
    ax.set_title("Etch depth vs laser parameters", fontsize=fs.TITLE)

    # ONE legend, not two in opposite corners (review: "move the legends together"). The pass-count
    # series fill the first column and the marker vocabulary the second -- matplotlib fills a
    # multi-column legend column-major, so each group is padded to the taller one's length.
    pass_handles = [Line2D([], [], marker="o", color=style[pv][0], ls=style[pv][1],
                           lw=PARAM_LINE_LW, ms=11, mec="0.15", label=f"{pv} passes")
                    for pv in passes_vals]
    shape_handles = []
    for fam, (marker, _) in fam_style.items():
        if not (d["family"] == fam).any():
            continue
        lo, hi = d.loc[d["family"] == fam, "drawn_diameter_um"].agg(["min", "max"])
        shape_handles.append(Line2D([], [], marker=marker, ls="none", mfc="0.4", mec="white",
                                    ms=9, label=f"{fam} array (drawn Ø {lo:g}–{hi:g} µm)"))
    shape_handles.append(Line2D([], [], marker="o", ls="-", color="0.4", ms=10,
                                label="cell median"))
    rows = max(len(pass_handles), len(shape_handles))
    pad = lambda h: h + [Line2D([], [], ls="none", label=" ") for _ in range(rows - len(h))]
    ax.legend(handles=pad(pass_handles) + pad(shape_handles), ncol=2, fontsize=fs.LEGEND_SM,
              loc="upper right", framealpha=0.92, handlelength=2.8, columnspacing=1.4)

    fig.tight_layout()
    p = Path(out_dir) / "figures" / "param_depth_scatter.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p}")


# --------------------------------------------- radial-average overlay figures #
def _combo_color(i, n):
    """Distinct colour for combo ``i`` of ``n`` overlaid profiles: tab10 for small sets, a
    continuous map (turbo) once there are more combos than tab10 has distinct colours."""
    if n <= 10:
        return plt.get_cmap("tab10")(i % 10)
    return plt.get_cmap("turbo")(i / max(1, n - 1))


def load_radial_sets(csv_path):
    """Read the radial-overlay comparison CSV -> list of sets, each a list of P{passes}_S{speed}
    labels (one row = one set; blank cells and ``#`` comment rows ignored). Returns None if the
    file is absent or has no entries, which signals 'compare every laser parameter present'."""
    p = Path(csv_path)
    if not p.exists():
        return None
    sets = []
    with p.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if row and str(row[0]).strip().startswith("#"):
                continue
            labels = [c.strip() for c in row if c.strip()]
            if labels:
                sets.append(labels)
    return sets or None


def make_radial_overlays(template, placements, params, res_by_cell, out_dir, sets_csv=None):
    """One figure per nominal geometry (per array in the unit cell), overlaying the mean-pin
    radial-average profile of several laser-parameter combos.

    The combos come from ``sets_csv`` (CSV/radial_sets.csv): each row is one comparison set of
    P{passes}_S{speed} labels -> len(arrays) figures per set, under radial_overlays/set{N}/. If
    that CSV is absent or empty, every laser parameter present in the sample is overlaid instead
    (a single 'all' set). Profiles are the (rc, prof) extract_array already computed per array,
    referenced to each cell's clean floor so the curves share z=0."""
    # label -> cell_id, keyed by both the canonical P{passes}_S{speed} and any CSV label. NOTE:
    # single-valued, so if two cells share the same passes/speed (a replicate) the later one wins
    # and the earlier replicate is dropped from the overlays -- warn rather than lose it silently.
    label_to_cell = {}
    for pl in placements:
        pr = params.get((pl.cell_row, pl.cell_col))
        if not (pr and pr.valid):
            continue
        key = f"P{pr.passes}_S{pr.speed:g}"
        if key in label_to_cell:
            print(f"WARNING: radial overlays: duplicate laser setting {key} on multiple cells; "
                  f"only one is shown per overlay (replicates not yet averaged).")
        label_to_cell[key] = pl.cell_id
        if pr.label:
            label_to_cell[pr.label] = pl.cell_id
    if not label_to_cell:
        print("No labelled cells with laser params -> skipping radial overlays.")
        return

    sets = load_radial_sets(sets_csv) if sets_csv else None
    if sets is None:                          # absent/empty CSV -> compare every laser parameter
        seen = {}                             # (passes, speed) -> canonical label, ordered
        for pl in placements:
            pr = params.get((pl.cell_row, pl.cell_col))
            if pr and pr.valid:
                seen[(pr.passes, pr.speed)] = f"P{pr.passes}_S{pr.speed:g}"
        named_sets = {"all": [seen[k] for k in sorted(seen)]}
        print(f"radial overlays: no radial_sets.csv (or empty) -> comparing all "
              f"{len(named_sets['all'])} laser params per geometry")
    else:
        named_sets = {f"set{i + 1}": s for i, s in enumerate(sets)}

    root = Path(out_dir) / "figures" / "radial_overlays"
    n_fig = 0
    for set_name, combos in named_sets.items():
        sdir = root / set_name
        sdir.mkdir(parents=True, exist_ok=True)
        missing = [c for c in combos if c not in label_to_cell]
        if missing:
            print(f"  [{set_name}] no registered cell for: {', '.join(missing)} "
                  f"(available: {', '.join(sorted(label_to_cell))})")
        for a in sorted(template.arrays, key=lambda a: a.array_id):
            fig, ax = plt.subplots(figsize=(9, 6))
            n_lines = 0
            for i, combo in enumerate(combos):
                cid = label_to_cell.get(combo)
                if cid is None:
                    continue
                res = res_by_cell.get(cid, {}).get(a.array_id)
                if res is None or res.rc is None or res.prof is None:
                    continue
                rc = np.asarray(res.rc, float)
                prof = np.asarray(res.prof, float)
                if not np.isfinite(prof).any():
                    continue
                floor = res.floor_um if np.isfinite(res.floor_um) else np.nanmin(prof)
                z = prof - floor
                depth = res.depth_um if np.isfinite(res.depth_um) else np.nanmax(z)
                ax.plot(rc, z, "-", lw=2.6, color=_combo_color(i, len(combos)),
                        label=f"{combo} — depth {depth:.0f} µm")
                n_lines += 1
            if not n_lines:
                plt.close(fig)
                continue
            ax.axhline(0, color="grey", lw=0.8, ls="-")
            ax.axvline(a.diameter_um / 2, color="green", ls=":", lw=1.2,
                       label=f"drawn radius {a.diameter_um/2:g} µm")
            ax.set_xlabel("radius from pin centre (µm)", fontsize=fs.LABEL)
            ax.set_ylabel("mean height above clean floor (µm)", fontsize=fs.LABEL)
            ax.set_title(f"Mean radial pin profile — Ø {a.diameter_um:g} µm, "
                         f"pitch {a.pitch_um:g} µm", fontsize=fs.TITLE)
            ax.tick_params(labelsize=fs.TICK); ax.grid(alpha=0.3)
            ax.legend(fontsize=fs.LEGEND_SM, ncol=2 if n_lines > 8 else 1)
            fig.tight_layout()
            fname = f"a{a.array_id:02d}_D{a.diameter_um:g}_P{a.pitch_um:g}.png"
            fig.savefig(sdir / fname, dpi=170); plt.close(fig)
            n_fig += 1
    print(f"Wrote {n_fig} radial-overlay figures -> {root}")


# ------------------------------------------------------------- presentation #
def _flip_lr(a):
    return a[:, ::-1]


def save_sample_heightmap(scan, path, ds=4):
    """Full-sample height map, flipped left-right to DESIGN orientation for presentation.

    x/y are physical distance in mm from the assembled raster's bottom-left corner (not stage
    coordinates -- the assembled origin is arbitrary), so cell size and spacing can be read off
    the axes directly."""
    valid = scan.height_raw != 0
    z = np.where(valid, scan.height_um, np.nan)[::ds, ::ds]
    z = _flip_lr(z)                                       # un-mirror for presentation
    hh, ww = z.shape
    # mm extent from the DECIMATED pixel pitch; with aspect='equal' in mm the figure keeps true
    # physical proportions even when the scan's x and y pitches differ
    w_mm = ww * ds * scan.x_um_per_px / 1000.0
    h_mm = hh * ds * scan.y_um_per_px / 1000.0
    fig_w = 11.0                                          # size the figure to the physical aspect
    fig, ax = plt.subplots(figsize=(fig_w, max(3.5, fig_w * h_mm / w_mm)))
    im = ax.imshow(z, origin="lower", extent=(0.0, w_mm, 0.0, h_mm), aspect="equal", cmap="viridis",
                   vmin=np.nanpercentile(z, 2), vmax=np.nanpercentile(z, 98))
    ax.set_title("Assembled sample height", fontsize=fs.TITLE)
    ax.set_xlabel("x (mm, design orientation)", fontsize=fs.LABEL)
    ax.set_ylabel("y (mm)", fontsize=fs.LABEL)
    ax.tick_params(labelsize=fs.TICK)
    _image_cb(im, ax, "height (µm)")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def save_sample_overview(scan, template, placements, path, ds=6):
    """Labeled cell map (design orientation): each cell's (row,col) + pin overlay, un-mirrored."""
    H, W = scan.height_raw.shape
    valid = scan.height_raw != 0
    base = np.where(valid, scan.intensity if scan.intensity is not None else scan.height_um,
                    np.nan)[::ds, ::ds]
    base = _flip_lr(base)
    allc = template.all_centers_um()
    bx0, bx1, by0, by1 = _cell_content_box(template)
    ccx, ccy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)     # true cell centre (marker + pins)
    hh, ww = base.shape                                 # size the figure to the image aspect
    fig_w = 12.0
    fig, ax = plt.subplots(figsize=(fig_w, max(4.0, fig_w * hh / ww)))
    ax.imshow(base, origin="lower", cmap="gray",
              vmin=np.nanpercentile(base, 2), vmax=np.nanpercentile(base, 98))
    for p in placements:
        cols, rows = p.dxf_to_px(allc[:, 0], allc[:, 1])
        fx = (W - 1 - cols) / ds                          # apply the same L-R flip
        fy = rows / ds
        ax.plot(fx, fy, "c.", ms=0.5)
        cx, cy = p.dxf_to_px(ccx, ccy)
        ax.text((W - 1 - cx) / ds, cy / ds, f"({p.cell_row},{p.cell_col})",
                color="yellow", ha="center", va="center", fontsize=fs.OVERLAY, weight="bold")
    ax.set_title(f"{len(placements)} unit cells", fontsize=fs.TITLE)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)


# ============================================================================ #
# Multi-snapshot mode: DISJOINT crops of ONE uniform unit cell                  #
# ============================================================================ #
# A dataset here is a set of independent VK4 snapshots (e.g. a "Center" interior crop and a
# "TopLeft" corner crop) of the SAME uniform-cell DXF geometry, with NO known relative position.
# Each snapshot is registered INDEPENDENTLY (phase-only for an interior crop) and measured on
# whatever pins it contains; the "combined height map" is a side-by-side MONTAGE of the tiles, not
# a spatial mosaic (assemble.py is deliberately bypassed -- its overlap/step model cannot place
# disjoint crops and would alias on the periodic pin lattice). All snapshots share one laser dose.

_RASTER_RE = __import__("re").compile(r"_Y\d+_X\d+$", __import__("re").IGNORECASE)


def _label_from_name(path):
    """Human label for a snapshot from its filename: the trailing '_'-token (e.g. 'Center',
    'TopLeft'). Falls back to the whole stem when there is no underscore."""
    stem = Path(path).stem
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def snapshots_from_dir(vk4_dir):
    """Discover disjoint snapshots in ``vk4_dir``: every ``*.vk4`` that is NOT a ``_Y{n}_X{m}``
    raster tile. Label each by its trailing filename token; if those collide, fall back to the
    full stem so identities never merge. Returns a sorted list of ``(Path, label)``."""
    files = sorted(p for p in Path(vk4_dir).glob("*.vk4") if not _RASTER_RE.search(p.stem))
    labels = [_label_from_name(p) for p in files]
    if len(set(labels)) != len(labels):                  # collision -> disambiguate, never merge
        labels = [p.stem for p in files]
    return list(zip(files, labels))


def _parse_ps_label(text):
    """Parse a single ``P{passes}_S{speed}`` dose token -> (passes, speed); (0, nan) if absent."""
    if not text:
        return 0, float("nan")
    m = __import__("re").search(r"P(\d+)_S(\d+(?:\.\d+)?)", str(text))
    if not m:
        return 0, float("nan")
    return int(m.group(1)), float(m.group(2))


def _visible_content_box(scan, placement, template, margin_um=None):
    """Marker-relative design bbox (x0, x1, y0, y1) of the pins this placement maps INSIDE the
    scan, plus a pin radius and ~1-pitch margin. ``None`` if fewer than 3 pins are in frame. Frames
    a partial snapshot on the region it actually covers, not the whole (mostly-empty) design cell."""
    allc = template.all_centers_um()
    cols, rows = placement.dxf_to_px(allc[:, 0], allc[:, 1])
    H, W = scan.height_raw.shape
    inb = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    if int(inb.sum()) < 3:
        return None
    if margin_um is None:
        margin_um = float(np.median([a.pitch_um for a in template.arrays]))
    r = 0.5 * max(a.diameter_um for a in template.arrays)
    xs, ys = allc[inb, 0], allc[inb, 1]
    return (xs.min() - margin_um - r, xs.max() + margin_um + r,
            ys.min() - margin_um - r, ys.max() + margin_um + r)


def _snapshot_panel(scan, placement, template, res_um=2.0):
    """Design-oriented crops of one snapshot's visible region. Returns
    ``(zdisp, inten, extent_um)`` where ``zdisp`` is height referenced to the local trench floor
    (floor = 0) and ``inten`` is the same-footprint intensity image (``None`` if the scan has no
    intensity channel). ``(None, None, None)`` when nothing is in frame."""
    box = _visible_content_box(scan, placement, template)
    if box is None:
        return None, None, None
    valid = scan.height_raw != 0
    z, ext = _resample_cell(scan.height_um, placement, template, valid, res_um=res_um, box=box)
    if not np.isfinite(z).any():
        return None, None, None
    floor = np.nanpercentile(z, 20)
    inten = None
    if getattr(scan, "intensity", None) is not None:                # same box/placement -> aligned
        inten = _resample_cell(scan.intensity.astype(float), placement, template, valid,
                               res_um=res_um, box=box)[0]
    return z - floor, inten, ext


def _stitch_snapshot_panels(panels, gutter_px=24):
    """Lay design-oriented panels side by side on shared canvases (vertically centred, NaN gutter +
    NaN padding). Panels are INDEPENDENT captures with no known relative position, so the horizontal
    arrangement is presentation only -- never a spatial mosaic. Height and intensity use the SAME
    layout (same boxes) so the two rows line up column-for-column.

    ``panels`` : list of ``(label, placement, zdisp, inten)`` (``inten`` may be ``None``). Returns
    ``(hcanvas, icanvas, boxes)``; ``icanvas`` is ``None`` when no panel has intensity. Each box is
    ``dict(label, placement, c0, c1, r0, r1)`` locating that panel within the canvases."""
    heights = [z.shape[0] for _, _, z, _ in panels]
    widths = [z.shape[1] for _, _, z, _ in panels]
    hc = max(heights)
    wc = sum(widths) + gutter_px * (len(panels) - 1)
    hcanvas = np.full((hc, wc), np.nan)
    has_int = any(ip is not None for _, _, _, ip in panels)
    icanvas = np.full((hc, wc), np.nan) if has_int else None
    boxes, c = [], 0
    for (label, pl, z, ip), w, h in zip(panels, widths, heights):
        r0 = (hc - h) // 2
        hcanvas[r0:r0 + h, c:c + w] = z
        if icanvas is not None and ip is not None:
            icanvas[r0:r0 + h, c:c + w] = ip
        boxes.append(dict(label=label, placement=pl, c0=c, c1=c + w, r0=r0, r1=r0 + h))
        c += w + gutter_px
    return hcanvas, icanvas, boxes


def build_snapshot_montage(tiles, template, *, res_um=2.0, gutter_px=24):
    """Pure builder (no I/O) for the snapshot montage: independent, floor-referenced,
    design-oriented panels stitched side by side, with a matching intensity montage. Returns
    ``dict(canvas, intensity, boxes, vmax, ivmin, ivmax, labels, gutter_px)`` or ``None`` when no
    tile has a visible region. Separate from :func:`save_snapshot_montage` so the self-test can
    assert on the composited arrays directly."""
    panels = []
    for tile in tiles:
        z, ip, _ = _snapshot_panel(tile["scan"], tile["placement"], template, res_um=res_um)
        if z is not None:
            panels.append((tile["label"], tile["placement"], z, ip))
    if not panels:
        return None
    hcanvas, icanvas, boxes = _stitch_snapshot_panels(panels, gutter_px=gutter_px)
    vmax = max(np.nanpercentile(z, 98) for _, _, z, _ in panels)
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
    ivmin = ivmax = None
    if icanvas is not None and np.isfinite(icanvas).any():
        ivmin = float(np.nanpercentile(icanvas, 2))
        ivmax = float(np.nanpercentile(icanvas, 98))
    return dict(canvas=hcanvas, intensity=icanvas, boxes=boxes, vmax=vmax,
                ivmin=ivmin, ivmax=ivmax, labels=[lab for lab, _, _, _ in panels],
                gutter_px=gutter_px)


def _annotate_snapshot_panels(ax, boxes, canvas_h, *, fontsize=fs.NOTE, scale=1.0):
    """Label each tiled panel with its snapshot name (Center / TopLeft / ...) in a boxed callout
    above the panel, so the reader can tell which crop is which. Adds headroom above the panels so
    the labels are never clipped. ``canvas_h`` is the composited canvas height in pixels.

    ``scale`` converts canvas pixels to the axes' data units (mm/px when the montage is drawn on a
    physical axis); the default 1.0 keeps the callouts in pixel coordinates."""
    ax.set_ylim(-0.03 * canvas_h * scale, canvas_h * 1.16 * scale)   # headroom for the labels
    for b in boxes:
        ax.text(0.5 * (b["c0"] + b["c1"]) * scale, (b["r1"] + 0.02 * canvas_h) * scale, b["label"],
                color="black", ha="center", va="bottom", fontsize=fontsize, weight="bold",
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))


def save_snapshot_heightmap(tiles, template, path, *, res_um=2.0, gutter_px=24, montage=None):
    """Full-sample HEIGHT map for the multi-snapshot mode, written under the SAME filename as the
    single-scan heightmap (``sample_heightmap.png``): each snapshot's floor-referenced,
    design-oriented crop tiled side by side and annotated with its snapshot name. Panels are
    separate captures, so the inter-panel spacing is presentation only, not a spatial mosaic."""
    m = montage if montage is not None else build_snapshot_montage(
        tiles, template, res_um=res_um, gutter_px=gutter_px)
    if m is None:
        print("No snapshot has a visible in-frame region -> skipping height map.")
        return
    canvas, boxes, vmax = m["canvas"], m["boxes"], m["vmax"]
    hh, ww = canvas.shape                                 # size the figure to the image aspect so
    fig_w = max(8.0, 4.8 * len(boxes))                    # the colorbar tracks the plot, not empty space
    fig, ax = plt.subplots(figsize=(fig_w, max(3.2, fig_w * (1.22 * hh) / ww)))
    # the canvas is composited at a uniform res_um per pixel, so an mm axis carries a true SCALE;
    # the origin and the panel-to-panel offsets are presentation only (independent captures + a
    # gutter), hence the 'within a panel' caveat on the x label
    mm_px = res_um / 1000.0
    im = ax.imshow(canvas, origin="lower", extent=(0.0, ww * mm_px, 0.0, hh * mm_px),
                   cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
    ax.set_xlabel("x (mm — physical within a panel; panels are separate captures)",
                  fontsize=fs.LABEL)
    ax.set_ylabel("y (mm)", fontsize=fs.LABEL)
    ax.tick_params(labelsize=fs.TICK)
    _annotate_snapshot_panels(ax, boxes, canvas.shape[0], scale=mm_px)
    _image_cb(im, ax, "height above local floor (µm)")
    ax.set_title("Sample height — tiled snapshots", fontsize=fs.TITLE)
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {path}")


def save_snapshot_overview(tiles, template, path, *, res_um=2.0, gutter_px=24, montage=None):
    """INTENSITY overview for the multi-snapshot mode, written under the SAME filename as the
    single-scan overview (``intensity_map.png``): the same snapshots tiled in the same layout as
    the height map, annotated with their snapshot names."""
    m = montage if montage is not None else build_snapshot_montage(
        tiles, template, res_um=res_um, gutter_px=gutter_px)
    if m is None or m["intensity"] is None or m["ivmax"] is None:
        print("No snapshot intensity available -> skipping intensity overview.")
        return
    icanvas, boxes = m["intensity"], m["boxes"]
    hh, ww = icanvas.shape
    fig_w = max(8.0, 4.8 * len(boxes))
    fig, ax = plt.subplots(figsize=(fig_w, max(3.2, fig_w * (1.22 * hh) / ww)))
    imi = ax.imshow(icanvas, origin="lower", cmap="gray",
                    vmin=m["ivmin"], vmax=m["ivmax"], aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    _annotate_snapshot_panels(ax, boxes, icanvas.shape[0])
    _image_cb(imi, ax, "intensity")
    ax.set_title("Sample intensity — tiled snapshots", fontsize=fs.TITLE)
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {path}")


# ------------------------------------------------------ 3D centre-block maps #
def _center_block_box(scan, placement, array, block=5, margin_um=None):
    """Design-space bbox (x0, x1, y0, y1) covering the ``block``x``block`` pins at the CENTRE of
    ``array`` that are visible in ``scan``. For a fully-captured array (raster cell) this is the
    array's geometric centre; for a phase-only snapshot (no absolute centre) it centres on the
    visible pins. Fewer pins are used near a frame edge. ``None`` if < 4 pins of the block are in
    frame."""
    allc = array.centers_um
    cols, rows = placement.dxf_to_px(allc[:, 0], allc[:, 1])
    H, W = scan.height_raw.shape
    inb = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    if int(inb.sum()) < 4:
        return None
    vx, vy = allc[inb, 0], allc[inb, 1]
    px, py = array.pitch_x_um, array.pitch_y_um
    cx = vx[np.argmin(np.abs(vx - np.median(vx)))]       # snap centre to the nearest visible pin
    cy = vy[np.argmin(np.abs(vy - np.median(vy)))]
    hw = block / 2.0 - 0.3                                # 5 -> 2.2 pitches: keeps +-2, drops +-3
    sel = (np.abs(vx - cx) <= hw * px) & (np.abs(vy - cy) <= hw * py)
    if int(sel.sum()) < 4:
        return None
    r = array.diameter_um / 2.0
    m = margin_um if margin_um is not None else 0.3 * min(px, py)
    sx, sy = vx[sel], vy[sel]
    return (sx.min() - r - m, sx.max() + r + m, sy.min() - r - m, sy.max() + r + m)


def save_3d_pin_map(scan, placement, template, array, path, *, res_um=2.0, block=5, param_label=""):
    """Render a 3D height surface of the centre ``block``x``block`` pins of ``array`` to ``path``.

    Height is design-oriented (via the registration transform) and referenced to the local trench
    floor (floor = 0). The axes are rendered at TRUE physical aspect (one micron is the same length
    on x, y and z), so pin height/taper read at real proportions. Returns True if a map was written,
    False when too little of the block is in frame.

    Drawn as a presentation figure throughout: low camera, tight framing, large type, 300 dpi."""
    box = _center_block_box(scan, placement, array, block=block)
    if box is None:
        return False
    valid = scan.height_raw != 0
    z, ext = _resample_cell(scan.height_um, placement, template, valid, res_um=res_um, box=box)
    if not np.isfinite(z).any():
        return False
    z = z - np.nanpercentile(z, 20)                      # local trench floor -> 0
    x0, x1, y0, y1 = ext
    ny, nx = z.shape
    xx, yy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    zf = np.nan_to_num(z, nan=0.0)                        # scan-edge gaps drop to the floor
    vmax = np.nanpercentile(z, 99.5)
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, zf, cmap="viridis", vmin=0, vmax=vmax,
                           rcount=160, ccount=160, linewidth=0, antialiased=True)
    ax.set_xlabel("x (µm)", labelpad=22, fontsize=fs.LABEL)   # labelpad clears the (larger) ticks
    ax.set_ylabel("y (µm)", labelpad=22, fontsize=fs.LABEL)
    ax.set_zlabel("")                                     # height shown on the colorbar (no overlap)
    ax.set_title(f"3D height — D{array.diameter_um:g} P{array.pitch_um:g}", fontsize=fs.HEADLINE)
    ax.view_init(elev=20, azim=-55)                       # low camera, so height reads at a glance
    ztop = float(np.nanmax(zf))
    ax.set_zlim(0, max(1.0, ztop)); ax.set_zticks([0, round(ztop)])  # only 0 + top tick (rest cramp)
    ax.tick_params(labelsize=fs.TICK)
    dz = max(1e-6, ztop - float(np.nanmin(zf)))
    # TRUE aspect (1 µm is the same length on x, y, z), zoomed in to fill the frame.
    ax.set_box_aspect((x1 - x0, y1 - y0, dz), zoom=1.4)
    cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.22)
    cb.set_label("height above floor (µm)", fontsize=fs.LABEL)
    cb.ax.tick_params(labelsize=fs.TICK)
    if param_label:                                       # laser-parameter info box, top-left
        ax.text2D(0.02, 0.98, param_label, transform=ax.transAxes, fontsize=fs.OVERLAY,
                  va="top", ha="left",
                  bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.5", alpha=0.9))
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    return True


def write_3d_pin_maps(figures_dir, items, template, *, res_um=2.0):
    """Write a 3D centre-5x5 height map per array for each item into ``figures_dir/3D height map/``.

    ``items`` : list of ``(subdir, scan, placement, param_label)`` -- ``subdir`` is a per-cell /
    per-snapshot folder name (e.g. ``cell_x2_y1`` or ``Center``), or ``None`` to write straight into
    the root (single cell); ``param_label`` is the laser-parameter string shown in a top-left info
    box (``""`` to omit). One PNG per array, named ``array{id}_D{d}_P{p}.png``. Every map is
    presentation-styled (see :func:`save_3d_pin_map`), so any of them can go straight on a slide."""
    root = Path(figures_dir) / "3D height map"
    n = 0
    for sub, scan, placement, param_label in items:
        d = root / sub if sub else root
        for a in template.arrays:
            fname = f"array{a.array_id:02d}_D{a.diameter_um:g}_P{a.pitch_um:g}.png"
            if save_3d_pin_map(scan, placement, template, a, d / fname, res_um=res_um,
                               param_label=param_label):
                n += 1
    print(f"Wrote {n} 3D centre-5×5 height maps -> {root}")
    return n


# ------------------------------------- averaging over snapshots (replicates) #
def _average_results(res_list, dose_label="averaged"):
    """Average same-array PinFinResults from the disjoint snapshots (replicate views of one uniform
    cell) into a single result: scalar metrics -> nanmean, the mean-pin radial profile -> per-bin
    nanmean on the shared radius axis, flags unioned. Marked ``averaged`` / phase-only so no absolute
    position is implied. Returns None if the list is empty."""
    from extract import PinFinResult
    rs = [r for r in res_list if r is not None]
    if not rs:
        return None
    base = rs[0]
    avg_fields = ("pitch_um pitch_x_um pitch_y_um diameter_um base_diameter_um top_diameter_um "
                  "depth_um floor_um top_um lattice_strength coverage n_cells meas_pitch_um "
                  "meas_pitch_x_um meas_pitch_y_um floor_flatness_um debris_fraction pin_sat_frac "
                  "reg_score cx_um cy_um").split()

    def _m(attr):
        v = np.array([getattr(r, attr) for r in rs], float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    flags = sorted({f for r in rs for f in (r.flags.split(";") if r.flags else []) if f})
    avg = PinFinResult(
        filename=f"averaged_b{base.band}c{base.col}_D{base.nominal_diameter_um:g}",
        vk4_stem=dose_label, cell_id=0, array_id=base.array_id, band=base.band, col=base.col,
        passes=base.passes, speed=base.speed,
        nominal_diameter_um=base.nominal_diameter_um, target_diameter_um=base.target_diameter_um,
        nominal_pitch_um=base.nominal_pitch_um,
        reg_method="averaged", absolute_origin=False, ambiguous_axes="xy",
        base_extrapolated=any(getattr(r, "base_extrapolated", False) for r in rs),
        flags=";".join(flags))
    for a in avg_fields:
        setattr(avg, a, _m(a))
    profs = [(np.asarray(r.rc, float), np.asarray(r.prof, float)) for r in rs
             if getattr(r, "rc", None) is not None and getattr(r, "prof", None) is not None]
    if profs:
        rc0 = profs[0][0]
        stack = [p for (rc, p) in profs if p.shape == rc0.shape]
        if stack:
            avg.rc = rc0
            avg.prof = np.nanmean(np.vstack(stack), axis=0)
    return avg


def render_snapshot_cell_report(tiles, template, avg_by_array, dose_label, path, *, res_um=2.0):
    """Averaged cell report for a snapshot dataset: the snapshots tiled (height over intensity) on
    the left, and a per-array table AVERAGED over the snapshots on the right."""
    m = build_snapshot_montage(tiles, template, res_um=res_um)
    if m is None:
        return
    canvas, icanvas, boxes = m["canvas"], m["intensity"], m["boxes"]
    fig = plt.figure(figsize=(21, 10))                    # wider: the table needs it at table size
    gs = GridSpec(2, 2, width_ratios=[1.5, 1.0], figure=fig)
    axh = fig.add_subplot(gs[0, 0]); axi = fig.add_subplot(gs[1, 0])
    imh = axh.imshow(canvas, origin="lower", cmap="viridis", vmin=0, vmax=m["vmax"], aspect="equal")
    axh.set_xticks([]); axh.set_yticks([])
    axh.set_title("Height above floor", fontsize=fs.TITLE)
    _annotate_snapshot_panels(axh, boxes, canvas.shape[0], fontsize=fs.TITLE)
    _image_cb(imh, axh, "height above local floor (µm)")
    if icanvas is not None and m["ivmax"] is not None:
        imi = axi.imshow(icanvas, origin="lower", cmap="gray",
                         vmin=m["ivmin"], vmax=m["ivmax"], aspect="equal")
        _annotate_snapshot_panels(axi, boxes, icanvas.shape[0], fontsize=fs.TITLE)
        _image_cb(imi, axi, "intensity")
    axi.set_xticks([]); axi.set_yticks([])
    axi.set_title("Intensity", fontsize=fs.TITLE)

    axr = fig.add_subplot(gs[:, 1]); axr.axis("off")
    # snapshot count/names are dropped from the title -- each panel is already labelled with its
    # snapshot name by _annotate_snapshot_panels, so nothing is lost
    axr.text(0.0, 1.0, f"Averaged cell — {dose_label}",
             transform=axr.transAxes, va="top", fontsize=fs.HEADLINE, weight="bold")

    def _f(v):
        return f"{v:.1f}" if np.isfinite(v) else "—"

    rows_t = []
    for a in sorted(template.arrays, key=lambda a: (a.band, a.col)):
        r = avg_by_array.get(a.array_id)
        if r is None:
            continue
        rows_t.append([f"D{a.diameter_um:g} P{a.pitch_um:g}", _f(r.depth_um), _f(r.top_diameter_um),
                       _f(r.diameter_um), _f(r.base_diameter_um),
                       f"{r.meas_pitch_um:.0f}/{a.pitch_um:g}", f"{100 * r.debris_fraction:.0f}"])
    col_labels = ["array (D/P)", "depth µm", "Ø top", "Ø nom", "Ø floor",
                  "pitch m/e", "debris%"]
    tbl_bottom = 0.80
    if rows_t:
        _th = min(0.82, 0.09 * (len(rows_t) + 1))
        tbl_bottom = 0.80 - _th
        tbl = axr.table(cellText=rows_t, colLabels=col_labels, cellLoc="center",
                        bbox=[0.0, tbl_bottom, 1.0, _th])
        tbl.auto_set_font_size(False); tbl.set_fontsize(fs.TABLE)
        tbl.auto_set_column_width(col=list(range(len(col_labels))))  # headers need it at this size
        for c in range(len(col_labels)):
            tbl[0, c].set_facecolor("#dddddd"); tbl[0, c].set_text_props(weight="bold")
    # Mean radial pin profile in the space left under the table (one line per array geometry, same
    # curve as figures/radial_overlays/). Skipped when a tall table leaves no room.
    _ph = (tbl_bottom - 0.10) - 0.06
    if _ph >= 0.22:
        axp = axr.inset_axes([0.10, 0.06, 0.86, _ph])
        drew = False
        for a in sorted(template.arrays, key=lambda a: a.array_id):
            r = avg_by_array.get(a.array_id)
            if r is None or getattr(r, "rc", None) is None or getattr(r, "prof", None) is None:
                continue
            rc = np.asarray(r.rc, float); prof = np.asarray(r.prof, float)
            if not np.isfinite(prof).any():
                continue
            floor = r.floor_um if np.isfinite(r.floor_um) else np.nanmin(prof)
            z = prof - floor
            depth = r.depth_um if np.isfinite(r.depth_um) else np.nanmax(z)
            ln, = axp.plot(rc, z, "-", lw=2.4,
                           label=f"D{a.diameter_um:g} P{a.pitch_um:g} — depth {depth:.0f} µm")
            axp.axvline(a.diameter_um / 2, color=ln.get_color(), ls=":", lw=1.1)
            drew = True
        if drew:
            axp.axhline(0, color="grey", lw=0.8)
            axp.set_xlabel("radius from pin centre (µm)", fontsize=fs.LABEL)
            axp.set_ylabel("mean height above floor (µm)", fontsize=fs.LABEL)
            axp.set_title("Mean radial pin profile — averaged over snapshots "
                          "(dotted = drawn radius)", fontsize=fs.TITLE_SM)
            axp.tick_params(labelsize=fs.TICK); axp.grid(alpha=0.3)
            axp.legend(fontsize=fs.LEGEND_SM)
        else:
            axp.remove()
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Wrote {path}")


def make_snapshot_radial_overlays(template, avg_by_array, out_dir, dose_label=""):
    """One radial-profile figure per array geometry, showing the mean-pin profile AVERAGED over the
    snapshots (into figures/radial_overlays/, matching the tiled workflow's naming)."""
    root = Path(out_dir) / "figures" / "radial_overlays"; root.mkdir(parents=True, exist_ok=True)
    n = 0
    for a in sorted(template.arrays, key=lambda a: a.array_id):
        res = avg_by_array.get(a.array_id)
        if res is None or getattr(res, "rc", None) is None or getattr(res, "prof", None) is None:
            continue
        rc = np.asarray(res.rc, float); prof = np.asarray(res.prof, float)
        if not np.isfinite(prof).any():
            continue
        floor = res.floor_um if np.isfinite(res.floor_um) else np.nanmin(prof)
        z = prof - floor
        depth = res.depth_um if np.isfinite(res.depth_um) else np.nanmax(z)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(rc, z, "-", lw=2.6, label=f"{dose_label} — depth {depth:.0f} µm")
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(a.diameter_um / 2, color="green", ls=":", lw=1.6,
                   label=f"drawn radius {a.diameter_um/2:g} µm")
        ax.set_xlabel("radius from pin centre (µm)", fontsize=fs.LABEL)
        ax.set_ylabel("mean height above clean floor (µm)", fontsize=fs.LABEL)
        ax.set_title(f"Mean radial pin profile — Ø {a.diameter_um:g} µm, "
                     f"pitch {a.pitch_um:g} µm", fontsize=fs.TITLE)
        ax.tick_params(labelsize=fs.TICK); ax.grid(alpha=0.3); ax.legend(fontsize=fs.LEGEND_SM)
        fig.tight_layout()
        fig.savefig(root / f"a{a.array_id:02d}_D{a.diameter_um:g}_P{a.pitch_um:g}.png", dpi=170)
        plt.close(fig)
        n += 1
    print(f"Wrote {n} averaged radial-overlay figures -> {root}")


def _save_snapshot_provenance(figures_dir, snapshots, dxf_path, passes, speed):
    """Self-describing provenance for a multi-snapshot run: copy the DXF, zip the exact snapshot
    VK4s, and write ``run_manifest.json`` (git commit + inputs + labels + shared dose)."""
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)
    dxf_path = Path(dxf_path)
    if dxf_path.exists():
        shutil.copy2(dxf_path, figures_dir / dxf_path.name)
    zpath = figures_dir / "vk4_source.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, _label in snapshots:
            p = Path(path)
            if p.exists():
                zf.write(p, arcname=p.name)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(HERE),
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "mode": "multi-snapshot",
        "dxf": str(dxf_path),
        "passes": passes,
        "speed": speed,
        "n_snapshots": len(snapshots),
        "snapshots": [{"file": Path(p).name, "label": lab} for p, lab in snapshots],
    }
    (figures_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote multi-snapshot provenance -> {figures_dir}")


def analyze_multi_snapshot(snapshots, out_dir, dxf_path, *, passes=0, speed=float("nan"),
                           cell_label="", make_qc=False, jobs=None, results_root=None):
    """Analyze a set of DISJOINT snapshots (crops) of ONE uniform unit-cell DXF.

    ``snapshots`` : list of ``(vk4_path, label)``. Each is registered independently (phase-only for
    an interior crop) and measured on the pins it contains; all share one laser (passes, speed).
    Writes ``legacy/measurements.csv`` (one row per array per snapshot, tagged with ``snapshot``),
    a per-snapshot report, and one side-by-side montage. Absolute pin position is never claimed.
    Returns ``(df, results, tiles)``.

    ``results_root`` overrides the root that ``out_dir`` must live under (default ``Results/``);
    ``run_row.py`` leaves it as ``None`` in production and the selftest points it at a temp dir.
    """
    snapshots = [(Path(p), str(lab)) for p, lab in snapshots]
    if not snapshots:
        raise SystemExit("analyze_multi_snapshot: no snapshots provided.")
    labels = [lab for _, lab in snapshots]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"analyze_multi_snapshot: snapshot labels must be unique, got {labels}.")

    stage_dir, final_out_dir = _prepare_output_transaction(
        out_dir,
        protect=(*[p for p, _ in snapshots], dxf_path),
        results_root=results_root or DEF_OUT_DIR,
    )
    out_dir = Path(stage_dir)
    design = read_design(dxf_path)
    validate_equivalent_cells(design)
    template = design.cells[0]
    band_target = _band_targets(template)
    print(design.summary())

    # Register each snapshot INDEPENDENTLY (no assemble; positions are unknown / phase-only).
    tiles = []
    for i, (path, label) in enumerate(snapshots, start=1):
        scan = read_vk4(path)
        try:
            pls = register_sample(scan, template)
        except RegistrationAmbiguityError as e:
            print(f"WARNING: snapshot '{label}' ({path.name}) failed registration ({e}); skipping.")
            continue
        if not pls:
            print(f"WARNING: snapshot '{label}' ({path.name}) registered no cell; skipping.")
            continue
        pl = max(pls, key=lambda p: p.score)
        if len(pls) > 1:
            print(f"WARNING: snapshot '{label}' registered {len(pls)} cells; "
                  f"keeping the highest-score placement.")
        tiles.append(dict(label=label, scan=scan, placement=pl, snapshot_id=i, source=path.name))
        note = ("" if pl.absolute_origin
                else f"  <-- PHASE ONLY; absolute {pl.ambiguous_axes.upper()} index unresolved")
        print(f"  snapshot {i:>2} '{label:<10}' method={pl.method:<13} score={pl.score:.2f} "
              f"rot={pl.rotation_deg:+.2f}{note}")
    if not tiles:
        raise SystemExit("No snapshots could be registered against the DXF.")

    qc_dir = out_dir / "legacy" / "qc"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "legacy").mkdir(parents=True, exist_ok=True)
    _save_snapshot_provenance(out_dir / "figures", snapshots, dxf_path, passes, speed)

    # Measure every array of every snapshot. Each snapshot has its OWN scan, so the shared-scan
    # process fan-out runs per snapshot (results come back in submission order -> stable CSV).
    rows, results = [], []
    for tile in tiles:
        pl, scan, label = tile["placement"], tile["scan"], tile["label"]
        work, book = [], []
        for a in template.arrays:
            sample = ArraySample(
                filename=f"{label}_b{a.band}c{a.col}_D{a.diameter_um:g}",
                vk4_stem=label, cell_id=tile["snapshot_id"],
                array_id=a.array_id, band=a.band, col=a.col, passes=passes, speed=speed,
                nominal_diameter_um=a.diameter_um, target_diameter_um=band_target[a.band],
                nominal_pitch_um=a.pitch_um, nominal_pitch_x_um=a.pitch_x_um,
                nominal_pitch_y_um=a.pitch_y_um, nx=a.nx, ny=a.ny,
                cx_um=a.cx_um, cy_um=a.cy_um, cell_label=cell_label)
            work.append((pl, a, sample, qc_dir / (sample.filename + ".png"), make_qc))
            book.append(sample)
        extracted = parallel.pmap_shared(_extract_worker, work, scan, jobs=jobs)
        for sample, res in zip(book, extracted):
            reliable = (passes > 0
                        and not any(k in res.flags for k in ra.CRITICAL_FLAGS))
            row = ra.result_to_row(res, reliable)
            row["snapshot"], row["snapshot_id"] = label, tile["snapshot_id"]
            row["reg_score"], row["cell_label"] = pl.score, cell_label
            rows.append(row)
            results.append((sample, res, reliable))

    df = pd.DataFrame(rows).sort_values(["snapshot_id", "band", "col"])
    meas_csv = out_dir / "legacy" / "measurements.csv"
    df.to_csv(meas_csv, index=False)
    print(f"\nWrote {meas_csv} ({len(df)} rows across {len(tiles)} snapshot(s))")

    # Tiled figures under the SAME filenames as the single-scan pipeline (no new figure names):
    # the snapshots tiled side by side and annotated by name -- sample_heightmap.png = tiled height,
    # intensity_map.png = tiled intensity (identical layout). One montage build feeds both.
    montage = build_snapshot_montage(tiles, template)
    save_snapshot_heightmap(tiles, template, out_dir / "figures" / "sample_heightmap.png",
                            montage=montage)
    save_snapshot_overview(tiles, template, out_dir / "figures" / "intensity_map.png",
                           montage=montage)
    _plabel = f"Passes: {passes}\nSpeed: {speed:g} mm/s" if passes > 0 else ""
    write_3d_pin_maps(out_dir / "figures",
                      [(t["label"], t["scan"], t["placement"], _plabel) for t in tiles], template)

    # Average the snapshots (replicate views of one uniform cell) -> averaged cell report + radial
    # overlays. (No diameter_model: a single-geometry, single-dose snapshot has one point, which
    # cannot fit the process model Ø ~ drawn+passes+speed. measurements.csv keeps the per-tile rows.)
    by_array = {}
    for _sm, _res, _rel in results:
        by_array.setdefault(_sm.array_id, []).append(_res)
    _dose_tag = f"P{passes}_S{speed:g}" if passes > 0 else "averaged"
    avg_by_array = {aid: _average_results(rl, dose_label=_dose_tag) for aid, rl in by_array.items()}
    avg_by_array = {k: v for k, v in avg_by_array.items() if v is not None}
    _dose_label = _dose_tag if passes > 0 else "laser params: unset"   # P{passes}_S{speed} form
    cells_dir = out_dir / "figures" / "cells"; cells_dir.mkdir(parents=True, exist_ok=True)
    render_snapshot_cell_report(tiles, template, avg_by_array, _dose_label,
                                cells_dir / "cell_averaged.png")
    make_snapshot_radial_overlays(template, avg_by_array, out_dir, dose_label=_dose_tag)

    n_dose = df[["passes", "speed"]].drop_duplicates().shape[0]
    if n_dose < 2:
        print("Single laser dose across all snapshots -> skipping laser-parameter sweep plots "
              "(depth/Ø-vs-dose need >=2 doses); montage + per-snapshot geometry are the outputs.")
    _commit_output_transaction(out_dir, final_out_dir)
    return df, results, tiles


def analyze_multi_snapshot_dir(vk4_dir, out_dir, dxf_path, *, passes=0, speed=float("nan"),
                               cell_label="", make_qc=False, jobs=None, results_root=None):
    """Convenience wrapper: discover snapshots in ``vk4_dir`` (non-raster ``*.vk4``) and analyze.

    NOTE this treats EVERY non-raster ``*.vk4`` in the folder as a snapshot of ONE uniform cell. A
    flat folder holding several different wafer samples must NOT come through here -- see
    ``run_row.py``, which passes an explicit per-sample snapshot list instead."""
    snaps = snapshots_from_dir(vk4_dir)
    if not snaps:
        raise SystemExit(f"No disjoint snapshot .vk4 files found in {vk4_dir} "
                         f"(a '_Y*_X*' raster belongs to the tiled mode -> use analyze_sample).")
    return analyze_multi_snapshot(snaps, out_dir, dxf_path, passes=passes, speed=speed,
                                  cell_label=cell_label, make_qc=make_qc, jobs=jobs,
                                  results_root=results_root)


def main():
    # Suppress only the known-benign numpy noise from phase-correlation (divide/invalid on
    # all-zero windows), not every warning -- a blanket ignore would also hide real deprecation
    # and runtime warnings.
    warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
    warnings.filterwarnings("ignore", message=".*divide by zero encountered.*")
    argv = sys.argv[1:]
    if argv and argv[0] in ("--snapshots", "--multi"):
        # python run_sample.py --snapshots <vk4_dir> <out_dir> <dxf> <P{passes}_S{speed}>
        a = argv[1:]
        vk4_dir = Path(a[0]) if len(a) > 0 else DEF_VK4_DIR
        out_dir = Path(a[1]) if len(a) > 1 else DEF_OUT_DIR / "direct"
        dxf_path = Path(a[2]) if len(a) > 2 else next(DEF_DXF_DIR.glob("*.dxf"))
        passes, speed = _parse_ps_label(a[3]) if len(a) > 3 else (0, float("nan"))
        analyze_multi_snapshot_dir(vk4_dir, out_dir, dxf_path, passes=passes, speed=speed)
        return
    vk4_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEF_VK4_DIR
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEF_OUT_DIR / "direct"
    dxf_path = Path(sys.argv[3]) if len(sys.argv) > 3 else next(DEF_DXF_DIR.glob("*.dxf"))
    cell_csv = Path(sys.argv[4]) if len(sys.argv) > 4 else DEF_CSV_DIR / CELL_CSV_NAME
    analyze_sample(vk4_dir, out_dir, dxf_path, cell_csv)


if __name__ == "__main__":
    main()
