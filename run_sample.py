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

sys.path.insert(0, str(Path(__file__).parent))
from dxf_geometry import read_design, validate_equivalent_cells
from assemble import assemble_tiles
from register import RegistrationAmbiguityError, register_sample
from extract import ArraySample, extract_array
from laser_params import load_cell_params, CELL_CSV_NAME
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


def _prepare_output_transaction(out_dir, protect=(), results_root=DEF_OUT_DIR):
    """Create an owned hidden staging sibling while leaving the last good result untouched."""
    final_dir = _validate_output_target(out_dir, protect=protect, results_root=results_root)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    backup = final_dir.parent / f".{final_dir.name}.previous"
    if backup.exists():
        if final_dir.exists():
            if not (_sentinel_valid(backup, final_dir, ("complete",))
                    or _looks_like_legacy_output(backup)):
                raise SystemExit(f"refusing to remove unowned recovery directory {backup}")
            shutil.rmtree(backup)
        else:
            os.replace(backup, final_dir)
            print(f"Recovered previous completed output after an interrupted swap -> {final_dir}")
    for stale in final_dir.parent.glob(f".{final_dir.name}.staging-*"):
        if (stale.is_dir()
                and _sentinel_valid(stale, final_dir, ("staging",))):
            shutil.rmtree(stale)
    stage = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.staging-", dir=final_dir.parent))
    _write_results_sentinel(stage, final_dir, "staging")
    return stage, final_dir


def _commit_output_transaction(stage, final_dir):
    """Validate and swap a staged run into place, restoring the previous run on swap failure."""
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
    backup = final_dir.parent / f".{final_dir.name}.previous"
    if backup.exists():
        raise RuntimeError(f"recovery directory unexpectedly exists: {backup}")
    if final_dir.exists():
        os.replace(final_dir, backup)
    try:
        os.replace(stage, final_dir)
    except BaseException:
        if backup.exists() and not final_dir.exists():
            os.replace(backup, final_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)
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

    # Clean outputs live in figures/ (cell_overview, sample_heightmap, param_depth_scatter,
    # radial_overlays/, cells/, diameter_calibration.txt). Legacy outputs (v1 figure set,
    # measurements.csv, qc/) are regenerated under legacy/ ahead of the planned refactor.
    save_sample_overview(scan, template, placements, out_dir / "figures" / "cell_overview.png")
    save_sample_heightmap(scan, out_dir / "figures" / "sample_heightmap.png")
    make_param_summary(df, out_dir)          # per-cell median depth & Ø-oversizing vs params
    make_param_depth_scatter(df, out_dir)    # same, with every array scattered around the median
    make_radial_overlays(template, placements, params, res_by_cell, out_dir,
                         sets_csv=Path(cell_csv).parent / RADIAL_CSV_NAME)
    ra.print_diameter_calibration(df, out_dir / "figures")
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


def _overlay_design(ax, template, res_by_array, box=None):
    """Red pin + alignment-marker outlines, dashed array dividers, per-array D/P + discrepancy."""
    ax.add_patch(Rectangle((0, 0), template.marker_size_um, template.marker_size_um,
                           ec="red", fill=False, lw=1.6))                # alignment marker
    for a in template.arrays:
        ax.add_patch(Rectangle((a.x0_um - 18, a.y0_um - 18), a.width_um + 36, a.height_um + 36,
                               ec="red", ls="--", fill=False, lw=0.8))   # array divider
        for (x, y) in a.centers_um:
            ax.add_patch(Circle((x, y), a.diameter_um / 2, ec="red", fill=False, lw=0.35))
        r = res_by_array.get(a.array_id)
        txt = f"D{a.diameter_um:g} P{a.pitch_um:g}"
        if r is not None and np.isfinite(r.diameter_um) and a.diameter_um:
            txt += f" {100*(r.diameter_um-a.diameter_um)/a.diameter_um:+.0f}%"
        ax.text(a.x0_um - 10, a.y1_um + 26, txt, color="red", fontsize=6.5,
                va="bottom", ha="left")
    x0, x1, y0, y1 = box if box is not None else _cell_content_box(template)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("x (µm, design)"); ax.set_ylabel("y (µm, design)")


def render_cell_report(scan, placement, template, res_by_array, params, path):
    """One report per unit cell: height (floor=0) + intensity with pin/marker/array overlays,
    a title with laser params + cell position, and a measured-vs-expected table per array."""
    valid = scan.height_raw != 0
    box = _cell_content_box(template)
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
    plt.colorbar(im, ax=ax0, shrink=0.85, label="height above floor (µm)")
    ax0.set_title("height (floor = local zero)")
    if inten is not None:
        im1 = ax1.imshow(inten, origin="lower", extent=ext, cmap="gray", aspect="equal",
                         vmin=np.nanpercentile(inten, 2), vmax=np.nanpercentile(inten, 98))
        plt.colorbar(im1, ax=ax1, shrink=0.85, label="intensity")   # keeps x-axes aligned
    _overlay_design(ax1, template, res_by_array, box=box)
    ax1.set_title("intensity — alignment marker (red square, bottom-left)")

    axr = fig.add_subplot(gs[:, 1]); axr.axis("off")

    cx = placement.origin_col * scan.x_um_per_px / 1000
    cy = placement.origin_row * scan.y_um_per_px / 1000
    lp = (f"P{params.passes} S{params.speed:g}" if params and params.valid
          else "laser params: (fill CSV/cell_params.csv)")
    title = (f"Unit cell (row {placement.cell_row}, col {placement.cell_col})   —   {lp}\n"
             f"cell marker in sample: x = {cx:.2f} mm, y = {cy:.2f} mm   ·   "
             f"registration {placement.score:.2f}")
    axr.text(0.0, 1.0, title, transform=axr.transAxes, va="top", fontsize=13, weight="bold")

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
    col_labels = ["array (D drawn / P)", "depth µm", "Ø top", "Ø nom", "Ø floor",
                  "pitch m/e", "debris%"]
    tbl = axr.table(cellText=rows_t, colLabels=col_labels, cellLoc="center",
                    bbox=[0.0, 0.02, 1.0, 0.80])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for c in range(len(col_labels)):
        tbl[0, c].set_facecolor("#dddddd"); tbl[0, c].set_text_props(weight="bold")
    axr.text(0.0, 0.86, "Ø floor is '—' where redeposition debris buries the pin base "
             "(see debris%); Ø top/nom are the reliable measures.",
             transform=axr.transAxes, va="top", fontsize=8, style="italic", color="0.3")

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


# ------------------------------------------------- laser-parameter summary #
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
    cmap = plt.get_cmap("viridis")
    colors = {p: cmap(i / max(1, len(passes_vals) - 1)) for i, p in enumerate(passes_vals)}
    speeds = sorted(g["speed"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(11, 11), sharex=True)
    for metric, ax, ylab, ref in [
            ("depth", axes[0], "median etch depth (µm)", (55, "design target 55 µm")),
            ("overs", axes[1], "mid-Ø oversizing  (meas − drawn)/drawn  (%)", (0, None))]:
        for pas in passes_vals:
            gg = g[g["passes"] == pas].sort_values("speed")
            ax.plot(gg["speed"], gg[metric], "o-", color=colors[pas], ms=10, mew=1.2,
                    label=f"{pas} passes")
            for _, r in gg.iterrows():
                ax.annotate(r["label"], (r["speed"], r[metric]), fontsize=7,
                            textcoords="offset points", xytext=(6, 6), color="0.25")
        if ref[0] is not None:
            ax.axhline(ref[0], color="grey", ls="--", lw=1)
            if ref[1]:
                ax.text(0.01, ref[0], " " + ref[1], transform=ax.get_yaxis_transform(),
                        va="bottom", color="grey", fontsize=8)
        ax.set_ylabel(ylab); ax.set_xscale("log"); ax.grid(alpha=0.3)
    axes[0].set_xticks(speeds); axes[0].set_xticklabels([f"{s:g}" for s in speeds])
    axes[1].set_xlabel("scan speed (mm/s, log axis)")
    axes[0].legend(title="passes", fontsize=9)
    axes[0].set_title("Laser-parameter effect on depth and diameter "
                      "(each point = one unit cell, labelled P{passes}_S{speed})")
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
    cmap = plt.get_cmap("viridis")
    colors = {p: cmap(i / max(1, len(passes_vals) - 1)) for i, p in enumerate(passes_vals)}
    speeds = sorted(d["speed"].unique())

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
            ax.plot(x, sub["depth_um"], marker=marker, ls="none", ms=5.5, mew=0.4,
                    mfc=colors[pas], mec="white", alpha=0.55, zorder=2)

    # per-passes line through the cell medians, drawn on top of its scatter
    for pas in passes_vals:
        gg = med[med["passes"] == pas].sort_values("speed")
        ax.plot(gg["speed"], gg["depth"], "o-", color=colors[pas], ms=10, mew=1.2,
                mec="0.15", label=f"{pas} passes", zorder=3)
        for _, r in gg.iterrows():
            ax.annotate(r["label"], (r["speed"], r["depth"]), fontsize=7, ha="center",
                        textcoords="offset points", xytext=(0, 10), color="0.25", zorder=4)

    ax.axhline(55, color="grey", ls="--", lw=1)
    ax.text(0.01, 55, " design target 55 µm", transform=ax.get_yaxis_transform(),
            va="bottom", color="grey", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(speeds); ax.set_xticklabels([f"{s:g}" for s in speeds])
    ax.set_xlabel("scan speed (mm/s, log axis)")
    ax.set_ylabel("etch depth (µm)")
    ax.grid(alpha=0.3)
    ax.set_title("Etch depth of every pin array vs laser parameters\n"
                 "(o-line = per-cell median; small markers = individual arrays, "
                 "◇ D300  □ D100  △ D50)")

    passes_leg = ax.legend(title="passes", fontsize=9, loc="upper right")
    ax.add_artist(passes_leg)
    shape_handles = []
    for fam, (marker, _) in fam_style.items():
        if not (d["family"] == fam).any():
            continue
        lo, hi = d.loc[d["family"] == fam, "drawn_diameter_um"].agg(["min", "max"])
        shape_handles.append(Line2D([], [], marker=marker, ls="none", mfc="0.4", mec="white",
                                    ms=8, label=f"{fam} array (drawn Ø {lo:g}–{hi:g} µm)"))
    shape_handles.append(Line2D([], [], marker="o", ls="-", color="0.4", ms=9,
                                label="cell median"))
    ax.legend(handles=shape_handles, fontsize=8, loc="lower left", framealpha=0.9)

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
                ax.plot(rc, z, "-", lw=1.9, color=_combo_color(i, len(combos)),
                        label=f"{combo}   (depth {depth:.0f} µm)")
                n_lines += 1
            if not n_lines:
                plt.close(fig)
                continue
            ax.axhline(0, color="grey", lw=0.8, ls="-")
            ax.axvline(a.diameter_um / 2, color="green", ls=":", lw=1.2,
                       label=f"drawn radius {a.diameter_um/2:g} µm")
            ax.set_xlabel("radius from pin centre (µm)")
            ax.set_ylabel("mean height above clean floor (µm)")
            ax.set_title(f"Radial-average pin profile — drawn Ø {a.diameter_um:g} µm, "
                         f"pitch {a.pitch_um:g} µm\nband {a.band} col {a.col}  ·  {set_name}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2 if n_lines > 8 else 1)
            fig.tight_layout()
            fname = f"a{a.array_id:02d}_D{a.diameter_um:g}_P{a.pitch_um:g}.png"
            fig.savefig(sdir / fname, dpi=170); plt.close(fig)
            n_fig += 1
    print(f"Wrote {n_fig} radial-overlay figures -> {root}")


# ------------------------------------------------------------- presentation #
def _flip_lr(a):
    return a[:, ::-1]


def save_sample_heightmap(scan, path, ds=4):
    """Full-sample height map, flipped left-right to DESIGN orientation for presentation."""
    valid = scan.height_raw != 0
    z = np.where(valid, scan.height_um, np.nan)[::ds, ::ds]
    z = _flip_lr(z)                                       # un-mirror for presentation
    fig, ax = plt.subplots(figsize=(11, 12))
    im = ax.imshow(z, origin="lower", cmap="viridis",
                   vmin=np.nanpercentile(z, 2), vmax=np.nanpercentile(z, 98))
    ax.set_title("Assembled sample height (design orientation)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.85, label="height (µm)")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200); plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(12, 13))
    ax.imshow(base, origin="lower", cmap="gray",
              vmin=np.nanpercentile(base, 2), vmax=np.nanpercentile(base, 98))
    for p in placements:
        cols, rows = p.dxf_to_px(allc[:, 0], allc[:, 1])
        fx = (W - 1 - cols) / ds                          # apply the same L-R flip
        fy = rows / ds
        ax.plot(fx, fy, "c.", ms=0.5)
        cx, cy = p.dxf_to_px(ccx, ccy)
        ax.text((W - 1 - cx) / ds, cy / ds, f"({p.cell_row},{p.cell_col})",
                color="yellow", ha="center", va="center", fontsize=11, weight="bold")
    ax.set_title(f"{len(placements)} unit cells — design (row,col), design orientation")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170); plt.close(fig)


def main():
    # Suppress only the known-benign numpy noise from phase-correlation (divide/invalid on
    # all-zero windows), not every warning -- a blanket ignore would also hide real deprecation
    # and runtime warnings.
    warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
    warnings.filterwarnings("ignore", message=".*divide by zero encountered.*")
    vk4_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEF_VK4_DIR
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEF_OUT_DIR / "direct"
    dxf_path = Path(sys.argv[3]) if len(sys.argv) > 3 else next(DEF_DXF_DIR.glob("*.dxf"))
    cell_csv = Path(sys.argv[4]) if len(sys.argv) > 4 else DEF_CSV_DIR / CELL_CSV_NAME
    analyze_sample(vk4_dir, out_dir, dxf_path, cell_csv)


if __name__ == "__main__":
    main()
