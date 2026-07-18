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

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
from dxf_geometry import read_design
from assemble import assemble_tiles
from register import register_sample
from extract import ArraySample, extract_array
from laser_params import load_cell_params, write_cell_template, CELL_CSV_NAME
import run_analysis as ra

HERE = Path(__file__).parent
DEF_DXF_DIR = HERE / "DXF"
DEF_VK4_DIR = HERE / "VK4"
DEF_CSV_DIR = HERE / "CSV"
DEF_OUT_DIR = HERE / "Results"


def _band_targets(template):
    out = {}
    for b in sorted(set(a.band for a in template.arrays)):
        out[b] = float(np.median([a.diameter_um for a in template.arrays if a.band == b]))
    return out


def analyze_sample(vk4_dir, out_dir, dxf_path, cell_csv, *, make_qc=False):
    design = read_design(dxf_path)
    template = design.cells[0]
    band_target = _band_targets(template)
    print(design.summary())

    scan = assemble_tiles(vk4_dir)
    placements = register_sample(scan, template)
    if not placements:
        raise SystemExit("No unit cells could be registered in the assembled sample.")
    nrow = max(p.cell_row for p in placements)
    ncol = max(p.cell_col for p in placements)
    print(f"\nRegistered {len(placements)} unit cells in a {nrow}x{ncol} grid "
          f"(design frame, (1,1)=top-left):")
    for p in placements:
        print(f"  cell ({p.cell_row},{p.cell_col}): origin=({p.origin_col:.0f},"
              f"{p.origin_row:.0f}) rot={p.rotation_deg:+.2f}deg reg={p.score:.2f}"
              + ("  <-- low reg, check QC" if p.score < 0.5 else ""))

    params = load_cell_params(cell_csv)
    if params:
        print(f"\nLoaded laser params for {len(params)} cells from {cell_csv}")
    else:
        print(f"\nNo cell params at {cell_csv} (geometry still measured; fill the template).")

    out_dir = Path(out_dir)
    qc_dir = out_dir / "qc"; qc_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    rows, results = [], []
    res_by_cell = {}
    for pl in placements:
        pr = params.get((pl.cell_row, pl.cell_col))
        passes = pr.passes if pr else 0
        speed = pr.speed if pr else float("nan")
        label = pr.label if pr else ""
        reliable_cell = pl.score >= 0.5
        res_by_cell[pl.cell_id] = {}
        for a in template.arrays:
            sample = ArraySample(
                filename=f"r{pl.cell_row}c{pl.cell_col}_b{a.band}c{a.col}_D{a.diameter_um:g}",
                vk4_stem=f"r{pl.cell_row}c{pl.cell_col}", cell_id=pl.cell_id,
                array_id=a.array_id, band=a.band, col=a.col, passes=passes, speed=speed,
                nominal_diameter_um=a.diameter_um, target_diameter_um=band_target[a.band],
                nominal_pitch_um=a.pitch_um, nominal_pitch_x_um=a.pitch_x_um,
                nominal_pitch_y_um=a.pitch_y_um, nx=a.nx, ny=a.ny,
                cx_um=a.cx_um, cy_um=a.cy_um, cell_label=label)
            res = extract_array(scan, pl, a, sample, make_qc=make_qc,
                                qc_path=qc_dir / (sample.filename + ".png"))
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
    df.to_csv(out_dir / "measurements.csv", index=False)
    print(f"\nWrote {out_dir/'measurements.csv'} ({len(df)} rows)")

    missing = [(p.cell_row, p.cell_col) for p in placements
               if (p.cell_row, p.cell_col) not in params]
    if missing:                                          # only when the CSV doesn't cover them
        tmpl_csv = write_cell_template(Path(cell_csv).parent,
                                       [(p.cell_row, p.cell_col) for p in placements],
                                       overwrite=True)
        print(f"Wrote cell-params template ({len(missing)} cells need params) -> {tmpl_csv}")

    # per-unit-cell report figures
    cells_dir = out_dir / "cells"; cells_dir.mkdir(parents=True, exist_ok=True)
    for pl in placements:
        render_cell_report(scan, pl, template, res_by_cell[pl.cell_id],
                           params.get((pl.cell_row, pl.cell_col)),
                           cells_dir / f"cell_r{pl.cell_row}_c{pl.cell_col}.png")
    print(f"Wrote {len(placements)} per-cell reports -> {cells_dir}")

    save_sample_overview(scan, template, placements, out_dir / "figures" / "cell_overview.png")
    save_sample_heightmap(scan, out_dir / "figures" / "sample_heightmap.png")
    make_param_summary(df, out_dir)
    ra.make_plots(df, results, out_dir)
    ra.print_diameter_calibration(df, out_dir)
    return df, results, placements


# ---------------------------------------------------- per-unit-cell report #
def _resample_cell(field, placement, template, valid, res_um=2.0):
    """Resample a cell into DESIGN orientation (un-mirrored, y up), sampling ``field`` at each
    design pixel via the registration transform. Returns (image, extent_um)."""
    cw, ch = template.size_um
    nx, ny = int(round(cw / res_um)), int(round(ch / res_um))
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * res_um, (np.arange(ny) + 0.5) * res_um)
    cols, rows = placement.dxf_to_px(gx.ravel(), gy.ravel())
    ci = np.round(cols).astype(int); ri = np.round(rows).astype(int)
    H, W = field.shape
    m = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    out = np.full(gx.size, np.nan)
    idx = np.nonzero(m)[0]
    vv = valid[ri[m], ci[m]]
    out[idx[vv]] = field[ri[m][vv], ci[m][vv]]
    return out.reshape(ny, nx), (0.0, cw, 0.0, ch)


def _overlay_design(ax, template, res_by_array):
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
    ax.set_xlim(-120, template.size_um[0] + 20)
    ax.set_ylim(-120, template.size_um[1] + 60)
    ax.set_xlabel("x (µm, design)"); ax.set_ylabel("y (µm, design)")


def render_cell_report(scan, placement, template, res_by_array, params, path):
    """One report per unit cell: height (floor=0) + intensity with pin/marker/array overlays,
    a title with laser params + cell position, and a measured-vs-expected table per array."""
    valid = scan.height_raw != 0
    z_cell, ext = _resample_cell(scan.height_um, placement, template, valid)
    floor = np.nanpercentile(z_cell, 20) if np.isfinite(z_cell).any() else 0.0
    zdisp = z_cell - floor                                # local zero = trench floor
    inten = (_resample_cell(scan.intensity.astype(float), placement, template, valid)[0]
             if scan.intensity is not None else None)

    fig = plt.figure(figsize=(21, 12))
    gs = GridSpec(2, 2, width_ratios=[1.3, 1.0], figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)   # aligned x/y with height panel

    vmax = np.nanpercentile(zdisp, 98) if np.isfinite(zdisp).any() else 1.0
    im = ax0.imshow(zdisp, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax,
                    aspect="equal")
    _overlay_design(ax0, template, res_by_array)
    plt.colorbar(im, ax=ax0, shrink=0.85, label="height above floor (µm)")
    ax0.set_title("height (floor = local zero)")
    if inten is not None:
        im1 = ax1.imshow(inten, origin="lower", extent=ext, cmap="gray", aspect="equal",
                         vmin=np.nanpercentile(inten, 2), vmax=np.nanpercentile(inten, 98))
        plt.colorbar(im1, ax=ax1, shrink=0.85, label="intensity")   # keeps x-axes aligned
    _overlay_design(ax1, template, res_by_array)
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
                       f"{r.pitch_um:.0f}/{a.pitch_um:g}", f"{100*r.debris_fraction:.0f}"])
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
    d = df[(df["passes"] > 0) & (df["speed"] > 0) & df["depth_um"].notna()].copy()
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
    Wd = base.shape[1]
    allc = template.all_centers_um()
    fig, ax = plt.subplots(figsize=(12, 13))
    ax.imshow(base, origin="lower", cmap="gray",
              vmin=np.nanpercentile(base, 2), vmax=np.nanpercentile(base, 98))
    for p in placements:
        cols, rows = p.dxf_to_px(allc[:, 0], allc[:, 1])
        fx = (W - 1 - cols) / ds                          # apply the same L-R flip
        fy = rows / ds
        ax.plot(fx, fy, "c.", ms=0.5)
        cx, cy = p.dxf_to_px(template.size_um[0] / 2, template.size_um[1] / 2)
        ax.text((W - 1 - cx) / ds, cy / ds, f"({p.cell_row},{p.cell_col})",
                color="yellow", ha="center", va="center", fontsize=11, weight="bold")
    ax.set_title(f"{len(placements)} unit cells — design (row,col), design orientation")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170); plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    vk4_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEF_VK4_DIR
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEF_OUT_DIR
    dxf_path = Path(sys.argv[3]) if len(sys.argv) > 3 else next(DEF_DXF_DIR.glob("*.dxf"))
    cell_csv = Path(sys.argv[4]) if len(sys.argv) > 4 else DEF_CSV_DIR / CELL_CSV_NAME
    analyze_sample(vk4_dir, out_dir, dxf_path, cell_csv)


if __name__ == "__main__":
    main()
