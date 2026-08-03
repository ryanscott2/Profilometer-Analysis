"""
Row-level rollup: pool a whole wafer row's per-sample results into one table and one figure set.

Each sample of a row is a single geometry at a single dose, so its own report has nothing to
compare against -- ``analyze_multi_snapshot`` explicitly skips the dose-sweep plots when a dataset
carries fewer than two doses. The ROW is the first place a comparison exists: within one row the
lattice is constant, the geometry takes three values paired by wafer column, and each geometry
carries two doses.

That also bounds what may honestly be drawn. With two points per geometry and a constant scan
speed, the dose axis collapses to *passes* and no model can be fitted: ``report.make_diameter_model``
would decline on its own degrees-of-freedom guard, and the saturating depth models in
``calibrate_depth`` need a sweep. So these figures connect measured points and label the segment
slope; they never draw a regression line. Pool several rows with ``calibrate_depth.py`` for that.

Everything here is written to the ROW CONTAINER under ``row_*`` names. The container must never
gain a ``figures/`` directory or a ``legacy/measurements.csv``: ``run_sample`` recognises exactly
that pair as an output dataset, and a transaction committed on the container would delete every
per-sample result inside it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs
import report
import extract
import porosity_table as pt
import run_sample as rs

#: identity block prepended to every rollup row -- who this measurement belongs to
IDENT_COLS = ["wafer_row", "wafer_col", "sample", "geometry", "lattice", "laser", "laser_raw",
              "map_passes", "map_speed", "dxf", "status", "n_snapshots"]
#: columns analyze_multi_snapshot adds on top of report.result_to_row
MODE_EXTRAS = ["snapshot", "snapshot_id", "cell_label"]

TARGET_DEPTH_UM = 55.0
#: assigned by ascending drawn diameter so the same geometry keeps its colour across figures
ROW_GEOM_COLORS = ("#4E79A7", "#E8A33D", "#7B4FA0")
MISSING_COLOR = "#B22222"                                  # firebrick, the project's failure colour


def canonical_columns():
    """The measurement columns, in ``report.result_to_row`` order, computed at RUNTIME.

    Reading the order off a live empty result rather than hard-coding it means a new
    ``PinFinResult`` field cannot silently reorder or drop a column from the rollup."""
    return list(report.result_to_row(extract.PinFinResult(filename=""), False))


# ============================================================== rollup assembly == #
def _identity(record, row, date_tag):
    e = record.planned.entry
    return {
        "wafer_row": row, "wafer_col": e.col, "sample": record.planned.out_name,
        "geometry": e.geometry, "lattice": e.lattice,
        "laser": e.laser, "laser_raw": e.laser_raw,
        "map_passes": e.passes, "map_speed": e.speed,
        "dxf": Path(record.planned.dxf).name if record.planned.dxf else "",
        "status": record.status, "n_snapshots": record.n_registered,
    }


def _placeholder_row(record, row, date_tag, columns):
    """One all-NaN measurement row for a sample that produced no data.

    A missing design point must never be a MISSING ROW: the row records that the point was
    attempted and why it is empty. It is safe downstream by construction -- ``report._fit_subset``
    and ``calibrate_depth.apply_gates`` both drop ``reliable=False`` and non-finite depths."""
    e = record.planned.entry
    out = {c: np.nan for c in columns}
    out.update(_identity(record, row, date_tag))
    out.update({
        "passes": e.passes, "speed": e.speed, "dose_ratio": np.nan,
        "nominal_diameter_um": e.nominal_d_um, "drawn_diameter_um": np.nan,
        "nominal_pitch_um": e.nominal_p_um,
        "band": 1, "col": 1, "array_id": 1, "cell_id": 1,
        "snapshot": "", "snapshot_id": 0, "cell_label": e.laser,
        "reliable": False, "flags": "", "filename": record.planned.out_name,
        "n_snapshots": 0,
    })
    return out


def build_rollup(records, *, row, date_tag=""):
    """Concatenate every sample's measurements into one row-level table. PURE over ``records``.

    Column order is fixed explicitly (``IDENT_COLS`` + canonical + mode extras + any leftover):
    ``pd.concat`` otherwise takes its order from whichever frame happens to be first."""
    canon = canonical_columns()
    base = IDENT_COLS + canon + MODE_EXTRAS
    frames, leftovers = [], []
    for rec in records:
        if rec.produced_data:
            df = rec.df.copy()
            for k, v in _identity(rec, row, date_tag).items():
                df[k] = v
            leftovers += [c for c in df.columns if c not in base]
            frames.append(df)
        else:
            frames.append(pd.DataFrame([_placeholder_row(rec, row, date_tag, canon + MODE_EXTRAS)]))
    if not frames:
        return pd.DataFrame(columns=base)

    extra = sorted(set(leftovers))
    if extra:
        print(f"row rollup: keeping {len(extra)} unrecognised column(s) from the per-sample CSVs: "
              f"{extra}")
    out = pd.concat(frames, ignore_index=True, sort=False).reindex(columns=base + extra)

    # dtype guards, each from a real failure mode:
    #  * NaN.astype(bool) is True -- that would let unvetted rows through every reliability gate;
    #  * 'flags' is float64 when every value is empty, and .str.contains then raises.
    out["reliable"] = out["reliable"].map(
        lambda v: False if (v is None or (isinstance(v, float) and math.isnan(v)))
        else bool(v) if isinstance(v, (bool, np.bool_)) else str(v).strip().lower()
        in ("1", "true", "yes", "y", "t"))
    out["flags"] = out["flags"].fillna("").astype(str)
    out = out.sort_values(["wafer_col", "snapshot_id", "band", "col"],
                          kind="mergesort").reset_index(drop=True)
    dup = out[out["snapshot_id"] > 0].duplicated(subset=["sample", "snapshot", "array_id"])
    if bool(dup.any()):
        print(f"row rollup: WARNING {int(dup.sum())} duplicated (sample, snapshot, array_id) row(s)")
    return out


def _phi(d_um, p_um, lattice):
    """Open-area fraction for this lattice. ``porosity_table.porosity`` returns (square, hex)."""
    if not (np.isfinite(d_um) and np.isfinite(p_um)) or d_um <= 0 or p_um <= 0:
        return np.nan
    sq, hexa = pt.porosity(float(d_um), float(p_um))
    return hexa if lattice == "hex" else sq


def build_units(rollup):
    """One row per SAMPLE: medians over its snapshots, plus design and achieved open-area fraction.

    This is the natural unit for a cross-row comparison later (hex vs square at matched geometry
    and dose), which is why it is written out alongside the raw measurement rows."""
    med = ["depth_um", "diameter_um", "top_diameter_um", "base_diameter_um", "taper_um",
           "disc_mid_um", "disc_top_um", "disc_base_um", "debris_fraction", "reg_score",
           "drawn_diameter_um", "nominal_pitch_um", "nominal_diameter_um", "floor_flatness_um",
           "coverage", "dose_ratio"]
    keys = ["wafer_row", "wafer_col", "sample", "geometry", "lattice", "laser", "laser_raw",
            "map_passes", "map_speed", "dxf", "status"]
    rows = []
    for key, g in rollup.groupby(keys, dropna=False, sort=True):
        rec = dict(zip(keys, key))
        good = g[g["reliable"]] if bool(g["reliable"].any()) else g
        for c in med:
            rec[c] = float(pd.to_numeric(good.get(c), errors="coerce").median()) \
                if c in good else np.nan
        depth = pd.to_numeric(good.get("depth_um"), errors="coerce").dropna()
        rec["depth_spread_um"] = float(depth.max() - depth.min()) if len(depth) > 1 else 0.0
        rec["n_rows"] = int(len(g))
        rec["n_reliable"] = int(g["reliable"].sum())
        rec["n_snapshots"] = int(g.loc[g["snapshot_id"] > 0, "snapshot_id"].nunique())
        rec["flags"] = "; ".join(sorted({f for s in g["flags"] for f in str(s).split(";")
                                         if f.strip()}))
        lat = rec.get("lattice", "")
        pitch = rec.get("nominal_pitch_um", np.nan)
        # A sample that failed to register has no DRAWN diameter (nothing parsed the DXF for it),
        # but its design point is still known from the wafer map. Fall back to the nominal so the
        # design reference line exists for every column -- a missing sample must leave a visible
        # gap against its target, not collapse the axis to zero.
        d_design = rec.get("drawn_diameter_um", np.nan)
        if not np.isfinite(d_design):
            d_design = rec.get("nominal_diameter_um", np.nan)
        rec["phi_design"] = _phi(d_design, pitch, lat)
        rec["phi_achieved"] = _phi(rec.get("diameter_um", np.nan), pitch, lat)
        rec["phi_top"] = _phi(rec.get("top_diameter_um", np.nan), pitch, lat)
        rows.append(rec)
    units = pd.DataFrame(rows)
    if not units.empty:
        units = units.sort_values("wafer_col", kind="mergesort").reset_index(drop=True)
    return units


# ==================================================================== montage == #
def capture_row_panel(tiles, template, *, res_um=8.0):
    """A coarse, floor-referenced, design-oriented height panel for one sample's best snapshot.

    Captured DURING the run and kept as a small ``float32`` array (sub-MB at 8 µm/px) so the VK4
    scans can be released immediately afterwards -- holding the ``tiles`` for a whole row would
    retain hundreds of MB. Returns ``None`` when nothing is in frame."""
    if not tiles:
        return None
    best = max(tiles, key=lambda t: getattr(t["placement"], "score", 0.0))
    z, _inten, _ext = rs._snapshot_panel(best["scan"], best["placement"], template, res_um=res_um)
    if z is None:
        return None
    return np.asarray(z, dtype=np.float32)


# ==================================================================== figures == #
def _geometry_colors(df):
    """Geometry -> colour, ordered by ascending drawn diameter so the mapping is stable and the
    legend reads small-to-large."""
    order = (df.groupby("geometry")["nominal_diameter_um"].median()
             .sort_values(kind="mergesort").index.tolist())
    return {g: ROW_GEOM_COLORS[i % len(ROW_GEOM_COLORS)] for i, g in enumerate(order)}


def _column_axis(ax, units, colors, *, label_blocks=True, block_y=1.02):
    """Lay the wafer columns out along x, separated and labelled by geometry block.

    Shared by the diameter and porosity figures: one x slot per wafer column keeps every sample
    legible (annotating points in data space collides as soon as two doses are close), and the
    blocks make the column->geometry pairing -- and its reversal between wafer rows -- visible."""
    u = units.sort_values("wafer_col")
    xs = list(range(len(u)))
    geoms = list(u["geometry"])
    for i in range(1, len(geoms)):
        if geoms[i] != geoms[i - 1]:
            ax.axvline(i - 0.5, color="0.85", lw=1.0, zorder=0)
    if label_blocks:
        start = 0
        for i in range(1, len(geoms) + 1):
            if i == len(geoms) or geoms[i] != geoms[start]:
                ax.text(0.5 * (start + i - 1), block_y, geoms[start],
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=fs.NOTE, weight="bold", color=colors.get(geoms[start], "0.3"))
                start = i
    ax.set_xlim(-0.6, len(u) - 0.4)
    ax.grid(alpha=0.3, axis="y")
    return u, xs


def _column_ticks(ax, units):
    u = units.sort_values("wafer_col")
    ax.set_xticks(list(range(len(u))))
    ax.set_xticklabels([f"c{int(r['wafer_col'])}\n{r['laser']}" for _, r in u.iterrows()])
    ax.set_xlabel("wafer column")


def _finish(fig, path, *, dpi=200, transparent=False, facecolor=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    kw = {"dpi": dpi, "bbox_inches": "tight"}
    if transparent:
        kw["transparent"] = True
    elif facecolor:
        kw["facecolor"] = facecolor
    fig.savefig(path, **kw)
    plt.close(fig)
    print(f"Wrote {path}")
    return Path(path)


def _fig_depth_vs_passes(units, rollup, path, *, transparent=False):
    """R1 — how much depth does each extra pass buy, and does the rate depend on geometry?"""
    colors = _geometry_colors(units)
    single_speed = units["map_speed"].nunique(dropna=True) <= 1
    speed = float(units["map_speed"].dropna().iloc[0]) if single_speed and \
        len(units["map_speed"].dropna()) else float("nan")
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    handles = []
    for geom, g in units.groupby("geometry", sort=False):
        g = g.sort_values("map_passes")
        c = colors[geom]
        ok = g[np.isfinite(g["depth_um"])]
        miss = g[~np.isfinite(g["depth_um"])]
        x = ok["map_passes"] if single_speed else ok["dose_ratio"]
        ax.plot(x, ok["depth_um"], "-", color=c, lw=1.6, zorder=2)
        ax.plot(x, ok["depth_um"], "o", color=c, ms=9, zorder=3)
        # the individual snapshot values behind each sample point, so spread is visible
        for k, (_, r) in enumerate(ok.iterrows()):
            sub = rollup[(rollup["sample"] == r["sample"]) & rollup["reliable"]]
            xv = r["map_passes"] if single_speed else r["dose_ratio"]
            ax.plot([xv] * len(sub), pd.to_numeric(sub["depth_um"], errors="coerce"),
                    "o", mfc="none", mec=c, ms=5, alpha=0.55, zorder=1)
            # stagger above/below: adjacent geometries can share nearly the same pass count
            # (this row has 16 and 17), and centred labels then overprint each other
            dy = 12 if (int(r["wafer_col"]) % 2) else -20
            ax.annotate(f"c{int(r['wafer_col'])} · {r['laser']}", (xv, r["depth_um"]),
                        textcoords="offset points", xytext=(0, dy), ha="center",
                        fontsize=fs.ANNOT, color="0.25")
        lab = geom
        # Slope per PASS is only meaningful when speed is constant; on a mixed-speed row the x axis
        # is dose ratio, and a "µm / pass" label anchored in pass coordinates would sit in the wrong
        # place AND mean the wrong thing.
        if len(ok) == 2 and single_speed:                   # 2 points -> a segment, never a fit
            (x0, y0), (x1, y1) = ok[["map_passes", "depth_um"]].values
            if x1 != x0:
                ax.annotate(f"{(y1 - y0) / (x1 - x0):+.1f} µm / pass",
                            (0.5 * (x0 + x1), 0.5 * (y0 + y1)),
                            textcoords="offset points", xytext=(6, -14),
                            fontsize=fs.ANNOT, color=c, weight="bold")
        if len(miss):
            lab += f" — {len(ok)} of {len(g)} doses"
            for _, r in miss.iterrows():
                xv = r["map_passes"] if single_speed else r["dose_ratio"]
                if not np.isfinite(xv):
                    continue
                ax.axvline(xv, color="0.7", ls=":", lw=1.0, zorder=0)
                # place the callout in AXES fraction, not the not-yet-autoscaled data limits
                ax.text(xv, 0.02, f"  c{int(r['wafer_col'])} — no data",
                        transform=ax.get_xaxis_transform(),
                        rotation=90, va="bottom", ha="right", fontsize=fs.ANNOT_SM,
                        color=MISSING_COLOR)
        handles.append(Line2D([], [], color=c, marker="o", ls="-", label=lab))
    ax.axhline(TARGET_DEPTH_UM, color="0.45", ls="--", lw=1.2, zorder=0)
    ax.text(0.995, TARGET_DEPTH_UM, f" design target {TARGET_DEPTH_UM:g} µm ", ha="right",
            va="bottom", fontsize=fs.NOTE, color="0.35", transform=ax.get_yaxis_transform())
    if single_speed:
        xs = sorted(units["map_passes"].dropna().unique())
        ax.set_xticks(xs)
        ax.set_xlabel(f"laser passes  (scan speed {speed:g} mm/s, constant across the row)")
    else:
        ax.set_xscale("log")
        ax.set_xlabel("dose ratio  (passes / speed, normalised)")
    ax.set_ylabel("etch depth (µm)")
    ax.set_title("Etch depth vs laser passes")
    ax.margins(y=0.12)                       # headroom for the staggered point callouts
    ax.grid(alpha=0.3)
    ax.legend(handles=handles, loc="best", frameon=True)
    return _finish(fig, path, transparent=transparent)


def _fig_diameter_fidelity(units, path, *, transparent=False):
    """R2 — what the dose that buys depth costs in sidewall oversizing and taper.

    x is the wafer column, blocked by geometry: this is the figure that makes the column-to-geometry
    pairing -- and its reversal between wafer rows -- visible at a glance."""
    colors = _geometry_colors(units)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    u, xs = _column_axis(ax1, units, colors, block_y=1.03)
    _column_axis(ax2, units, colors, label_blocks=False)
    for ax in (ax1, ax2):
        ax.axhline(0, color="k", lw=0.6)
    # marker vocabulary from report.make_diameter_fit: open ▲ top, filled ● mid, open ▼ base
    for x, (_, r) in zip(xs, u.iterrows()):
        c = colors.get(r["geometry"], "0.4")
        if not np.isfinite(r.get("disc_mid_um", np.nan)):
            ax1.plot(x, 0, "x", color="0.6", ms=11, mew=2)
            ax1.text(x, 0, "\nno data", ha="center", va="top", fontsize=fs.ANNOT_SM,
                     color=MISSING_COLOR)
            ax2.plot(x, 0, "x", color="0.6", ms=11, mew=2)
            continue
        ax1.plot(x, r["disc_top_um"], "^", mfc="none", mec=c, ms=10, mew=1.8)
        ax1.plot(x, r["disc_mid_um"], "o", color=c, ms=10)
        ax1.plot(x, r["disc_base_um"], "v", mfc="none", mec=c, ms=10, mew=1.8)
        ax2.plot(x, r["taper_um"], "s", color=c, ms=10)
    ax1.set_ylabel("measured − drawn Ø (µm)")
    ax2.set_ylabel("taper, base − top Ø (µm)")
    ax1.margins(y=0.18)
    ax2.margins(y=0.18)
    _column_ticks(ax2, units)
    # legend outside the axes: with three markers per column there is no reliable empty corner
    ax1.legend(handles=[Line2D([], [], marker="^", mfc="none", mec="0.3", ls="", label="top Ø"),
                        Line2D([], [], marker="o", color="0.3", ls="", label="mid Ø"),
                        Line2D([], [], marker="v", mfc="none", mec="0.3", ls="", label="base Ø")],
               loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=True)
    ax1.set_title("Diameter fidelity and taper across the row", pad=34)
    fig.tight_layout()
    return _finish(fig, path, transparent=transparent)


def _fig_porosity(units, path, *, transparent=False):
    """R3 — does each geometry still deliver its designed open area after machining?

    The one figure where the lattice does real work: ``porosity_table.porosity`` returns
    ``(square, hex)`` and the row's lattice picks the element."""
    colors = _geometry_colors(units)
    lat = (units["lattice"].dropna().iloc[0] if len(units["lattice"].dropna()) else "")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.0, 8.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.35, 1.0]})
    u, xs = _column_axis(ax1, units, colors, block_y=1.03)
    _column_axis(ax2, units, colors, label_blocks=False)
    ax2.axhline(0, color="k", lw=0.6)

    # Panel (a) absolute Φ. Across three geometries Φ spans ~0.33 to ~0.77, so on this axis the
    # machining LOSS is a few pixels -- panel (b) is what actually answers the question.
    for x, (_, r) in zip(xs, u.iterrows()):
        c = colors.get(r["geometry"], "0.4")
        design, ach, top = r["phi_design"], r["phi_achieved"], r.get("phi_top", np.nan)
        if np.isfinite(design):
            ax1.hlines(design, x - 0.34, x + 0.34, color="0.45", lw=1.5)
            ax1.plot(x, design, "D", mfc="none", mec="0.35", ms=9, mew=1.5)
        if not np.isfinite(ach):
            ax1.plot(x, design if np.isfinite(design) else 0, "x", color="0.6", ms=11, mew=2)
            ax1.text(x, design if np.isfinite(design) else 0, "\nno data", ha="center", va="top",
                     fontsize=fs.ANNOT_SM, color=MISSING_COLOR)
            ax2.plot(x, 0, "x", color="0.6", ms=11, mew=2)
            continue
        if np.isfinite(design):
            ax1.vlines(x, ach, design, color=c, lw=1.4, alpha=0.75)
        ax1.plot(x, ach, "o", color=c, ms=10)
        if np.isfinite(top):
            ax1.plot(x, top, "o", mfc="none", mec=c, ms=7, mew=1.5)
        # Panel (b) the same loss in PERCENTAGE POINTS -- comparable across geometries, which is
        # the whole point of running three of them in one row.
        if np.isfinite(design):
            d_pp = 100.0 * (ach - design)
            ax2.vlines(x, 0, d_pp, color=c, lw=6, alpha=0.85)
            ax2.annotate(f"{d_pp:+.1f}", (x, d_pp), textcoords="offset points",
                         xytext=(0, 6 if d_pp >= 0 else -16), ha="center", fontsize=fs.ANNOT,
                         color=c)
            if np.isfinite(top):
                ax2.plot(x, 100.0 * (top - design), "o", mfc="none", mec=c, ms=7, mew=1.5)
    ax1.set_ylabel("open-area fraction Φ")
    ax1.margins(y=0.16)
    ax2.set_ylabel("Φ change vs design\n(percentage points)")
    ax2.margins(y=0.28)
    _column_ticks(ax2, units)
    ax2.set_xlabel(f"wafer column  ({lat or 'unknown'} lattice)")
    ax1.legend(handles=[
        Line2D([], [], marker="D", mfc="none", mec="0.35", ls="", label="designed Φ (drawn Ø)"),
        Line2D([], [], marker="o", color="0.3", ls="", label="achieved Φ (mid Ø)"),
        Line2D([], [], marker="o", mfc="none", mec="0.3", ls="", label="achieved Φ (top Ø)")],
        loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=True)
    ax1.set_title("Achieved vs designed open-area fraction", pad=34)
    fig.tight_layout()
    return _finish(fig, path, transparent=transparent)


def _f(v, nd=1):
    """House '—' formatter for a missing number (from run_sample.render_cell_report)."""
    try:
        return f"{float(v):.{nd}f}" if np.isfinite(float(v)) else "—"
    except (TypeError, ValueError):
        return "—"


_STATUS_COLOR = {"ok": "#5A8F5A", "partial": "#E8A33D", "suspect": "#E8A33D",
                 "failed": MISSING_COLOR, "stale": MISSING_COLOR, "skipped": "0.75"}


def _fig_summary_table(units, records, path, *, transparent=False):
    """R4 — the whole row on one slide: per-column geometry, dose, depth, Ø, taper, Φ and status."""
    cols = ["col", "geometry", "lattice", "laser", "snaps", "depth µm", "spread", "Ø top",
            "Ø mid", "Ø base", "taper", "debris %", "Φ", "status"]
    body = []
    for _, r in units.sort_values("wafer_col").iterrows():
        body.append([
            f"c{int(r['wafer_col'])}", r["geometry"] or "—", r["lattice"] or "—",
            r["laser"] or "—", str(int(r.get("n_snapshots", 0) or 0)),
            _f(r.get("depth_um")),
            _f(r.get("depth_spread_um")) if np.isfinite(r.get("depth_um", np.nan)) else "—",
            _f(r.get("top_diameter_um")), _f(r.get("diameter_um")), _f(r.get("base_diameter_um")),
            _f(r.get("taper_um")),
            _f(100.0 * r["debris_fraction"]) if np.isfinite(r.get("debris_fraction", np.nan)) else "—",
            _f(r.get("phi_achieved"), 3), r.get("status", "—")])
    n = max(len(body), 1)
    fig, ax = plt.subplots(figsize=(15, 1.15 * (n + 1) + 1.4))
    ax.axis("off")
    tbl = ax.table(cellText=body or [["—"] * len(cols)], colLabels=cols, cellLoc="center",
                   bbox=[0, 0, 1, 0.86])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fs.TABLE)
    tbl.auto_set_column_width(range(len(cols)))
    for j in range(len(cols)):
        cell = tbl[0, j]
        cell.set_facecolor("#dddddd")
        cell.set_text_props(weight="bold")
    for i, r in enumerate(body, start=1):
        st = r[-1]
        if st != "ok":
            tbl[i, len(cols) - 1].set_text_props(
                color=_STATUS_COLOR.get(st.split(":")[0], MISSING_COLOR), weight="bold")
    n_ok = sum(1 for rec in records if rec.produced_data)
    n_try = sum(1 for rec in records if rec.planned.status == "ready")
    ax.set_title(f"Wafer row summary — {n_ok} / {n_try} samples registered",
                 fontsize=fs.HEADLINE, weight="bold", pad=16)
    return _finish(fig, path, dpi=300, transparent=transparent, facecolor="white")


def _fig_montage(units, panels, path, *, transparent=False):
    """R5 — does the row look right? One height panel per wafer column, shared colour scale."""
    if not panels:
        print("No montage panels captured -> skipping the row montage "
              "(--rollup-only cannot recover them; re-run the row to get one).")
        return None
    by_col = {int(c): z for c, _name, _geom, _laser, z in panels}
    if not by_col:
        return None
    h = max(z.shape[0] for z in by_col.values())
    w = int(np.median([z.shape[1] for z in by_col.values()]))
    # every wafer column keeps its slot, in order; a failed one becomes an all-NaN panel so the
    # gap is visible rather than silently closed up
    entries = []
    for _, r in units.sort_values("wafer_col").iterrows():
        col = int(r["wafer_col"])
        if col in by_col:
            entries.append((f"c{col} · {r['geometry']} · {r['laser']}", by_col[col]))
        else:
            entries.append((f"c{col} · not registered",
                            np.full((h, w), np.nan, dtype=np.float32)))
    stitch = [(lab, None, z, None) for lab, z in entries]
    canvas, _icanvas, boxes = rs._stitch_snapshot_panels(stitch)
    hh, ww = canvas.shape
    res_um, gut = 8.0, 24
    mm_px = res_um / 1000.0
    fig_w = max(8.0, 4.2 * len(entries))
    fig, ax = plt.subplots(figsize=(fig_w, max(3.2, fig_w * (1.22 * hh) / ww)))
    vmax = float(np.nanpercentile(canvas, 99.5)) if np.isfinite(canvas).any() else 1.0
    im = ax.imshow(canvas, origin="lower", extent=(0.0, ww * mm_px, 0.0, hh * mm_px),
                   cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
    ax.set_xlabel("x (mm — physical within a panel; panels are separate captures)")
    ax.set_ylabel("y (mm)")
    rs._annotate_snapshot_panels(ax, boxes, hh, scale=mm_px)
    rs._image_cb(im, ax, "height above local floor (µm)")
    row_no = int(units["wafer_row"].iloc[0]) if len(units) else 0
    ax.set_title(f"Sample height — wafer row {row_no}")
    fig.tight_layout()
    return _finish(fig, path, transparent=transparent)


def make_row_figures(rollup, units, out_dir, *, panels=None, plan=None, records=(),
                     transparent=False):
    """Build the five row-level figures. Returns the paths actually written.

    Every figure is drawn inside ``report.PLOT_RC`` so the row set matches the per-sample set
    (the house typography in ``figstyle``)."""
    out_dir = Path(out_dir)
    written = []
    # A SKIPPED column carries a placeholder row with no geometry and passes=0. It belongs in the
    # CSVs (the design point was deliberately abandoned, and that is worth recording) but plotting
    # it would add a blank series in a duplicate colour and stretch every axis to zero.
    plot_units = units[units["geometry"].astype(str).str.strip() != ""]
    n_dropped = len(units) - len(plot_units)
    if n_dropped:
        print(f"row figures: excluding {n_dropped} skipped column(s) with no geometry "
              f"(they remain in row_measurements.csv / row_units.csv)")
    units = plot_units
    if units.empty:
        print("row figures: no column has a geometry -> no figures written")
        return written
    with plt.rc_context(report.PLOT_RC):
        for fn, name in ((lambda p: _fig_depth_vs_passes(units, rollup, p,
                                                         transparent=transparent),
                          "row_depth_vs_passes.png"),
                         (lambda p: _fig_diameter_fidelity(units, p, transparent=transparent),
                          "row_diameter_fidelity.png"),
                         (lambda p: _fig_porosity(units, p, transparent=transparent),
                          "row_porosity.png"),
                         (lambda p: _fig_summary_table(units, records, p,
                                                       transparent=transparent),
                          "row_summary_table.png"),
                         (lambda p: _fig_montage(units, panels, p, transparent=transparent),
                          "row_montage.png")):
            try:
                got = fn(out_dir / name)
                if got is not None:
                    written.append(got)
            except Exception as e:                         # one bad figure never loses the CSV
                print(f"row figure {name} failed: {type(e).__name__}: {e}")
    return written


# ==================================================================== summary == #
def render_row_summary(rollup, units, records, plan, *, empty=False):
    """The text twin of the summary table, in the ``calibrate_depth`` gate-report idiom."""
    lat = plan.samples[0].entry.lattice if plan.samples else ""
    lines = []
    n_try = sum(1 for r in records if r.planned.status == "ready")
    n_ok = sum(1 for r in records if r.produced_data)
    n_fail = sum(1 for r in records if r.planned.status == "ready" and not r.produced_data)
    n_skip = sum(1 for r in records if r.planned.status == "skipped")
    lines.append(f"Row {plan.row} ({lat or 'lattice unknown'}, {plan.date_tag or 'undated'}): "
                 f"{n_ok}/{n_try} samples registered  ({n_fail} failed, {n_skip} skipped)")
    for r in records:
        if r.produced_data and r.status == "ok":
            continue
        e = r.planned.entry
        tag = f"c{e.col}" + (f" ({e.geometry}, {e.laser})" if e.geometry else "")
        lines.append(f"  − {tag}: {r.status}" + (f" — {r.reason}" if r.reason else ""))
    if empty or units is None or units.empty:
        lines.append("No sample produced measurements; no CSV and no figures were written.")
        return "\n".join(lines) + "\n"

    speeds = units["map_speed"].dropna().unique()
    per_geom = units.groupby("geometry")["map_passes"].nunique()
    if len(speeds) == 1 and (per_geom <= 2).all():
        lines.append(f"Speed is constant ({speeds[0]:g} mm/s) and each geometry has at most "
                     f"{int(per_geom.max())} dose(s) — too few points for a depth model.")
        lines.append("Pool several rows with calibrate_depth.py for that.")
    lines.append("")
    lines.append(f"{'col':>4}  {'geometry':<11} {'laser':<10} {'depth µm':>9} {'Ø mid':>8} "
                 f"{'taper':>7} {'Φ':>7}  status")
    for _, r in units.sort_values("wafer_col").iterrows():
        lines.append(f"c{int(r['wafer_col']):<3}  {str(r['geometry']):<11} {str(r['laser']):<10} "
                     f"{_f(r.get('depth_um')):>9} {_f(r.get('diameter_um')):>8} "
                     f"{_f(r.get('taper_um')):>7} {_f(r.get('phi_achieved'), 3):>7}  "
                     f"{r.get('status', '')}")
    return "\n".join(lines) + "\n"
