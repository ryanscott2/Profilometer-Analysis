"""
Shared measurement-row builder + the legacy (v1) plot suite.

Both drivers reuse this: ``run_sample.py`` (the tiled full-sample workflow) imports
``result_to_row``, ``CRITICAL_FLAGS``, ``print_diameter_calibration`` and ``make_plots``;
``selftest.py`` additionally uses ``_band_targets``. The figures produced here are the v1
overview/dose/per-band/diameter-fit/grid set, written under ``<out_dir>/figures``.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# flags that mean the READ is untrustworthy (bad data, not merely a non-ideal result). Debris
# ("wide-D") is deliberately NOT here: a debris-widened pin is reliable data that happens to show
# debris -- it is excluded from the diameter-calibration FIT (see _fit_subset), not marked bad.
CRITICAL_FLAGS = ("no relief", "weak lattice", "off-scan", "floor uncertain")


# --------------------------------------------------------------------------- #
def result_to_row(res, reliable):
    """Flatten a PinFinResult (+ reliability) into a measurements.csv row dict."""
    row = res.as_row()
    row["drawn_diameter_um"] = res.nominal_diameter_um
    row["dose_ratio"] = (res.passes / res.speed) if res.speed and res.speed > 0 else np.nan
    row["disc_mid_um"] = res.diameter_um - res.nominal_diameter_um
    row["disc_top_um"] = res.top_diameter_um - res.nominal_diameter_um
    row["disc_base_um"] = res.base_diameter_um - res.nominal_diameter_um
    row["taper_um"] = res.base_diameter_um - res.top_diameter_um
    row["reliable"] = bool(reliable)
    return row


def _band_targets(template):
    """Median drawn diameter per band -> the band's 'target' reference diameter."""
    out = {}
    for b in sorted(set(a.band for a in template.arrays)):
        ds = [a.diameter_um for a in template.arrays if a.band == b]
        out[b] = float(np.median(ds))
    return out


# ============================================================ plot helpers == #
def _pass_colors(df):
    uniq = sorted({int(p) for p in df["passes"].dropna().unique() if p and p > 0})
    cmap = plt.get_cmap("tab10")
    colors = {p: cmap(i % 10) for i, p in enumerate(uniq)}
    return colors


def _color_for(colors, passes):
    return colors.get(int(passes), "0.4") if passes and passes > 0 else "0.4"


def _legend(fig, colors, *, extra=None):
    h = [Line2D([], [], marker="o", color=c, ls="none", label=f"{p} passes")
         for p, c in colors.items()]
    h += [Line2D([], [], marker="o", color="grey", mfc="none", ls="none",
                 label="flagged / unreliable")]
    if extra:
        h += extra
    fig.legend(handles=h, loc="lower center", ncol=max(1, len(h)), frameon=False,
               bbox_to_anchor=(0.5, 0.0))


def _scatter(ax, df, xcol, ycol, colors):
    for _, r in df.iterrows():
        if not np.isfinite(r.get(xcol, np.nan)) or not np.isfinite(r.get(ycol, np.nan)):
            continue
        c = _color_for(colors, r["passes"])
        ax.plot(r[xcol], r[ycol], marker="o", color=c,
                mfc=(c if r["reliable"] else "none"),
                ms=7, mew=1.3, ls="none", alpha=0.9)


# ---------------------------------------------------------------- the plots #
def make_overview(df, out_dir, colors):
    responses = [("diameter_um", "measured mid diameter (µm)"),
                 ("pitch_um", "design pitch (µm)"),   # pitch_um is the DXF lattice, not measured
                 ("depth_um", "measured depth (µm)")]
    predictors = [("passes", "laser passes"),
                  ("speed", "scan speed (mm/s)"),
                  ("drawn_diameter_um", "drawn / design diameter (µm)")]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for i, (yc, yl) in enumerate(responses):
        for j, (xc, xl) in enumerate(predictors):
            ax = axes[i, j]
            _scatter(ax, df, xc, yc, colors)
            if j == 2 and yc == "diameter_um" and df["drawn_diameter_um"].notna().any():
                lim = [0, np.nanmax(df["drawn_diameter_um"]) * 1.1]
                ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
                ax.text(0.97, 0.9, "measured = drawn", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color="k", alpha=0.6)
            if yc == "pitch_um":
                for p in sorted(df["nominal_pitch_um"].dropna().unique()):
                    ax.axhline(p, color="grey", ls="--", lw=0.7)
            if i == 2:
                ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel(yl)
    fig.suptitle("Pin-fin geometry vs process parameters and swept diameter", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    _legend(fig, colors)
    fig.savefig(out_dir / "figures" / "overview_3x3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_depth_dose(df, out_dir, colors):
    d = df[df.reliable & df.dose_ratio.notna()]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if len(d):
        for p, g in d.groupby("passes"):
            c = _color_for(colors, p)
            ax.plot(g.dose_ratio, g.depth_um, "o", color=c, ls="none", ms=8, mew=1.3,
                    label=f"P{int(p)}")
            m = g.groupby("dose_ratio").depth_um.mean().sort_index()
            ax.plot(m.index, m.values, "-", color=c, alpha=0.5)
    ax.axhline(55, color="grey", ls="--", lw=1.2)
    ax.text(0.02, 0.96, "design target 55 µm", transform=ax.transAxes,
            va="top", color="grey", fontsize=9)
    if len(d):
        ax.legend(fontsize=9)
    ax.set_xlabel("dose proxy = passes / speed")
    ax.set_ylabel("etch depth (µm)   [pin top − clean floor]")
    ax.set_title("Etch depth vs dose (reliable pins)")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "depth_vs_dose.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_dose_collapse(df, out_dir, colors):
    d = df[df.reliable].copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    # (a) depth vs speed, one line per (passes, band)
    if len(d):
        for (p, band), gg in d.groupby(["passes", "band"]):
            gg = gg.sort_values("speed")
            axes[0].plot(gg["speed"], gg["depth_um"], "o-", color=_color_for(colors, p),
                         alpha=0.8, ms=6, label=f"P{int(p)} b{band}")
    axes[0].set_xlabel("scan speed (mm/s)"); axes[0].set_ylabel("depth (µm)")
    axes[0].set_title("Ablation depth vs speed")
    if len(d):
        axes[0].legend(fontsize=6, ncol=2)
    # (b) depth vs dose
    dd = d[d.dose_ratio.notna()]
    for p, g in dd.groupby("passes"):
        axes[1].plot(g.dose_ratio, g.depth_um, "o", color=_color_for(colors, p),
                     ls="none", ms=8, label=f"P{int(p)}")
    axes[1].set_xlabel("dose proxy  passes / speed"); axes[1].set_ylabel("depth (µm)")
    axes[1].set_title("Depth vs passes/speed");
    if len(dd):
        axes[1].legend(fontsize=8)
    # (c) diameter discrepancy vs dose
    for p, g in dd.groupby("passes"):
        axes[2].plot(g.dose_ratio, g.disc_mid_um, "o", color=_color_for(colors, p),
                     ls="none", ms=8, label=f"P{int(p)}")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_xlabel("dose proxy  passes / speed")
    axes[2].set_ylabel("measured − drawn diameter (µm)")
    axes[2].set_title("Manufacturing discrepancy vs dose")
    if len(dd):
        axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "dose_collapse.png", dpi=200)
    plt.close(fig)


def make_per_row(df, out_dir, colors):
    bands = sorted(df["band"].dropna().unique())
    if not bands:
        return
    fig, axes = plt.subplots(2, len(bands), figsize=(4.4 * len(bands), 9), squeeze=False)
    for j, band in enumerate(bands):
        g = df[df["band"] == band]
        drawn_d = sorted(g["drawn_diameter_um"].dropna().unique())
        target_d = g["target_diameter_um"].dropna().iloc[0] if g["target_diameter_um"].notna().any() else np.nan
        pitch_nom = g["nominal_pitch_um"].dropna().iloc[0] if g["nominal_pitch_um"].notna().any() else np.nan
        for _, r in g.iterrows():
            c = _color_for(colors, r["passes"])
            mfc = c if r["reliable"] else "none"
            if np.isfinite(r.get("speed", np.nan)) and np.isfinite(r.get("depth_um", np.nan)):
                axes[0][j].plot(r["speed"], r["depth_um"], "o", color=c, mfc=mfc,
                                ms=7, mew=1.3, ls="none")
            if np.isfinite(r.get("speed", np.nan)) and np.isfinite(r.get("diameter_um", np.nan)):
                axes[1][j].plot(r["speed"], r["diameter_um"], "o", color=c, mfc=mfc,
                                ms=7, mew=1.3, ls="none")
        dd = f"{drawn_d[0]:g}–{drawn_d[-1]:g}" if drawn_d else "?"
        axes[0][j].set_title(f"band {band}\ndrawn Ø {dd} | target {target_d:g} | "
                             f"pitch {pitch_nom:g} µm", fontsize=9)
        axes[0][j].set_xlabel("speed (mm/s)"); axes[1][j].set_xlabel("speed (mm/s)")
        if j == 0:
            axes[0][j].set_ylabel("depth (µm)")
            axes[1][j].set_ylabel("mid diameter (µm)")
        for bd in drawn_d:
            axes[1][j].axhline(bd, color="grey", ls="--", lw=0.6, alpha=0.5)
        if np.isfinite(target_d):
            axes[1][j].axhline(target_d, color="g", ls="-.", lw=1.1)
    fig.suptitle("Per-band: depth (top) and mid diameter (bottom) vs speed", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    _legend(fig, colors)
    fig.savefig(out_dir / "figures" / "per_row.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fit_subset(df):
    d = df[df.reliable].copy()
    # Debris-widened pins are reliable READS but not representative of the clean drawn->measured
    # geometry, so drop them from the CALIBRATION fit ONLY (they remain reliable data everywhere
    # else). This is the debris guard -- it does not touch the reliability flag.
    if "flags" in d.columns:
        d = d[~d["flags"].astype(str).str.contains("wide-D", na=False)]
    d = d[d.drawn_diameter_um.notna() & d.top_diameter_um.notna() & d.diameter_um.notna()]
    # Symmetric [0.7x, 1.4x]-drawn window for BOTH top and mid (the mid measure is the more
    # debris-prone, so it must not get the looser bound). Keeps ~2x-drawn debris outliers out of
    # the small-n per-band OLS fit.
    ok = (d.top_diameter_um.between(0.7 * d.drawn_diameter_um, 1.4 * d.drawn_diameter_um) &
          d.diameter_um.between(0.7 * d.drawn_diameter_um, 1.4 * d.drawn_diameter_um))
    return d[ok]


def make_diameter_fit(df, out_dir, colors):
    d = _fit_subset(df)
    bands = sorted(d.band.dropna().unique())
    if not bands:
        return
    fig, axes = plt.subplots(1, len(bands), figsize=(4.6 * len(bands), 5.0), squeeze=False)
    for j, band in enumerate(bands):
        ax = axes[0][j]
        g = d[d.band == band]
        if not len(g):
            ax.set_title(f"band {band} (no data)"); continue
        tgt = g.target_diameter_um.iloc[0]
        xr = np.array([g.drawn_diameter_um.min() * 0.9, g.drawn_diameter_um.max() * 1.1])
        ax.plot(xr, xr, color="grey", ls="--", lw=0.8, label="measured = drawn")
        for meas, mk, name in [("top_diameter_um", "^", "top"),
                               ("diameter_um", "o", "mid")]:
            for _, r in g.iterrows():
                c = _color_for(colors, r.passes)
                ax.plot(r.drawn_diameter_um, r[meas], marker=mk, color=c,
                        mfc="none" if meas == "top_diameter_um" else c,
                        ms=7, mew=1.3, ls="none")
            if g.drawn_diameter_um.nunique() >= 2:
                a, b = np.polyfit(g.drawn_diameter_um, g[meas], 1)
                ax.plot(xr, a * xr + b, "-" if meas == "top_diameter_um" else ":",
                        color="k", lw=1.3, label=f"{name}: y={a:.2f}x{b:+.0f}")
        ax.axhline(tgt, color="g", ls="-.", lw=1.0)
        ax.set_title(f"band {band}")
        ax.set_xlabel("drawn / design Ø (µm)")
        if j == 0:
            ax.set_ylabel("measured Ø (µm)  ▲ top (open)  ● mid (filled)")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Measured vs drawn diameter with linear fit (colour = passes)", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "figures" / "diameter_fit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_grid_overlays(results, out_dir):
    if not results:
        return
    n = len(results)
    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.4 * nrows),
                             squeeze=False)
    for idx, (sample, res, reliable) in enumerate(results):
        ax = axes[idx // ncols][idx % ncols]
        thumb = res.thumb
        if thumb is None or not np.isfinite(thumb).any():
            ax.axis("off"); continue
        H, W = thumb.shape
        ax.imshow(thumb, origin="lower", cmap="viridis",
                  vmin=np.nanpercentile(thumb, 2), vmax=np.nanpercentile(thumb, 98))
        d = res.thumb_down
        pxf, pyf = res.grid_px_px / d, res.grid_py_px / d
        ph_c, ph_r = res.grid_phase[1] / d, res.grid_phase[0] / d
        if pxf > 1 and pyf > 1:
            xs = np.arange(ph_c % pxf, W, pxf)
            ys = np.arange(ph_r % pyf, H, pyf)
            gx, gy = np.meshgrid(xs, ys)
            ax.plot(gx.ravel(), gy.ravel(), "r+", ms=4, mew=0.7)
        ax.set_title(f"c{sample.cell_id} b{sample.band}c{sample.col} "
                     f"D{sample.nominal_diameter_um:g}\nP{sample.passes} S{sample.speed:g}"
                     + ("" if reliable else "  ⚠"),
                     fontsize=6, color=("black" if reliable else "firebrick"))
        ax.set_xticks([]); ax.set_yticks([])
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Known pin grid on each array  (red + = design pin centres; "
                 "⚠ = flagged)", y=1.0, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_dir / "figures" / "grid_overlays.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_diameter_calibration(df, out_dir):
    d = _fit_subset(df)
    dfd = df[df.drawn_diameter_um.notna()] if len(df) else df
    lines = ["Diameter calibration  (measured = a*drawn + b; to hit target, draw (target-b)/a)",
             "Primary = TOP diameter; MID also shown.",
             "Per-diameter lines give the measured Ø for each nominal geometry in the DXF and "
             "the drawn Ø that would hit it.", ""]
    if not len(d) or not len(dfd):
        lines.append("(no reliable pins with plausible diameters yet)")
        (out_dir / "diameter_calibration.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n" + "\n".join(lines))
        return

    def fit(g, col):
        if g.drawn_diameter_um.nunique() < 2:
            return None
        a, b = np.polyfit(g.drawn_diameter_um, g[col], 1)
        return float(a), float(b)

    def _first(series, default=float("nan")):
        s = series.dropna()
        return s.iloc[0] if len(s) else default

    def invert(target, a, b, ref):
        """measured = a*drawn + b  ->  drawn = (target-b)/a. A near-flat slope (|a|~0) or a result far
        outside the drawn range makes "draw X" meaningless -> NaN, so an absurd actionable value
        (e.g. a=0.01 -> 'draw 500 µm') is never printed. ``ref`` = the largest drawn Ø in the band."""
        if not (np.isfinite(a) and np.isfinite(b)) or abs(a) < 1e-3:
            return float("nan")
        x = (target - b) / a
        return x if (np.isfinite(x) and 0.0 < x < 3.0 * ref) else float("nan")

    for band in sorted(dfd.band.dropna().unique()):
        gband = dfd[dfd.band == band]                    # every nominal Ø drawn in this band
        g = d[d.band == band]                            # reliable+plausible subset for the fit
        tgt = _first(gband.target_diameter_um)
        pitch = _first(gband.nominal_pitch_um)
        seg = [f"band {band} (pitch {pitch:g}, target Ø {tgt:g} µm, n={len(g)}):"]
        nominal_ds = sorted(gband.drawn_diameter_um.unique())
        ref = max(nominal_ds) if nominal_ds else tgt      # sane upper bound for an inverted "draw X"
        for col, nm in [("top_diameter_um", "TOP"), ("diameter_um", "MID")]:
            fb = fit(g, col)
            if fb:
                a, b = fb
                draw = invert(tgt, a, b, ref)
                dtxt = (f"draw {draw:6.1f} µm to get {tgt:g} µm {nm.lower()}-Ø" if np.isfinite(draw)
                        else f"slope {a:.3g} too flat to invert to {tgt:g} µm {nm.lower()}-Ø")
                seg.append(f"    {nm}: measured = {a:.3f}*drawn {b:+.1f}  ->  {dtxt}")
            else:
                seg.append(f"    {nm}: (need >=2 reliable drawn diameters to fit a line)")
            for D in nominal_ds:                         # one result per nominal diameter
                gd = g[np.isclose(g.drawn_diameter_um, D)]
                meas = (f"measured {gd[col].median():6.1f} µm (n={len(gd)})" if len(gd)
                        else "no reliable measurement")
                di = invert(D, fb[0], fb[1], ref) if fb else float("nan")
                inv = (f"  ->  draw {di:6.1f} µm to get {D:g} µm {nm.lower()}-Ø"
                       if np.isfinite(di) else "")
                seg.append(f"        drawn {D:6.1f} µm: {meas}{inv}")
        lines.append("\n".join(seg))
    if d.drawn_diameter_um.nunique() >= 2:
        ga, gb = np.polyfit(d.drawn_diameter_um, d.top_diameter_um, 1)
        ma, mb = np.polyfit(d.drawn_diameter_um, d.diameter_um, 1)
        lines += ["", f"GLOBAL (all bands, {len(d)} pins):",
                  f"    TOP: measured = {ga:.3f}*drawn {gb:+.1f} µm",
                  f"    MID: measured = {ma:.3f}*drawn {mb:+.1f} µm"]
    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / "diameter_calibration.txt").write_text(text, encoding="utf-8")


def _ols_fit(X, y, alpha=0.05):
    """OLS via statsmodels. X already includes the intercept column. Returns
    (params, conf_int[p,2], r2, adj_r2, n, p, pvalues)."""
    import statsmodels.api as sm
    res = sm.OLS(np.asarray(y, float), np.asarray(X, float)).fit()
    return (np.asarray(res.params), np.asarray(res.conf_int(alpha)),
            float(res.rsquared), float(res.rsquared_adj),
            int(res.nobs), X.shape[1], np.asarray(res.pvalues))


def _model_design(g):
    """Centered design matrix for TOP-Ø ~ drawn + passes + speed + drawn:passes + drawn:speed.
    A predictor (and its interactions) is DROPPED when it is constant within the group, so the fit
    stays well-posed on a single-speed / single-diameter subset. Returns (X, y, term_names)."""
    y = g["top_diameter_um"].to_numpy(float)
    cols = [("Ø at mean (µm)", np.ones(len(g)))]
    centered, unit = {}, {"drawn": "µm/µm", "passes": "µm/pass", "speed": "µm per mm/s"}
    for nm, series in (("drawn", g["drawn_diameter_um"]), ("passes", g["passes"]),
                       ("speed", g["speed"])):
        v = series.to_numpy(float)
        if np.ptp(v) > 1e-9:                              # keep only non-constant predictors
            centered[nm] = v - v.mean()
            cols.append((f"dØ/d {nm} ({unit[nm]})", centered[nm]))
    for a, b in (("drawn", "passes"), ("drawn", "speed")):
        if a in centered and b in centered:
            cols.append((f"{a}×{b}", centered[a] * centered[b]))
    return np.column_stack([c[1] for c in cols]), y, [c[0] for c in cols]


def make_diameter_model(df, out_dir):
    """ADDITIVE calibration (does not replace print_diameter_calibration): per band, fit
    TOP Ø ~ drawn + passes + speed + interactions by OLS on reliable pins, with R²/adj-R² and 95%
    CIs -- so the drawn->measured relationship is conditional on the laser process rather than a
    single line pooled over an ~8x dose range. Writes diameter_model.txt + diameter_model.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import statsmodels.api  # noqa: F401 -- used by _ols_fit; skip this figure if absent
    except Exception:
        msg = "diameter_model: statsmodels not installed -> skipping (pip install statsmodels)"
        print(msg)
        (out_dir / "diameter_model.txt").write_text(msg + "\n", encoding="utf-8")
        return
    # Use the same clean calibration subset as the simpler diameter fits. In particular,
    # debris-widened ("wide-D") reads are valid measurements but must not teach the process
    # model that debris is a reproducible diameter gain.
    d = _fit_subset(df)
    d = d[d["top_diameter_um"].notna() & d["drawn_diameter_um"].notna()
          & (d["passes"] > 0) & (d["speed"] > 0)]
    lines = ["Diameter model  (ADDITIVE to diameter_calibration.txt)",
             "TOP Ø ~ drawn + passes + speed + drawn:passes + drawn:speed  (OLS, reliable pins)",
             "Predictors are CENTERED per band, so the intercept is the top Ø at that band's mean",
             "drawn/passes/speed; a term is dropped when constant within the band. 95% CIs shown.",
             "Conditional on process -- unlike the single pooled per-band line in the calibration.", ""]
    if not len(d):
        lines.append("(no reliable pins with a top diameter to fit)")
        (out_dir / "diameter_model.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    cmap = plt.get_cmap("tab10")
    lo_hi = []
    for i, band in enumerate(sorted(d["band"].unique())):
        g = d[d["band"] == band]
        X, y, names = _model_design(g)
        pitch = g["nominal_pitch_um"].dropna().iloc[0] if g["nominal_pitch_um"].notna().any() else float("nan")
        if X.shape[0] < X.shape[1] + 2:
            lines.append(f"band {band} (pitch {pitch:g} µm): n={len(g)} too few for a "
                         f"{X.shape[1]}-term fit; skipped\n")
            continue
        beta, ci, r2, adj, n, p, pvals = _ols_fit(X, y)
        pred = X @ beta
        lines.append(f"band {band} (pitch {pitch:g} µm, n={n}, df_resid={n - p}):  "
                     f"R² = {r2:.3f}   adj-R² = {adj:.3f}")
        cond = float(np.linalg.cond(X)) if X.shape[1] > 1 else 1.0
        if cond > 1e8:            # near-collinear design (e.g. speed co-swept with passes)
            lines.append(f"    WARNING: design condition number {cond:.1e} — predictors are nearly "
                         f"collinear; the per-term coefficients below are NOT individually reliable "
                         f"(their combination still fits; don't read dØ/d passes vs dØ/d speed apart).")
        if n - p < 5:             # too few residual DOF -> R² is structurally optimistic
            lines.append(f"    WARNING: only {n - p} residual DOF (n={n}, p={p}); the R² above is "
                         f"optimistically inflated — treat this band as underpowered (lean on adj-R² "
                         f"and the 95% CIs, not R²).")
        for nm, b_, (lo, hi), pv in zip(names, beta, ci, pvals):
            lines.append(f"    {nm:>22} = {b_:+9.3f}   [95% CI {lo:+.3f}, {hi:+.3f}]   p={pv:.2g}")
        lines.append("")
        c = cmap(i % 10)
        axes[0].scatter(pred, y, s=16, color=c, alpha=0.6, edgecolors="white", linewidths=0.3,
                        label=f"band {band}  R²={r2:.2f}")
        axes[1].scatter(pred, y - pred, s=16, color=c, alpha=0.6, edgecolors="white", linewidths=0.3)
        lo_hi += [float(np.min([pred.min(), y.min()])), float(np.max([pred.max(), y.max()]))]
    if lo_hi:
        lim = [min(lo_hi), max(lo_hi)]
        axes[0].plot(lim, lim, "k--", lw=0.8, alpha=0.6)
    axes[0].set_xlabel("model-predicted top Ø (µm)"); axes[0].set_ylabel("measured top Ø (µm)")
    axes[0].set_title("Diameter model parity (per-band OLS on drawn, passes, speed)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("model-predicted top Ø (µm)"); axes[1].set_ylabel("residual measured-predicted (µm)")
    axes[1].set_title("residuals"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p_png = out_dir / "diameter_model.png"
    fig.savefig(p_png, dpi=170); plt.close(fig)
    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / "diameter_model.txt").write_text(text, encoding="utf-8")
    print(f"Wrote {p_png}")


def make_plots(df, results, out_dir):
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    if not len(df):
        print("No rows -> no plots.")
        return
    colors = _pass_colors(df)
    make_overview(df, out_dir, colors)
    make_dose_collapse(df, out_dir, colors)
    make_per_row(df, out_dir, colors)
    make_diameter_fit(df, out_dir, colors)
    make_depth_dose(df, out_dir, colors)
    make_grid_overlays(results, out_dir)
    print(f"Wrote figures to {out_dir/'figures'}")
