"""
Cross-sample etch-depth calibration:  ``depth = f(passes, speed)`` conditioned on pin band.

This is the reframed inverse of the diameter model in ``report.make_diameter_model`` (change #4,
which fit ``TOP-Ø = f(drawn, passes, speed)``). Here the RESPONSE is etch depth and the PREDICTORS
are the laser dose (passes, speed): "for this pin geometry, what depth does a given (passes, speed)
produce -- and inversely, what (passes, speed) hits a target depth (e.g. 55 µm)?".

Why depth pools across datasets (unlike absolute height): ``depth_um`` = pin-top − clean-floor is a
LOCAL differential measured per array, so per-tile Z offsets and stage tilt cancel. That makes it
comparable across samples/wafers/dates, so we can pool the completed runs for a calibration with far
more (passes, speed) coverage than any single sample has.

It is POST-HOC and ADDITIVE: it reads the saved ``Results/<sample>/legacy/measurements.csv`` files
(no re-registration / re-extraction) and writes its own report + figures under a separate output
folder. Nothing in the per-run pipeline (``run_sample.analyze_sample``) changes. Re-runnable with a
different sample selection each time (that is what the UI include/exclude control drives).

Usage:
    python calibrate_depth.py                                  # all samples, target 55 µm
    python calibrate_depth.py --include A,B --targets 45,55,65
    python calibrate_depth.py --results Results --out Results/_depth_calibration --exclude bad_run
    python calibrate_depth.py --cell-filters cells.json        # keep/drop cell_ids per sample

Deliverables (under --out):  depth_calibration.txt, depth_vs_dose.png, depth_parity.png,
depth_heatmap.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the exact OLS helper the diameter model uses so both calibrations report CIs/R² the same
# way. X must already include the intercept column; returns (params, conf_int, r2, adj_r2, n, p, pv).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import _ols_fit                                     # noqa: E402

HERE = Path(__file__).resolve().parent
DEF_RESULTS = HERE / "Results"
OUT_NAME = "etch depth"                # cross-sample output dir; has no legacy/ so it self-skips discovery
MEAS_REL = Path("legacy") / "measurements.csv"
DEF_TARGET_UM = 55.0                   # the design target used throughout the codebase
DEF_MAX_DEBRIS = 0.6                   # depth-specific: a debris-buried floor gives untrustworthy depth
PITCH_MATCH_TOL_UM = 1.0               # a row's nominal pitch must equal a band's declared pitch within
                                       # this (design pitches are discrete, so this is effectively exact)
MIN_PREDICTIVE_R2 = 0.10               # below this, a fit is too weak to drive process inversion
_MC_SEED = 0                           # inverse-mean confidence Monte-Carlo (seeded = reproducible)


# ============================================================ discover + pool == #
def discover_samples(results_dir, out_dir, include=None, exclude=None):
    """Sample folders under ``results_dir`` that hold a ``legacy/measurements.csv``.

    ``include``/``exclude`` are lists of folder names (exact match); ``include`` (when given)
    whitelists, ``exclude`` blacklists. The output folder is skipped explicitly (it also has no
    ``legacy/`` so it would self-skip anyway). Returns an ordered list of (name, csv_path)."""
    results_dir = Path(results_dir)
    out_resolved = Path(out_dir).resolve()
    inc = set(include) if include else None
    exc = set(exclude) if exclude else set()
    found = []
    if not results_dir.is_dir():
        return found
    for sub in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if sub.resolve() == out_resolved:
            continue
        csv = sub / MEAS_REL
        if not csv.is_file():
            continue
        if inc is not None and sub.name not in inc:
            continue
        if sub.name in exc:
            continue
        found.append((sub.name, csv))
    return found


def parse_cell_spec(spec):
    """Parse a per-sample cell-selection spec into ``(mode, ids)``.

    Grammar (matches the UI's inline 'cells' field):
        "1-5, 8, 12-16"  -> ('include', {1,2,3,4,5,8,12,13,14,15,16})  keep ONLY these cell_ids
        "!3, 7-9"        -> ('exclude', {3,7,8,9})                     drop these cell_ids
        "" / None        -> ('include', None)                         no filter (keep all)

    ``cell_id`` is the 1-based unit-cell index written by register/run_sample (row-major: row 1 =
    top, col 1 = left; see register._assign_grid_indices). Whitespace around tokens is ignored.
    Raises ValueError on a malformed token, an out-of-range (< 1) id, an inverted range (low >
    high), or a non-empty spec that resolves to zero cells."""
    if spec is None:
        return "include", None
    s = str(spec).strip()
    if not s:
        return "include", None
    mode = "include"
    if s[0] == "!":
        mode, s = "exclude", s[1:].strip()
    ids = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo_s, _, hi_s = tok.partition("-")
            lo_s, hi_s = lo_s.strip(), hi_s.strip()
            if not (lo_s.isdigit() and hi_s.isdigit()):
                raise ValueError(f"bad cell range '{tok}' (use e.g. 12-16)")
            lo, hi = int(lo_s), int(hi_s)
            if lo < 1 or hi < 1:
                raise ValueError(f"cell ids are 1-based; '{tok}' is out of range")
            if lo > hi:
                raise ValueError(f"inverted cell range '{tok}' (low > high)")
            ids.update(range(lo, hi + 1))
        elif tok.isdigit():
            v = int(tok)
            if v < 1:
                raise ValueError(f"cell ids are 1-based; '{tok}' is out of range")
            ids.add(v)
        else:
            raise ValueError(f"bad cell token '{tok}' (expected a number or a range like 12-16)")
    if not ids:
        raise ValueError("cell filter selects no cells")
    return mode, ids


def _apply_cell_filter(df, name, spec):
    """Restrict ``df`` to the cell_ids named by ``spec`` (see :func:`parse_cell_spec`).

    Returns the frame unchanged when the spec is empty. Returns ``None`` -- signalling the caller to
    DROP the whole sample (fail-closed) -- when the spec is non-trivial but the frame has no
    'cell_id' column, so an un-applicable filter never silently pools the full sample against the
    user's stated intent. Prints one line summarising what was kept/dropped (provenance in console)."""
    mode, ids = parse_cell_spec(spec)                    # raises ValueError on a malformed spec
    if ids is None:
        return df
    if "cell_id" not in df.columns:
        print(f"WARNING: sample '{name}' has no 'cell_id' column -> cannot apply cell filter "
              f"'{spec}'; SKIPPING this sample (fail-closed).")
        return None
    cid = pd.to_numeric(df["cell_id"], errors="coerce")
    present = {int(v) for v in cid.dropna().unique()}
    if mode == "include":
        keep = cid.isin(ids)
        used, missing = sorted(ids & present), sorted(ids - present)
        note = f"; requested but absent: {missing}" if missing else ""
        print(f"Cell filter [{name}]: INCLUDE {sorted(ids)} -> kept {int(keep.sum())}/{len(df)} rows "
              f"(cell_ids {used}{note}).")
    else:
        keep = ~cid.isin(ids)
        removed = sorted(ids & present)
        print(f"Cell filter [{name}]: EXCLUDE {sorted(ids)} -> dropped {int((~keep).sum())}/{len(df)} "
              f"rows (cell_ids {removed}).")
    return df[keep].copy()


def load_pooled(samples, cell_filters=None):
    """Read each (name, csv) and concat into one frame, injecting the ``sample`` column (the folder
    name -- the one field measurements.csv does not already carry). Missing optional columns are
    back-filled with NaN so a mix of old/new CSV schemas still concatenates.

    ``cell_filters`` (optional) maps a sample folder name -> a cell-selection spec (see
    :func:`parse_cell_spec`); a sample's rows are restricted to those cells BEFORE pooling. A sample
    whose filter cannot be applied (no 'cell_id' column) is dropped rather than pooled unfiltered."""
    cell_filters = cell_filters or {}
    frames = []
    for name, csv in samples:
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"WARNING: could not read {csv}: {e} -> skipping {name}")
            continue
        df["sample"] = name
        spec = cell_filters.get(name)
        if spec:
            df = _apply_cell_filter(df, name, spec)
            if df is None:                               # filter un-applicable -> skip whole sample
                continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # dose proxy = passes/speed (same def as report). Reconstruct PER ROW wherever it is missing --
    # not just when the whole column is absent -- so pooling an old CSV (no dose_ratio) with a new
    # one (which has it) doesn't leave the old rows NaN (they'd then be dropped as "dose not finite"
    # despite valid passes/speed).
    if "dose_ratio" not in df.columns:
        df["dose_ratio"] = np.nan
    if {"passes", "speed"} <= set(df.columns):
        need = df["dose_ratio"].isna() & df["passes"].notna() & (df["speed"] > 0)
        df.loc[need, "dose_ratio"] = df.loc[need, "passes"] / df.loc[need, "speed"]
    return df


# ================================================================= quality gates == #
def apply_gates(df, max_debris=DEF_MAX_DEBRIS, drop_shallow=False,
                allow_legacy_qc=False):
    """Keep only rows whose depth read is trustworthy for a calibration fit, tracking WHY rows drop.

    ``reliable`` already excludes the critical flags (no relief / weak lattice / off-scan / floor
    uncertain == floor_reliable False) and requires passes>0 and cell reg_score>=0.5, so the only
    depth-specific gates left are a stricter debris cut and (optionally) the ``shallow`` flag.
    ``shallow`` (<3 µm) points are KEPT by default: they are the low-dose anchor of the saturation
    curve's rise and dropping them biases the fit toward the plateau. Returns (kept_df, report)."""
    missing_qc = [c for c in ("reliable", "debris_fraction") if c not in df.columns]
    if missing_qc and not allow_legacy_qc:
        raise ValueError(
            "pooled measurements lack required quality-control column(s) "
            f"{missing_qc}; refusing to fit unvetted legacy data. Re-run extraction, or use "
            "--allow-legacy-qc only after reviewing the older measurements manually."
        )
    if missing_qc:
        print("WARNING: explicit legacy-QC override: missing " + ", ".join(missing_qc))

    n0 = len(df)
    rep = {"total": n0, "steps": []}

    def step(mask, why):
        nonlocal df
        before = len(df)
        df = df[mask]
        rep["steps"].append((why, before - len(df)))

    # Each gate is guarded on column presence: a mask built from an ABSENT column via df.get(...) is
    # a numpy scalar, and df[scalar] raises a cryptic KeyError instead of skipping. An empty frame is
    # fine (empty masks), so the caller's "no rows survived" check handles the all-failed-reads case.
    if "reliable" in df.columns:
        reliable = _truthy(df["reliable"])
        if allow_legacy_qc:
            reliable = reliable | df["reliable"].isna()
        step(reliable, "not reliable (critical flag / P=0 / low reg)")
    for col in ("depth_um", "passes", "speed", "dose_ratio"):
        if col not in df.columns:
            print(f"WARNING: column '{col}' absent from pooled data -> cannot gate on it.")
    if "depth_um" in df.columns:
        step(np.isfinite(df["depth_um"]) & (df["depth_um"] > 0), "depth not finite/>0")
    if {"passes", "speed", "dose_ratio"} <= set(df.columns):
        step((df["passes"] > 0) & (df["speed"] > 0) & np.isfinite(df["dose_ratio"]),
             "passes/speed/dose not positive-finite")
    if "debris_fraction" in df.columns:
        debris = pd.to_numeric(df["debris_fraction"], errors="coerce")
        if allow_legacy_qc:
            # The explicit compatibility mode preserves unmeasured legacy rows, but it is never
            # the default because NaN debris has not passed a debris-quality gate.
            step(debris.isna() | (debris <= max_debris),
                 f"debris_fraction > {max_debris:g}")
        else:
            step(np.isfinite(debris) & (debris <= max_debris),
                 f"debris_fraction missing or > {max_debris:g}")
    if drop_shallow and "flags" in df.columns:
        step(~df["flags"].astype(str).str.contains("shallow", na=False), "shallow flag (--drop-shallow)")
    rep["kept"] = len(df)
    return df.copy(), rep


def aggregate_cell_bands(df, allow_legacy_qc=False):
    """Collapse array rows to one independent observation per sample/cell/band.

    Arrays cut with the same cell-level laser exposure are technical replicates, not independent
    process trials. Their median depth and geometry therefore form the unit used by model
    selection, confidence intervals, mixed models, plots, and inverse recommendations.
    """
    d = df.copy()
    required = {"sample", "band", "passes", "speed", "dose_ratio", "depth_um",
                "drawn_diameter_um", "nominal_pitch_um"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"cannot aggregate calibration rows; missing required columns: {missing}")
    if "cell_id" not in d.columns:
        if not allow_legacy_qc:
            raise ValueError(
                "pooled measurements lack cell_id, so array technical replicates cannot be "
                "collapsed safely; re-run extraction or use --allow-legacy-qc after review"
            )
        print("WARNING: legacy rows have no cell_id; treating each row as its own analysis unit.")
        d["cell_id"] = [f"legacy-row-{i}" for i in range(len(d))]
    elif d["cell_id"].isna().any():
        if not allow_legacy_qc:
            raise ValueError("cell_id is missing on some rows; refusing pseudo-replicated fitting")
        missing_id = d["cell_id"].isna()
        d.loc[missing_id, "cell_id"] = [f"legacy-row-{i}" for i in d.index[missing_id]]

    keys = ["sample", "cell_id", "band", "nominal_pitch_um",
            "passes", "speed", "dose_ratio"]
    aggregations = {
        "depth_um": "median",
        "drawn_diameter_um": "median",
    }
    for col in ("debris_fraction", "cell_row", "cell_col"):
        if col in d.columns:
            aggregations[col] = "median"
    grouped = d.groupby(keys, dropna=False, observed=True)
    out = grouped.agg(aggregations).reset_index()
    counts = grouped.size().rename("array_rows").reset_index()
    out = out.merge(counts, on=keys, how="left", validate="one_to_one")
    return out


def _truthy(series):
    """Coerce a possibly-mixed 'reliable' column to a clean boolean mask. A missing column becomes
    NaN after pd.concat of old/new-schema CSVs, and Series.astype(bool) maps NaN -> True, which would
    let un-vetted rows through the reliability gate; treat NaN / anything not clearly true as False."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def _gate_report_lines(rep):
    lines = [f"Quality gates: {rep['kept']}/{rep['total']} rows retained"]
    for why, n in rep["steps"]:
        if n:
            lines.append(f"    − {n:>4} dropped: {why}")
    return lines


# ============================================================= user-defined bands == #
def load_band_defs(path):
    """Parse the band-definitions file -> list of (idx, dmin, dmax, pitch), idx = 1..N in file order.

    Each non-blank, non-``#`` line is ``min_Ø, max_Ø, pitch`` in µm: the first two numbers are the
    drawn-diameter range the band covers, the third is its centre-to-centre pitch. Returns None if
    the file is absent or has no band rows -> the caller then falls back to the measurements
    ``band`` column. Raises SystemExit on a malformed line so a typo cannot silently mis-bin data."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    defs, idx = [], 0
    for lineno, raw in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [c.strip() for c in line.replace("\t", ",").split(",") if c.strip()]
        if len(parts) != 3:
            raise SystemExit(f"band defs line {lineno}: expected 'min, max, pitch' (3 numbers), got {parts!r}")
        try:
            dmin, dmax, pitch = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            raise SystemExit(f"band defs line {lineno}: non-numeric value in {parts!r}")
        if dmin > dmax:
            raise SystemExit(f"band defs line {lineno}: min Ø {dmin:g} > max Ø {dmax:g}")
        if pitch <= 0:
            raise SystemExit(f"band defs line {lineno}: pitch must be > 0 (got {pitch:g})")
        idx += 1
        defs.append((idx, dmin, dmax, pitch))
    return defs or None


def assign_bands(df, defs):
    """Re-bin pooled rows into the user-defined bands. A row joins band i iff BOTH its drawn Ø ∈
    [min, max] (both ends inclusive) AND its nominal pitch matches band i's declared pitch (within
    ``PITCH_MATCH_TOL_UM`` -- pitch is a STRICT requirement). If several bands qualify (same pitch
    with overlapping ranges) the lowest-index band wins. Rows matching no band -- drawn Ø out of
    every range, a mismatched pitch, or a missing pitch -- are dropped and reported (with their
    distinct (Ø, pitch) so a pitch typo is easy to spot). Overwrites the ``band`` column with the
    band index (1..N). Returns (df, band_meta, report_lines)."""
    drawn = df["drawn_diameter_um"].to_numpy(float)
    pitch = (df["nominal_pitch_um"].to_numpy(float) if "nominal_pitch_um" in df.columns
             else np.full(len(df), np.nan))
    band = np.full(len(df), -1, dtype=int)
    for idx, dmin, dmax, bpitch in defs:
        inr = np.isfinite(drawn) & (drawn >= dmin) & (drawn <= dmax)
        pmatch = np.isfinite(pitch) & np.isclose(pitch, bpitch, atol=PITCH_MATCH_TOL_UM, rtol=0.0)
        take = inr & pmatch & (band == -1)               # Ø in range AND pitch matches; first wins
        band[take] = idx
    lines = [f"Band definitions ({len(defs)} band(s)) — a row joins a band only if its drawn Ø is "
             f"in range AND its pitch matches (±{PITCH_MATCH_TOL_UM:g} µm):"]
    for idx, dmin, dmax, bpitch in defs:
        lines.append(f"    band {idx}: Ø {dmin:g}–{dmax:g} µm @ {bpitch:g} µm pitch  ->  "
                     f"{int((band == idx).sum())} rows")
    n_out = int((band == -1).sum())
    if n_out:
        combos = sorted({(float(f"{d:g}"), (float(f"{p:g}") if np.isfinite(p) else float("nan")))
                         for d, p in zip(drawn[band == -1], pitch[band == -1])})
        lines.append(f"    dropped {n_out} row(s) matching no band — (drawn Ø, pitch): {combos}")
    out = df.assign(band=band)
    out = out[out["band"] != -1].copy()
    band_meta = {idx: (dmin, dmax, bpitch) for idx, dmin, dmax, bpitch in defs}
    return out, band_meta, lines


# ===================================================================== the fits == #
def _sat(dose, a, k):
    """Saturating dose-response: depth = a·(1 − exp(−k·dose)).  a = plateau depth, k = dose rate."""
    return a * (1.0 - np.exp(-k * dose))


def _aicc(n, ss_res, k):
    """Small-sample Akaike information criterion on a common response data set."""
    n, k = int(n), int(k)
    if n <= k + 1 or not np.isfinite(ss_res) or ss_res < 0:
        return float("inf")
    mse = max(float(ss_res) / n, np.finfo(float).tiny)
    aic = n * np.log(mse) + 2 * k
    return float(aic + (2 * k * (k + 1)) / (n - k - 1))


def fit_saturating(dose, depth, alpha=0.05):
    """Nonlinear least-squares saturating fit via scipy.  Returns a dict with a, k (+ 95% CI from
    the covariance), R², residual σ, the param covariance (for the inversion MC), n, and ok/msg."""
    dose = np.asarray(dose, float); depth = np.asarray(depth, float)
    m = np.isfinite(dose) & np.isfinite(depth)
    dose, depth = dose[m], depth[m]
    out = {"form": "saturating  depth = a·(1−exp(−k·dose))", "n": int(dose.size), "ok": False}
    if dose.size < 4 or np.ptp(dose) <= 0:
        out["msg"] = f"n={dose.size} or single dose: too few for a 2-parameter saturating fit"
        return out
    try:
        from scipy.optimize import curve_fit
        from scipy.stats import t as _t
        a0 = max(float(np.nanmax(depth)) * 1.1, 1.0)
        k0 = 1.0 / max(float(np.nanmedian(dose)), 1e-9)
        beta, pcov = curve_fit(_sat, dose, depth, p0=[a0, k0],
                               bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=20000)
        pred = _sat(dose, *beta)
        ss_res = float(np.sum((depth - pred) ** 2))
        ss_tot = float(np.sum((depth - depth.mean()) ** 2))
        dof = max(dose.size - 2, 1)
        se = np.sqrt(np.diag(pcov))
        tcrit = float(_t.ppf(1 - alpha / 2, dof))
        out.update(a=float(beta[0]), k=float(beta[1]),
                   a_ci=(float(beta[0] - tcrit * se[0]), float(beta[0] + tcrit * se[0])),
                   k_ci=(float(beta[1] - tcrit * se[1]), float(beta[1] + tcrit * se[1])),
                   r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                   ss_res=ss_res, k_params=2, aicc=_aicc(dose.size, ss_res, 2),
                   resid_sd=float(np.sqrt(ss_res / dof)), cov=pcov, beta=beta, ok=True)
    except Exception as e:                                  # non-convergence / singular cov
        out["msg"] = f"fit did not converge: {e}"
    return out


def _design_log_dose(g, with_diameter):
    """OLS design for depth ~ 1 + log(dose) [+ drawn_diameter].  Adds the drawn-Ø covariate only
    when the band spans >1 drawn diameter (else it is constant and unidentifiable)."""
    names = ["intercept", "log(dose)"]
    cols = [np.ones(len(g)), np.log(g["dose_ratio"].to_numpy(float))]
    if with_diameter and g["drawn_diameter_um"].nunique() >= 2:
        names.append("drawn_Ø (µm)")
        cols.append(g["drawn_diameter_um"].to_numpy(float))
    return np.column_stack(cols), g["depth_um"].to_numpy(float), names


def fit_log_dose_ols(g, alpha=0.05):
    """depth ~ 1 + log(dose) [+ drawn_Ø]  via report._ols_fit.  Returns a dict incl. beta/CI/p and,
    for the inversion, the intercept+slope on log(dose) at the band's median drawn Ø."""
    out = {"form": "log-dose OLS  depth ~ 1 + log(dose) [+ drawn_Ø]", "ok": False}
    X, y, names = _design_log_dose(g, with_diameter=True)
    if X.shape[0] < X.shape[1] + 2:
        out["msg"] = f"n={X.shape[0]} too few for a {X.shape[1]}-term fit"
        return out
    try:
        beta, ci, r2, adj, n, p, pv = _ols_fit(X, y, alpha)
        # effective intercept at the band's median drawn Ø (so the inversion is at a real geometry)
        b0 = beta[0]
        if "drawn_Ø (µm)" in names:
            b0 = beta[0] + beta[names.index("drawn_Ø (µm)")] * float(np.median(g["drawn_diameter_um"]))
        ss_res = float(np.sum((y - X @ beta) ** 2))
        out.update(names=names, beta=beta, ci=ci, r2=r2, adj_r2=adj, n=int(n), pvalues=pv,
                   ss_res=ss_res, k_params=X.shape[1], aicc=_aicc(n, ss_res, X.shape[1]),
                   b0_at_med=float(b0), b_logdose=float(beta[1]), ok=True)
    except Exception as e:
        out["msg"] = f"OLS failed: {e}"
    return out


def _design_interaction(g):
    """Centered design for depth ~ 1 + passes + speed + passes:speed; a predictor (and its
    interaction) is dropped when constant within the band, mirroring report._model_design. Also
    returns the centering means so the fit can be evaluated at arbitrary (passes, speed)."""
    names, cols, cen = ["intercept"], [np.ones(len(g))], {}
    means = {}
    for nm, series in (("passes", g["passes"]), ("speed", g["speed"])):
        v = series.to_numpy(float)
        means[nm] = float(v.mean())
        if np.ptp(v) > 1e-9:
            cen[nm] = v - v.mean()
            names.append(nm); cols.append(cen[nm])
    if "passes" in cen and "speed" in cen:
        names.append("passes×speed"); cols.append(cen["passes"] * cen["speed"])
    return np.column_stack(cols), g["depth_um"].to_numpy(float), names, means


def fit_interaction_ols(g, alpha=0.05):
    """depth ~ 1 + passes + speed + passes:speed (centered) via report._ols_fit."""
    out = {"form": "interaction OLS  depth ~ 1 + passes + speed + passes:speed", "ok": False}
    X, y, names, means = _design_interaction(g)
    if X.shape[0] < X.shape[1] + 2:
        out["msg"] = f"n={X.shape[0]} too few for a {X.shape[1]}-term fit"
        return out
    try:
        beta, ci, r2, adj, n, p, pv = _ols_fit(X, y, alpha)
        ss_res = float(np.sum((y - X @ beta) ** 2))
        out.update(names=names, beta=beta, ci=ci, r2=r2, adj_r2=adj, n=int(n), pvalues=pv,
                   ss_res=ss_res, k_params=X.shape[1], aicc=_aicc(n, ss_res, X.shape[1]),
                   means=means, ok=True)
    except Exception as e:
        out["msg"] = f"OLS failed: {e}"
    return out


def choose_recommended(sat, logd, inter):
    """Choose the smallest finite AICc among fits with meaningful explanatory signal.

    Returning ``None`` deliberately suppresses inversion when every candidate is uninformative.
    """
    candidates = []
    for key, fit in (("saturating", sat), ("log-dose", logd), ("interaction", inter)):
        if not fit.get("ok") or not np.isfinite(fit.get("aicc", np.inf)):
            continue
        quality = fit.get("r2", np.nan) if key == "saturating" else fit.get("adj_r2", np.nan)
        if not np.isfinite(quality) or quality < MIN_PREDICTIVE_R2:
            continue
        candidates.append((float(fit["aicc"]), key))
    if not candidates:
        return (None, f"no candidate has finite AICc and predictive R² >= {MIN_PREDICTIVE_R2:.2f}; "
                "inverse recommendation suppressed")
    aicc, key = min(candidates)
    return key, f"lowest AICc among acceptable independent-cell fits ({aicc:.2f})"


# ============================================================= cross-sample pooling == #
def fit_mixedlm(g, alpha=0.05):
    """Pooled CIs across samples: MixedLM depth ~ log(dose) with a random intercept per sample, so
    cross-wafer/sample baseline differences are modelled after array rows have already been
    collapsed to independent cell-band medians. Falls back to OLS with a sample fixed factor.
    Requires ≥2 samples. Returns a dict with the fixed effects + CI, sample variance, per-sample
    residual summary, and which path was used."""
    out = {"ok": False}
    nsamp = g["sample"].nunique()
    if nsamp < 2:
        out["msg"] = "single sample in this band — no cross-sample pooling (per-band fits above stand)"
        return out
    g = g.assign(_logdose=np.log(g["dose_ratio"].to_numpy(float)))
    try:
        import statsmodels.formula.api as smf
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                 # convergence chatter -> we report ok/fallback
            md = smf.mixedlm("depth_um ~ _logdose", data=g, groups=g["sample"])
            r = md.fit(method="lbfgs", disp=False)
        if not r.converged:
            raise RuntimeError("MixedLM did not converge")
        ci = r.conf_int(alpha=alpha)
        fe = {}
        for nm in ("Intercept", "_logdose"):
            fe[nm] = (float(r.params[nm]), float(ci.loc[nm, 0]), float(ci.loc[nm, 1]),
                      float(r.pvalues[nm]))
        g = g.assign(_resid=g["depth_um"].to_numpy(float) - r.fittedvalues.to_numpy(float))
        per = g.groupby("sample")["_resid"].agg(["count", "mean", "std"])
        out.update(path="MixedLM (random intercept | sample)", fe=fe,
                   group_var=float(r.cov_re.iloc[0, 0]), per_sample=per, ok=True)
        return out
    except Exception as e:
        out["msg_mixed"] = f"MixedLM fallback (reason: {e})"
    # fallback: OLS with sample dummies (sample as a fixed factor)
    try:
        dummies = pd.get_dummies(g["sample"], prefix="s", drop_first=True).astype(float)
        X = np.column_stack([np.ones(len(g)), g["_logdose"].to_numpy(float), dummies.to_numpy()])
        beta, cib, r2, adj, n, p, pv = _ols_fit(X, g["depth_um"].to_numpy(float), alpha)
        out.update(path="OLS + sample fixed factor (MixedLM fallback)", ok=True,
                   fe={"Intercept": (float(beta[0]), float(cib[0, 0]), float(cib[0, 1]), float(pv[0])),
                       "_logdose": (float(beta[1]), float(cib[1, 0]), float(cib[1, 1]), float(pv[1]))},
                   r2=float(r2), n=int(n))
        pred = X @ beta
        g = g.assign(_resid=g["depth_um"].to_numpy(float) - pred)
        out["per_sample"] = g.groupby("sample")["_resid"].agg(["count", "mean", "std"])
    except Exception as e:
        out["msg"] = f"pooled fit failed entirely: {e}"
    return out


# ==================================================================== inversion == #
def invert_target(rec_key, sat, logd, inter, target, alpha=0.05):
    """Dose that yields ``target`` depth under the recommended model, + a (1−alpha) interval.

    Saturating: dose* = −ln(1 − target/a)/k (defined only when target < plateau a); the interval
    comes from a seeded Monte-Carlo over the (a,k) covariance -> percentiles of dose*.
    Log-dose OLS: dose* = exp((target − b0)/b1) (point estimate; inverse CI unavailable).
    Interaction: depth depends on passes and speed SEPARATELY (not through dose alone), so there is
    no single dose* -> point to the (passes, speed) target contour on the heatmap.
    Returns (dose_star, lo, hi, note)."""
    ci_pct = 100 * (1 - alpha)
    if rec_key == "saturating" and sat.get("ok"):
        a, k = sat["a"], sat["k"]
        if target >= a:
            return (float("nan"), float("nan"), float("nan"),
                    f"target {target:g} ≥ plateau a={a:.1f} µm — unreachable at this geometry")
        dstar = -np.log1p(-target / a) / k
        rng = np.random.default_rng(_MC_SEED)
        draws = rng.multivariate_normal(sat["beta"], sat["cov"], size=4000)
        good = (draws[:, 0] > target) & (draws[:, 1] > 0)
        ds = -np.log1p(-target / draws[good, 0]) / draws[good, 1]
        ds = ds[np.isfinite(ds)]
        lo, hi = ((float(np.percentile(ds, 100 * alpha / 2)),
                   float(np.percentile(ds, 100 * (1 - alpha / 2)))) if ds.size > 50
                  else (float("nan"), float("nan")))
        return float(dstar), lo, hi, f"saturating inversion ({ci_pct:g}% inverse-mean CI via param-covariance MC)"
    if rec_key == "log-dose" and logd.get("ok"):
        b0, b1 = logd["b0_at_med"], logd["b_logdose"]
        if b1 == 0:
            return float("nan"), float("nan"), float("nan"), "log-dose slope 0 — not invertible"
        dstar = float(np.exp((target - b0) / b1))
        return dstar, float("nan"), float("nan"), "log-dose inversion (point; inverse CI unavailable)"
    if rec_key == "interaction" and inter.get("ok"):
        return (float("nan"), float("nan"), float("nan"),
                "interaction model: depth depends on passes & speed separately — read the "
                "(passes, speed) target contour off depth_heatmap.png (no single dose)")
    return float("nan"), float("nan"), float("nan"), "recommended model not invertible"


def _predict(rec_key, sat, logd, inter, dose=None, passes=None, speed=None, drawn=None):
    """Predicted depth from the recommended model. Saturating and log-dose use dose = passes/speed;
    interaction uses passes & speed directly (rebuilt from the stored centered coefficients + means).
    Returns an array shaped like the broadcast of the relevant inputs."""
    if rec_key == "saturating" and sat.get("ok"):
        return _sat(np.asarray(dose, float), sat["a"], sat["k"])
    if rec_key == "log-dose" and logd.get("ok"):
        b0, b1 = logd["b0_at_med"], logd["b_logdose"]
        return b0 + b1 * np.log(np.asarray(dose, float))
    if rec_key == "interaction" and inter.get("ok"):
        P, S = np.asarray(passes, float), np.asarray(speed, float)
        mp, ms = inter["means"]["passes"], inter["means"]["speed"]
        val = np.zeros(np.shape(P + S))                  # broadcast shape of the (P, S) grid
        term = {"intercept": 1.0, "passes": P - mp, "speed": S - ms,
                "passes×speed": (P - mp) * (S - ms)}
        for nm, b in zip(inter["names"], inter["beta"]):
            val = val + b * term[nm]
        return val
    return np.full(np.shape(dose), np.nan)


# ======================================================================= driver == #
def calibrate_depth(results_dir=DEF_RESULTS, out_dir=None, include=None, exclude=None,
                    targets=(DEF_TARGET_UM,), max_debris=DEF_MAX_DEBRIS, drop_shallow=False,
                    alpha=0.05, band_defs_path=None, cell_filters=None,
                    allow_legacy_qc=False):
    """Full calibration: discover -> pool -> gate -> (optionally re-bin into user bands) -> per-band
    fits + pooled MixedLM -> write depth_calibration.txt + the three figures. When ``band_defs_path``
    names a band-definitions file, rows are re-binned by drawn Ø into those bands; otherwise the
    measurements ``band`` column is used. Returns the per-band result dict (for tests)."""
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir / OUT_NAME
    targets = list(dict.fromkeys(float(t) for t in targets))   # de-dupe, preserve order

    samples = discover_samples(results_dir, out_dir, include, exclude)
    if not samples:
        raise SystemExit(f"No samples with {MEAS_REL} found under {results_dir} "
                         f"(include={include}, exclude={exclude}).")
    print(f"Pooling {len(samples)} sample(s): {', '.join(n for n, _ in samples)}")
    df = load_pooled(samples, cell_filters=cell_filters)
    if not len(df):
        raise SystemExit("No rows loaded from the selected samples' measurements.csv "
                         "(every file was empty or unreadable).")
    try:
        gated, gate_rep = apply_gates(
            df, max_debris=max_debris, drop_shallow=drop_shallow,
            allow_legacy_qc=allow_legacy_qc,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from None
    for ln in _gate_report_lines(gate_rep):
        print(ln)
    if not len(gated):
        raise SystemExit("No rows survived the quality gates — nothing to fit.")

    band_defs = load_band_defs(band_defs_path)
    if band_defs:
        gated, band_meta, band_lines = assign_bands(gated, band_defs)
        for ln in band_lines:
            print(ln)
        if not len(gated):
            raise SystemExit("No rows fell inside the defined band ranges — check the band definitions.")
    else:                                                # fall back to the DXF band column
        band_meta, band_lines = None, None
        gated["band"] = gated["band"].astype(int)

    try:
        units = aggregate_cell_bands(gated, allow_legacy_qc=allow_legacy_qc)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    print(f"Independent analysis units: {len(units)} cell-band medians from "
          f"{len(gated)} array rows.")

    per_band = {}
    for b, g in _band_groups(units, band_meta):
        sat = fit_saturating(g["dose_ratio"], g["depth_um"], alpha=alpha)
        logd = fit_log_dose_ols(g, alpha=alpha)
        inter = fit_interaction_ols(g, alpha=alpha)
        rec_key, rec_why = choose_recommended(sat, logd, inter)
        mixed = fit_mixedlm(g, alpha=alpha)
        per_band[b] = dict(g=g, sat=sat, logd=logd, inter=inter, rec_key=rec_key,
                           rec_why=rec_why, mixed=mixed, label=_fmt_band(b, g, band_meta))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(out_dir, samples, gate_rep, units, per_band, targets, alpha,
                 max_debris, drop_shallow, band_lines=band_lines, cell_filters=cell_filters,
                 array_row_count=len(gated), allow_legacy_qc=allow_legacy_qc)
    # The report is the primary deliverable; isolate each figure so one bad panel can't abort the
    # whole run (and the already-written report + other figures survive).
    for fn in (fig_depth_vs_dose, fig_parity, fig_heatmap):
        try:
            fn(out_dir, per_band, targets)
        except Exception as e:                           # pragma: no cover - defensive
            print(f"WARNING: {fn.__name__} failed ({e}); other outputs are unaffected.")
    print(f"\nWrote depth calibration -> {out_dir}")
    return per_band


# ============================================================== report writer == #
def _band_groups(units, band_meta=None):
    """Yield calibration families without aliasing local band numbers across designs.

    User band definitions already include a required pitch, so their integer ids are globally
    meaningful. Without definitions, a DXF's ``band`` is only local to that design: e.g. band 1
    can mean D50/P100 in one file and D300/P350 in another. The default key therefore includes
    nominal pitch and keeps those geometries separate.
    """
    if band_meta:
        for b in sorted(units["band"].unique()):
            yield b, units[units["band"] == b].copy()
        return
    keyed = units.assign(_pitch_key=units["nominal_pitch_um"].round(6))
    for (b, pitch), g in keyed.groupby(["band", "_pitch_key"], sort=True, dropna=False):
        key = (int(b), float(pitch))
        yield key, g.drop(columns="_pitch_key").copy()


def _fmt_band(b, g, band_meta=None):
    source_band = b[0] if isinstance(b, tuple) else b
    ds = sorted(g["drawn_diameter_um"].dropna().unique())
    drng = f"{ds[0]:g}–{ds[-1]:g}" if ds else "?"
    if band_meta and source_band in band_meta:            # user-defined band: show declared range/pitch
        dmin, dmax, pitch = band_meta[source_band]
        return (f"band {source_band} (Ø {dmin:g}–{dmax:g} µm @ {pitch:g} µm pitch; "
                f"drawn Ø present {drng} µm, n={len(g)})")
    pitch = g["nominal_pitch_um"].dropna().iloc[0] if g["nominal_pitch_um"].notna().any() else float("nan")
    return f"band {source_band} @ {pitch:g} µm pitch (drawn Ø {drng} µm, n={len(g)})"


def _short_band(b):
    return f"{b[0]}@{b[1]:g}" if isinstance(b, tuple) else str(b)


def _coverage_table(g):
    """sample × (passes, speed, dose) coverage so the reader can spot sample↔dose confounding
    (e.g. sample A only low-dose, B only high-dose -> 'sample' would alias dose)."""
    lines = [f"    {'sample':<20} {'n':>3} {'passes':>14} {'speed(mm/s)':>16} {'dose=P/S':>16}"]
    for s, gs in g.groupby("sample"):
        def rng(col, f="g"):
            v = gs[col].dropna()
            return f"{v.min():{f}}–{v.max():{f}}" if len(v) else "—"
        lines.append(f"    {str(s):<20} {len(gs):>3} {rng('passes'):>14} "
                     f"{rng('speed'):>16} {rng('dose_ratio', '.4g'):>16}")
    return lines


def write_report(out_dir, samples, gate_rep, gated, per_band, targets, alpha, max_debris,
                 drop_shallow, band_lines=None, cell_filters=None, array_row_count=None,
                 allow_legacy_qc=False):
    L = []
    ci_lbl = f"{100 * (1 - alpha):g}% CI"                # honour --alpha in every interval label
    L.append("Etch-depth calibration  —  depth = f(passes, speed) conditioned on pin band")
    L.append("=" * 78)
    L.append("Reframed inverse of the diameter model (report.make_diameter_model / change #4):")
    L.append("response = etch depth, predictors = laser dose. Pooled across samples because depth")
    L.append("is a LOCAL differential (pin-top − clean-floor) so per-tile Z offset / tilt cancel.")
    L.append("Bands are NOT pooled together (each band is a distinct pitch/diameter family).")
    L.append("Statistical unit = one sample/cell/band median; array rows within a cell exposure are")
    L.append("technical replicates and are not counted as independent process trials.")
    L.append("")
    L.append(f"Samples pooled ({len(samples)}): " + ", ".join(n for n, _ in samples))
    _applied = {n: s for n, s in (cell_filters or {}).items()
                if s and n in {nm for nm, _ in samples}}
    if _applied:
        L.append("Per-sample cell filters (cell_id selection applied BEFORE pooling; "
                 "'!' = exclude):")
        for n in sorted(_applied):
            L.append(f"    {n}:  {_applied[n]}")
    L.append(f"Targets: {', '.join(f'{t:g}' for t in targets)} µm   ·   "
             f"max debris_fraction {max_debris:g}   ·   drop_shallow={drop_shallow}   ·   "
             f"CI level {100*(1-alpha):g}%")
    L.append(f"QC schema mode: {'LEGACY OVERRIDE (missing QC may pass)' if allow_legacy_qc else 'strict'}")
    L += _gate_report_lines(gate_rep)
    if array_row_count is not None:
        L.append(f"Independent units: {len(gated)} cell-band medians from {array_row_count} gated array rows")
    if band_lines:
        L.append("")
        L += band_lines
    else:
        L.append("Bands: measurements 'band' qualified by nominal pitch (no definitions supplied).")
    L.append("")
    L.append("Extrapolation limits — trust the calibration ONLY inside the box the data cover:")
    L.append(f"    passes {gated['passes'].min():g}–{gated['passes'].max():g}   ·   "
             f"speed {gated['speed'].min():g}–{gated['speed'].max():g} mm/s   ·   "
             f"drawn Ø {gated['drawn_diameter_um'].min():g}–{gated['drawn_diameter_um'].max():g} µm   ·   "
             f"dose {gated['dose_ratio'].min():.4g}–{gated['dose_ratio'].max():.4g}")
    L.append("")

    for b, R in per_band.items():
        g, sat, logd, inter = R["g"], R["sat"], R["logd"], R["inter"]
        L.append("-" * 78)
        L.append(R["label"])
        L.append("")
        # -- the three candidate forms --
        if sat.get("ok"):
            L.append(f"  [saturating]  {sat['form']}   R²={sat['r2']:.3f}  AICc={sat['aicc']:.2f}  (n={sat['n']}, resid σ={sat['resid_sd']:.2f} µm)")
            L.append(f"      plateau a = {sat['a']:8.2f} µm   [{ci_lbl} {sat['a_ci'][0]:.2f}, {sat['a_ci'][1]:.2f}]")
            L.append(f"      rate    k = {sat['k']:8.4g}      [{ci_lbl} {sat['k_ci'][0]:.4g}, {sat['k_ci'][1]:.4g}]  (per dose unit)")
        else:
            L.append(f"  [saturating]  skipped: {sat.get('msg', '?')}")
        if logd.get("ok"):
            L.append(f"  [log-dose OLS]  {logd['form']}   R²={logd['r2']:.3f}  adj-R²={logd['adj_r2']:.3f}  AICc={logd['aicc']:.2f}  (n={logd['n']})")
            for nm, bta, (lo, hi), pv in zip(logd["names"], logd["beta"], logd["ci"], logd["pvalues"]):
                L.append(f"      {nm:>16} = {bta:+10.3f}   [{ci_lbl} {lo:+.3f}, {hi:+.3f}]   p={pv:.2g}")
        else:
            L.append(f"  [log-dose OLS]  skipped: {logd.get('msg', '?')}")
        if inter.get("ok"):
            L.append(f"  [interaction OLS]  {inter['form']}   R²={inter['r2']:.3f}  adj-R²={inter['adj_r2']:.3f}  AICc={inter['aicc']:.2f}  (n={inter['n']})")
            for nm, bta, (lo, hi), pv in zip(inter["names"], inter["beta"], inter["ci"], inter["pvalues"]):
                L.append(f"      {nm:>16} = {bta:+10.4g}   [{ci_lbl} {lo:+.4g}, {hi:+.4g}]   p={pv:.2g}")
        else:
            L.append(f"  [interaction OLS]  skipped: {inter.get('msg', '?')}")
        L.append("")
        L.append(f"  >> recommended form: {R['rec_key'] or 'none'}  ({R['rec_why']})")
        L.append("")
        # -- pooled cross-sample fit --
        mx = R["mixed"]
        if mx.get("ok"):
            L.append(f"  [pooled] {mx['path']}:")
            for nm, (est, lo, hi, pv) in mx["fe"].items():
                L.append(f"      {nm:>16} = {est:+10.3f}   [{ci_lbl} {lo:+.3f}, {hi:+.3f}]   p={pv:.2g}")
            if "group_var" in mx:
                L.append(f"      sample random-intercept variance = {mx['group_var']:.3f} µm²  "
                         f"(σ_sample = {np.sqrt(max(mx['group_var'],0)):.2f} µm)")
            if "per_sample" in mx:
                L.append("      per-sample residual (measured − pooled fit):")
                for s, row in mx["per_sample"].iterrows():
                    sd = row['std'] if np.isfinite(row['std']) else float('nan')
                    L.append(f"          {str(s):<20} n={int(row['count']):>3}  "
                             f"mean {row['mean']:+.2f} µm  σ {sd:.2f} µm")
        else:
            L.append(f"  [pooled] {mx.get('msg') or mx.get('msg_mixed') or 'not available'}")
        L.append("")
        # -- confounding coverage table --
        L.append("  sample × dose coverage (watch for a sample that only spans one end of the dose):")
        L += _coverage_table(g)
        L.append("")
        # -- inversion per target --
        L.append("  inversion — dose (and one (passes, speed) example) to hit each target depth:")
        speeds = sorted(g["speed"].dropna().unique())
        s_ref = float(np.median(speeds)) if speeds else float("nan")
        for t in targets:
            dstar, lo, hi, note = invert_target(R["rec_key"], sat, logd, inter, t, alpha)
            if np.isfinite(dstar):
                pi = f"  [dose {ci_lbl} {lo:.4g}, {hi:.4g}]" if np.isfinite(lo) else ""
                ex = (f"  e.g. at {s_ref:g} mm/s -> {dstar * s_ref:.0f} passes"
                      if np.isfinite(s_ref) else "")
                L.append(f"      {t:>5g} µm: dose* = {dstar:.4g}{pi}{ex}   ({note})")
            else:
                L.append(f"      {t:>5g} µm: {note}")
        L.append("")

    text = "\n".join(L)
    (out_dir / "depth_calibration.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


# ================================================================= the figures == #
def _sample_colors(per_band):
    samples = sorted({s for R in per_band.values() for s in R["g"]["sample"].unique()})
    cmap = plt.get_cmap("tab10")
    return {s: cmap(i % 10) for i, s in enumerate(samples)}


def _grid(n):
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def fig_depth_vs_dose(out_dir, per_band, targets):
    """Per-band depth vs dose with the recommended dose-only curve and target lines."""
    bands = list(per_band)
    nrows, ncols = _grid(len(bands))
    colors = _sample_colors(per_band)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False)
    for idx, b in enumerate(bands):
        ax = axes[idx // ncols][idx % ncols]
        R = per_band[b]; g = R["g"]; sat = R["sat"]; rk = R["rec_key"]
        for s, gs in g.groupby("sample"):
            ax.plot(gs["dose_ratio"], gs["depth_um"], "o", ms=6, mew=0.5, mec="white",
                    color=colors[s], ls="none", label=str(s), alpha=0.85)
        if rk == "saturating" and sat.get("ok"):
            xr = np.linspace(g["dose_ratio"].min(), g["dose_ratio"].max(), 200)
            yc = _sat(xr, sat["a"], sat["k"])
            ax.plot(xr, yc, "k-", lw=1.8, label=f"saturating (R²={sat['r2']:.2f})")
            band = 1.96 * sat["resid_sd"]
            ax.fill_between(xr, yc - band, yc + band, color="k", alpha=0.12, lw=0)
            ax.axhline(sat["a"], color="0.5", ls=":", lw=1)
            ax.text(0.98, 0.06, f"plateau a≈{sat['a']:.0f} µm", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="0.35")
        elif rk == "log-dose" and R["logd"].get("ok"):
            xr = np.linspace(g["dose_ratio"].min(), g["dose_ratio"].max(), 200)
            yc = _predict("log-dose", sat, R["logd"], R["inter"], dose=xr)
            ax.plot(xr, yc, "k-", lw=1.8,
                    label=f"log-dose (adj-R²={R['logd']['adj_r2']:.2f})")
        for t in targets:
            ax.axhline(t, color="crimson", ls="--", lw=1)
            # x in axes fraction, y in data units -> label sits just inside the left edge (the
            # get_yaxis_transform idiom used in run_sample.make_param_summary)
            ax.text(0.01, t, f" {t:g} µm", transform=ax.get_yaxis_transform(),
                    color="crimson", fontsize=7, va="bottom")
        ax.set_xscale("log")
        ax.set_title(R["label"], fontsize=9)
        ax.set_xlabel("dose = passes / speed (log axis)")
        ax.set_ylabel("etch depth (µm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for idx in range(len(bands), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Etch depth vs dose per band", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "depth_vs_dose.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p}")


def fig_parity(out_dir, per_band, targets=None):        # targets unused; accepted for a uniform call loop
    """Measured-vs-predicted parity + residuals (reuses the make_diameter_model two-panel layout),
    coloured by band, using each band's recommended model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    cmap = plt.get_cmap("tab10")
    lo_hi = []
    for i, (b, R) in enumerate(per_band.items()):
        g, rk = R["g"], R["rec_key"]
        if rk not in ("saturating", "log-dose", "interaction"):
            continue
        pred = _predict(rk, R["sat"], R["logd"], R["inter"],
                        dose=g["dose_ratio"].to_numpy(float),
                        passes=g["passes"].to_numpy(float), speed=g["speed"].to_numpy(float))
        y = g["depth_um"].to_numpy(float)
        m = np.isfinite(pred) & np.isfinite(y)
        if not m.any():
            continue
        c = cmap(i % 10)
        axes[0].scatter(pred[m], y[m], s=18, color=c, alpha=0.65, edgecolors="white",
                        linewidths=0.3, label=f"band {_short_band(b)} ({rk})")
        axes[1].scatter(pred[m], y[m] - pred[m], s=18, color=c, alpha=0.65, edgecolors="white",
                        linewidths=0.3)
        lo_hi += [float(np.min([pred[m].min(), y[m].min()])), float(np.max([pred[m].max(), y[m].max()]))]
    if lo_hi:
        lim = [min(lo_hi), max(lo_hi)]
        axes[0].plot(lim, lim, "k--", lw=0.8, alpha=0.6)
    axes[0].set_xlabel("model-predicted depth (µm)"); axes[0].set_ylabel("measured depth (µm)")
    axes[0].set_title("Depth model parity")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("model-predicted depth (µm)"); axes[1].set_ylabel("residual measured−predicted (µm)")
    axes[1].set_title("residuals"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "depth_parity.png"
    fig.savefig(p, dpi=170); plt.close(fig)
    print(f"Wrote {p}")


def fig_heatmap(out_dir, per_band, targets):
    """Per-band passes × speed heatmap of predicted depth (recommended model), with the target-depth
    contour(s) overlaid — read off which (passes, speed) hits a target. Actual data points marked."""
    # A passes×speed heatmap needs BOTH swept: a band with a single distinct passes (or speed) value
    # collapses the grid to a 1-D strip, and ax.contour requires a >=2x2 grid -> would crash. Skip
    # (with a note) those bands and any non-predictable model.
    invertible, skipped = [], []
    for b, R in per_band.items():
        g = R["g"]
        if R["rec_key"] not in ("saturating", "log-dose", "interaction"):
            continue
        if g["passes"].dropna().nunique() < 2 or g["speed"].dropna().nunique() < 2:
            skipped.append(b)
        else:
            invertible.append(b)
    if skipped:
        skipped_label = [_short_band(b) for b in skipped]
        print(f"heatmap: bands {skipped_label} have a single distinct passes or speed (a P×S heatmap "
              f"needs both swept) -> omitted; see depth_vs_dose.png for their dose response.")
    if not invertible:
        print("No band varies both passes and speed -> skipping heatmap.")
        return
    levels = sorted({float(t) for t in targets})         # strictly increasing (contour requires it)
    nrows, ncols = _grid(len(invertible))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.0 * nrows), squeeze=False)
    for idx, b in enumerate(invertible):
        ax = axes[idx // ncols][idx % ncols]
        R = per_band[b]; g = R["g"]; rk = R["rec_key"]
        p_grid = np.linspace(g["passes"].min(), g["passes"].max(), 120)
        s_grid = np.linspace(g["speed"].min(), g["speed"].max(), 120)
        P, S = np.meshgrid(p_grid, s_grid)
        Z = _predict(rk, R["sat"], R["logd"], R["inter"], dose=P / S, passes=P, speed=S)
        pcm = ax.pcolormesh(s_grid, p_grid, Z.T, shading="auto", cmap="viridis")
        fig.colorbar(pcm, ax=ax, label="predicted depth (µm)")
        cs = ax.contour(s_grid, p_grid, Z.T, levels=levels, colors="white", linewidths=1.5)
        ax.clabel(cs, fmt=lambda v: f"{v:g} µm", fontsize=8)
        ax.plot(g["speed"], g["passes"], "o", mfc="none", mec="crimson", ms=7, mew=1.2,
                label="measured cells")
        ax.set_xlabel("scan speed (mm/s)"); ax.set_ylabel("passes")
        ax.set_title(f"{R['label']} — predicted depth", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    for idx in range(len(invertible), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Predicted etch depth over passes × speed, with target-depth contours", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "depth_heatmap.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p}")


# ========================================================================= CLI == #
def _csv_list(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def main(argv=None):
    # The report lines contain µ/Ø/× -> force UTF-8 so a cp1252 Windows console can't crash on print.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Cross-sample etch-depth calibration (depth = f(passes, speed | band)).")
    ap.add_argument("--results", default=str(DEF_RESULTS), help="Results root holding <sample>/legacy/measurements.csv")
    ap.add_argument("--out", default=None, help=f"output folder (default: <results>/'{OUT_NAME}')")
    # nargs='*' -> each sample name is its own token, so names containing commas OR spaces (real
    # sample folders have spaces, e.g. '071826 D300 3x3') are passed intact; no delimiter to split on.
    ap.add_argument("--include", nargs="*", default=None, help="sample folder names to include "
                    "(space-separated; quote names with spaces)")
    ap.add_argument("--exclude", nargs="*", default=None, help="sample folder names to exclude")
    ap.add_argument("--targets", type=lambda s: [float(x) for x in _csv_list(s)], default=[DEF_TARGET_UM],
                    help="target depth(s) in µm, comma-separated (default 55)")
    ap.add_argument("--max-debris", type=float, default=DEF_MAX_DEBRIS, help="drop rows with debris_fraction above this")
    ap.add_argument("--drop-shallow", action="store_true", help="also drop 'shallow' (<3 µm) rows")
    ap.add_argument("--allow-legacy-qc", action="store_true",
                    help="explicitly allow old measurements lacking reliable/debris/cell QC fields")
    ap.add_argument("--alpha", type=float, default=0.05, help="CI significance level (default 0.05 -> 95%%)")
    ap.add_argument("--bands", default=None, help="band-definitions file: each row 'min_Ø, max_Ø, pitch' "
                    "(µm); omit to use the measurements 'band' column")
    ap.add_argument("--cell-filters", default=None, help="JSON file mapping sample folder name -> "
                    "cell-id spec. A spec lists cell_ids to KEEP (e.g. '1-5, 8, 12-16'); a leading "
                    "'!' EXCLUDES them (e.g. '!3, 7'). Omit to use every cell of every sample.")
    args = ap.parse_args(argv)
    cell_filters = None
    if args.cell_filters:
        try:                                             # utf-8-sig tolerates a BOM (hand-edited files)
            cell_filters = json.loads(Path(args.cell_filters).read_text(encoding="utf-8-sig"))
        except Exception as e:
            raise SystemExit(f"Could not read --cell-filters JSON '{args.cell_filters}': {e}")
        if not isinstance(cell_filters, dict):
            raise SystemExit('--cell-filters JSON must be an object, e.g. {"sample name": "1-5, 8"}.')
    calibrate_depth(results_dir=args.results, out_dir=args.out, include=args.include,
                    exclude=args.exclude, targets=args.targets, max_debris=args.max_debris,
                    drop_shallow=args.drop_shallow, alpha=args.alpha, band_defs_path=args.bands,
                    cell_filters=cell_filters, allow_legacy_qc=args.allow_legacy_qc)


if __name__ == "__main__":
    main()
