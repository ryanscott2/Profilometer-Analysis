"""
Register a DXF unit-cell layout onto a Keyence VK4 scan.

The profilometer scan and the CAD design share a known pixel scale (the VK4 stores
X/Y micrometres-per-pixel), so the only unknowns are a **translation** (where each unit
cell sits in the scan), a possible **y-axis flip** (image row order vs CAD y-up), and a
small **rotation** (stage misalignment; assumed ~0 and left as an optional refinement).

Strategy (in order, each falls back to the next)
------------------------------------------------
1. **Alignment-marker detection** — every unit cell carries a ~200 um square drawn at its
   bottom-left. We matched-filter a square-outline template against the *edge magnitude*
   of the scan (edges show up whether the marker is milled proud or recessed, so this is
   polarity-agnostic). For a tiled grid we take the N strongest, well-separated peaks and
   order them left->right, bottom->top.
2. **Lattice refinement** — with the marker giving a coarse origin, we cross-correlate the
   rasterised design pin pattern against the scan's pin mask in a window and snap the origin
   to the best pin-lattice overlap. This also picks the y-flip that scores highest.
3. **Manual override** — a caller-supplied (origin_col,row[,y_up,rotation]) skips detection.

Everything downstream (``extract.py``) consumes a :class:`CellPlacement`, whose
``dxf_to_px`` maps a marker-relative CAD coordinate (um) to a scan pixel (col,row).

Pixel convention: arrays are indexed ``[row, col]`` as stored in the file. ``y_up=+1``
means increasing CAD-y maps to increasing row index (matplotlib ``origin="lower"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
@dataclass
class CellPlacement:
    """Where one DXF unit cell sits in the scan, and how to map CAD um -> scan px."""

    cell_id: int
    origin_col: float            # scan col of the marker corner (CAD x = 0)
    origin_row: float            # scan row of the marker corner (CAD y = 0)
    x_um_per_px: float
    y_um_per_px: float
    y_up: int = 1                # +1: CAD +y -> +row; -1: flipped
    x_right: int = 1             # +1: CAD +x -> +col; -1: X-mirrored (Keyence imaging flip)
    rotation_deg: float = 0.0
    score: float = float("nan")  # registration quality (higher = better)
    method: str = ""             # "marker+lattice" / "lattice" / "manual" / "grid+lattice"
    cell_row: int = 0            # design-frame unit-cell index (1 = top), set by register_sample
    cell_col: int = 0            # design-frame unit-cell index (1 = left)

    def dxf_to_px(self, x_um, y_um):
        """CAD (marker-relative, um) -> scan pixel (col, row). Vectorised.

        Supports a left-right reflection (``x_right`` = -1, common: the scan is X-mirrored
        vs the DXF so the bottom-left design marker images at the cell's bottom-right) and a
        top-bottom reflection (``y_up`` = -1), plus a small rotation."""
        x_um = np.asarray(x_um, float)
        y_um = np.asarray(y_um, float)
        col_rel = self.x_right * (x_um / self.x_um_per_px)   # reflect first
        row_rel = self.y_up * (y_um / self.y_um_per_px)
        if self.rotation_deg:                                # then rotate in SCAN space
            th = np.deg2rad(self.rotation_deg)
            c, s = np.cos(th), np.sin(th)
            col = self.origin_col + c * col_rel - s * row_rel
            row = self.origin_row + s * col_rel + c * row_rel
        else:
            col = self.origin_col + col_rel
            row = self.origin_row + row_rel
        return col, row


# --------------------------------------------------------- scan feature maps #
def _level_plane(z, valid, max_fit=200000):
    """Remove first-order tilt using the low (floor) population; returns z - plane.
    The plane is fit on a strided subsample of valid pixels so this stays fast on a large
    stitched sample (the fit needs only a representative floor sample, not every pixel)."""
    ys, xs = np.nonzero(valid)
    n = xs.size
    if n < 50:
        return z - (np.median(z[valid]) if n else 0.0)
    step = max(1, n // max_fit)
    xs_s, ys_s = xs[::step], ys[::step]
    zv = z[ys_s, xs_s]
    fp = zv <= np.percentile(zv, 40.0)                 # floor population
    A = np.column_stack([xs_s[fp].astype(float), ys_s[fp].astype(float),
                         np.ones(int(fp.sum()))])
    coef, *_ = np.linalg.lstsq(A, zv[fp], rcond=None)
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    return z - (coef[0] * xx + coef[1] * yy + coef[2])


def scan_feature(scan):
    """Return (feature, edge_mag, pin_mask, valid).

    feature   : pin-bright map (intensity if present, else height), 0..1-ish
    edge_mag  : |gradient| of the feature (marker/pin edges), polarity-agnostic
    pin_mask  : boolean estimate of pin-top pixels (high feature)

    Intensity is preferred and needs no levelling, so this stays cheap on a large stitched
    sample (no global plane fit over tens of millions of pixels — a single plane would be
    inaccurate across a whole chip anyway; per-crop levelling in extract.py handles tilt).
    """
    valid = scan.height_raw != 0
    have_int = getattr(scan, "intensity", None) is not None
    # feature/edge for MARKER detection: intensity if present (bright marker + pin edges)
    if have_int:
        f = np.where(valid, scan.intensity.astype(np.float64), np.nan)
    else:
        f = np.where(valid, _level_plane(scan.height_um, valid), np.nan)
    lo, hi = np.nanpercentile(f, 2), np.nanpercentile(f, 98)
    feat = np.clip((np.nan_to_num(f, nan=lo) - lo) / (hi - lo + 1e-9), 0, 1)
    gy, gx = np.gradient(feat)
    edge = np.hypot(gx, gy)

    # pin_mask for lattice-overlap: raised pin tops from the leveled HEIGHT (robust across
    # dose and imaging; intensity alone is too noisy to discriminate the pin lattice).
    z0 = _level_plane(scan.height_um, valid)
    zt = np.where(valid, z0, -1e9)
    thr = np.percentile(z0[valid], 70) if valid.any() else 0.0
    pin_mask = valid & (zt > thr)
    return feat, edge, pin_mask, valid, z0


# ------------------------------------------------------------- FFT utilities #
def _fft_xcorr_same(image, template):
    """Cross-correlation (matched filter) of image with a zero-mean template, 'same' size.
    Peak location = best template centre. Uses scipy if available, else numpy FFT."""
    t = template - template.mean()
    try:
        from scipy.signal import fftconvolve
        corr = fftconvolve(image, t[::-1, ::-1], mode="same")
    except Exception:  # pragma: no cover
        H, W = image.shape
        th, tw = t.shape
        fs = (H + th, W + tw)
        F = np.fft.rfft2(image, fs) * np.conj(np.fft.rfft2(t, fs))
        full = np.fft.irfft2(F, fs)
        corr = full[th // 2: th // 2 + H, tw // 2: tw // 2 + W]
    return corr


def _phase_shift(a, b):
    """Translation (drow, dcol) that best aligns b onto a (both same shape), by
    normalised cross-correlation via FFT. Returns integer shift and peak value."""
    a = a - a.mean()
    b = b - b.mean()
    Fa = np.fft.rfft2(a)
    Fb = np.fft.rfft2(b)
    R = Fa * np.conj(Fb)
    denom = np.abs(R)
    R = np.where(denom > 1e-12, R / denom, 0.0)   # phase correlation
    corr = np.fft.irfft2(R, s=a.shape)
    pk = np.unravel_index(np.argmax(corr), corr.shape)
    drow = pk[0] if pk[0] <= a.shape[0] // 2 else pk[0] - a.shape[0]
    dcol = pk[1] if pk[1] <= a.shape[1] // 2 else pk[1] - a.shape[1]
    return int(drow), int(dcol), float(corr[pk])


# ------------------------------------------------------------ marker finding #
def _rect_outline(w_px, h_px, thick=None):
    """Zero-mean rectangular-ring template (width x height in px).

    The alignment marker is physically square, but on non-square-pixel scans it images as a
    rectangle, so the template must match the per-axis pixel extents."""
    w = max(5, int(round(w_px)))
    h = max(5, int(round(h_px)))
    thick = thick or max(2, int(round(0.12 * min(w, h))))
    tpl = np.zeros((h, w))
    tpl[:thick, :] = 1; tpl[-thick:, :] = 1
    tpl[:, :thick] = 1; tpl[:, -thick:] = 1
    return tpl


def detect_markers(edge_mag, marker_px_x, marker_px_y, expect_n=1, min_sep_px=None,
                   rel_snr=0.3):
    """Find alignment-marker centres in an edge map.

    ``marker_px_x``/``marker_px_y`` are the marker's per-axis pixel extents (equal on
    square-pixel scans). Up to ``expect_n`` strongest well-separated peaks are found, then
    those far weaker than the strongest (score < ``rel_snr`` * top) are dropped -- so asking
    for more cells than the scan holds does not manufacture markers out of pin-array edges.
    Returns a list of dicts ``{row, col, score, snr}`` ordered left->right, bottom->top.
    """
    tpl = _rect_outline(marker_px_x, marker_px_y)
    corr = _fft_xcorr_same(edge_mag, tpl)
    corr = np.nan_to_num(corr, nan=0.0)

    min_sep = int(min_sep_px or max(0.75 * (marker_px_x + marker_px_y), 20))
    flat = corr.copy()
    peaks = []
    for _ in range(max(1, expect_n)):
        idx = np.argmax(flat)
        r, c = np.unravel_index(idx, flat.shape)
        peaks.append(dict(row=int(r), col=int(c), score=float(corr[r, c])))
        r0, r1 = max(0, r - min_sep), min(flat.shape[0], r + min_sep)
        c0, c1 = max(0, c - min_sep), min(flat.shape[1], c + min_sep)
        flat[r0:r1, c0:c1] = -np.inf     # suppress neighbourhood, find the next one

    sd = float(np.std(corr)) + 1e-9
    for p in peaks:
        p["snr"] = p["score"] / sd

    # keep only peaks comparable to the strongest (real markers correlate far above stray
    # pin-array edges); always keep at least the top peak.
    top = max(p["score"] for p in peaks)
    peaks = [p for p in peaks if p["score"] >= rel_snr * top] or peaks[:1]

    # order tiled markers row-major: bottom->top (ascending row = bottom with origin=lower),
    # left->right within a row. Always sort (markers sharing a row must still order by col).
    if len(peaks) > 1:
        ytol = max(0.5 * min_sep, 1.0)
        peaks.sort(key=lambda p: (round(p["row"] / ytol), p["col"]))
    return peaks


# -------------------------------------------------------- design rasteriser #
def rasterize_cell_pins(cell, x_um_per_px, y_um_per_px, shape, origin,
                        y_up=1, x_right=1, rot_deg=0.0):
    """Binary pin-disk mask of one cell placed at ``origin`` (col,row) in a ``shape`` canvas.
    Reflection then scan-space rotation, matching ``CellPlacement.dxf_to_px``."""
    mask = np.zeros(shape, bool)
    H, W = shape
    oc, orow = origin
    th = np.deg2rad(rot_deg)
    c, s = np.cos(th), np.sin(th)
    for a in cell.arrays:
        rx = 0.5 * a.diameter_um / x_um_per_px      # per-axis pin radius (ellipse if px non-square)
        ry = 0.5 * a.diameter_um / y_um_per_px
        rrx, rry = int(np.ceil(rx)), int(np.ceil(ry))
        for (x_um, y_um) in a.centers_um:
            col_rel = x_right * (x_um / x_um_per_px)
            row_rel = y_up * (y_um / y_um_per_px)
            ci = int(round(oc + c * col_rel - s * row_rel))
            ri = int(round(orow + s * col_rel + c * row_rel))
            for dr in range(-rry, rry + 1):
                for dc in range(-rrx, rrx + 1):
                    if (dc / rx) ** 2 + (dr / ry) ** 2 <= 1.0:
                        yy, xx = ri + dr, ci + dc
                        if 0 <= yy < H and 0 <= xx < W:
                            mask[yy, xx] = True
    return mask


def _cell_pin_extent_px(cell, xppx, yppx):
    """Pin extent (col_min, col_max, row_min, row_max) from the marker origin, in px."""
    xs = np.concatenate([a.centers_um[:, 0] for a in cell.arrays])
    ys = np.concatenate([a.centers_um[:, 1] for a in cell.arrays])
    return (xs.min() / xppx, xs.max() / xppx, ys.min() / yppx, ys.max() / yppx)


def _overlap_score(win_bool, design_bool):
    """Fraction of design pin pixels that land on real pin pixels (0..1)."""
    ds = design_bool.sum()
    if ds < 1:
        return 0.0
    return float(np.logical_and(win_bool, design_bool).sum()) / float(ds)


# ------------------------------------------------------------- registration #
def _refine_origin(pin_mask, cell, origin, xppx, yppx, y_up, x_right=1, rot_deg=0.0,
                   search_px=60, max_shift_px=None):
    """Snap the origin to the best design/scan pin-lattice overlap.

    Returns (refined_origin, overlap_score) where overlap_score is in 0..1 (fraction of
    design pins sitting on real pins) so it is directly comparable across reflection options.
    ``max_shift_px`` clamps the correction: on a periodic pin lattice the phase peak can sit
    a full pitch away from the true (marker-anchored) position, so when the anchor is trusted
    (marker-based), pass a value below half the pin pitch to keep the refine alias-safe.
    """
    if not cell.arrays:                         # degenerate template: nothing to refine on
        return origin, float("nan")
    cxmin, cxmax, cymin, cymax = _cell_pin_extent_px(cell, xppx, yppx)
    oc, orow = origin
    th = np.deg2rad(rot_deg); cth, sth = np.cos(th), np.sin(th)
    rr_v, cc_v = [], []                         # tight window around the footprint corners
    for cx in (cxmin, cxmax):
        for cy in (cymin, cymax):
            col_rel, row_rel = x_right * cx, y_up * cy
            cc_v.append(oc + cth * col_rel - sth * row_rel)
            rr_v.append(orow + sth * col_rel + cth * row_rel)
    r_lo, r_hi = int(min(rr_v) - search_px), int(max(rr_v) + search_px + 1)
    c_lo, c_hi = int(min(cc_v) - search_px), int(max(cc_v) + search_px + 1)
    r0, r1 = max(0, r_lo), min(pin_mask.shape[0], r_hi)
    c0, c1 = max(0, c_lo), min(pin_mask.shape[1], c_hi)
    if r1 - r0 < 16 or c1 - c0 < 16:
        return origin, float("nan")
    win = pin_mask[r0:r1, c0:c1].astype(float)
    design = rasterize_cell_pins(cell, xppx, yppx, win.shape,
                                 (oc - c0, orow - r0), y_up, x_right, rot_deg).astype(float)
    if design.sum() < 5 or win.sum() < 5:
        return origin, float("nan")
    drow, dcol, _pk = _phase_shift(win, design)
    if max_shift_px is not None and (drow * drow + dcol * dcol) > max_shift_px ** 2:
        drow, dcol = 0, 0                        # phase peak aliased a pitch away; trust anchor
    new_origin = (oc + dcol, orow + drow)
    design2 = rasterize_cell_pins(cell, xppx, yppx, win.shape,
                                  (new_origin[0] - c0, new_origin[1] - r0), y_up,
                                  x_right, rot_deg).astype(bool)
    return new_origin, _overlap_score(win.astype(bool), design2)


def register_scan(scan, template, n_cells=1, overrides=None, refine=True,
                  resolve_yflip=True, resolve_xflip=True,
                  cell_pitch_mm=(float("nan"),) * 2, min_overlap=0.12):
    """Register ``n_cells`` copies of a single unit-cell ``template`` onto ``scan``.

    The DXF describes ONE unit cell; a scan may hold several tiled copies of it. We detect
    up to ``n_cells`` alignment markers, place the template at each, resolve the y-flip, and
    number the cells in CAD order (left->right, bottom->top) so cell_id matches the CSV.
    ``n_cells`` normally comes from the user's CSV (rows per file).

    overrides : optional dict ``cell_id -> {origin_col, origin_row, y_up?, rotation_deg?}``
                to bypass detection for specific cells (1-based cell_id).
    min_overlap : reject a detected placement whose refined pin-overlap is below this (a
                spurious peak, e.g. n_cells overstated). Near-duplicate placements are also
                de-duplicated so an extra peak that snaps onto a real cell cannot double-count.
    Returns list[CellPlacement] of length ``n_cells``.
    """
    overrides = overrides or {}
    xppx, yppx = scan.x_um_per_px, scan.y_um_per_px
    feat, edge, pin_mask, valid, z0 = scan_feature(scan)

    marker_um = template.marker_size_um
    if not np.isfinite(marker_um):
        marker_um = 200.0
    marker_px_x = marker_um / xppx
    marker_px_y = marker_um / yppx

    dx_mm, dy_mm = cell_pitch_mm
    if np.isfinite(dx_mm) and np.isfinite(dy_mm):
        min_sep = 0.5 * min(dx_mm * 1000 / xppx, dy_mm * 1000 / yppx)
    else:
        w_um, h_um = template.size_um       # fall back to the cell's own pin span
        min_sep = max(0.6 * min(w_um / xppx, h_um / yppx),
                      marker_px_x + marker_px_y, 40)

    n_detect = max(0, n_cells - len(overrides))
    peaks = detect_markers(edge, marker_px_x, marker_px_y, expect_n=max(1, n_detect),
                           min_sep_px=min_sep) if n_detect > 0 else []

    placements = []
    pk_i = 0
    for cid in range(1, n_cells + 1):
        if cid in overrides:
            o = overrides[cid]
            placements.append(CellPlacement(
                cid, float(o["origin_col"]), float(o["origin_row"]), xppx, yppx,
                y_up=int(o.get("y_up", 1)), x_right=int(o.get("x_right", 1)),
                rotation_deg=float(o.get("rotation_deg", 0.0)),
                score=float("nan"), method="manual"))
            continue

        pk = peaks[pk_i] if pk_i < len(peaks) else None
        pk_i += 1
        if pk is None:
            placements.append(CellPlacement(cid, np.nan, np.nan, xppx, yppx,
                                            method="failed"))
            continue

        best = None                    # resolve reflections by best pin-lattice overlap
        yflips = (1, -1) if resolve_yflip else (1,)
        xflips = (1, -1) if resolve_xflip else (1,)
        for yf in yflips:
            for xf in xflips:
                oc = pk["col"] - xf * 0.5 * marker_px_x
                orow = pk["row"] - yf * 0.5 * marker_px_y
                origin = (oc, orow)
                if refine:
                    origin, ref = _refine_origin(pin_mask, template, origin,
                                                 xppx, yppx, yf, xf)
                    if np.isfinite(ref):
                        score, method = ref, "marker+lattice"   # overlap 0..1, comparable
                    else:
                        score, method = -1.0, "marker"
                else:
                    score, method = pk.get("snr", 0.0), "marker"
                if best is None or score > best[0]:
                    best = (score, origin, yf, xf, method)
        score, origin, yf, xf, method = best
        placements.append(CellPlacement(
            cid, float(origin[0]), float(origin[1]), xppx, yppx,
            y_up=yf, x_right=xf, score=float(score), method=method))

    return _finalize_placements(placements, min_sep, min_overlap,
                                template.size_um[1] / yppx)


def _finalize_placements(placements, min_sep, min_overlap, cell_h_px):
    """Reject low-quality / duplicate detections and (for pure auto runs) number the
    surviving cells in CAD order.

    - A lattice-refined placement below ``min_overlap`` is a spurious peak -> mark failed.
    - Two auto placements closer than ``min_sep`` are the same physical cell (e.g. an extra
      peak that snapped onto a real cell); keep the higher-scoring one, fail the other.
    - With no manual overrides, surviving cells are renumbered 1..k by CAD y (= y_up*row, so
      a y-flipped scan still counts bottom->top) then column, matching the CSV convention.
      When overrides are present the caller's cell_id numbering is preserved.
    """
    def _fail(p):
        p.origin_col = p.origin_row = float("nan"); p.method = "failed"

    for p in placements:                        # quality gate on the refined overlap
        if p.method == "marker+lattice" and np.isfinite(p.score) and p.score < min_overlap:
            _fail(p)

    auto_ok = sorted([p for p in placements
                      if p.method not in ("manual", "failed") and np.isfinite(p.origin_col)],
                     key=lambda p: -(p.score if np.isfinite(p.score) else -1))
    kept = []
    for p in auto_ok:                           # de-duplicate near-coincident detections
        if any((p.origin_col - k.origin_col) ** 2 + (p.origin_row - k.origin_row) ** 2
               < min_sep ** 2 for k in kept):
            _fail(p)
        else:
            kept.append(p)

    if any(p.method == "manual" for p in placements):
        return sorted(placements, key=lambda p: p.cell_id)   # respect user numbering

    survivors = [p for p in placements if np.isfinite(p.origin_col)]
    failed = [p for p in placements if not np.isfinite(p.origin_col)]
    ytol = max(0.5 * cell_h_px, 1.0)
    survivors.sort(key=lambda p: (round(p.y_up * p.origin_row / ytol), p.origin_col))
    out = survivors + failed
    for i, p in enumerate(out, start=1):        # CAD-order cell numbering (matches the CSV)
        p.cell_id = i
    return out


def _cluster_1d(vals, tol):
    """Group indices whose values fall within ``tol`` of the running group; ascending."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    groups = []
    for i in order:
        if groups and vals[i] - groups[-1][-1][1] <= tol:
            groups[-1].append((i, vals[i]))
        else:
            groups.append([(i, vals[i])])
    return groups


def _inbounds_frac(placement, template, W, H):
    """Fraction of a placed cell's pins that fall inside the scan (0..1)."""
    xs = np.concatenate([a.centers_um[:, 0] for a in template.arrays])
    ys = np.concatenate([a.centers_um[:, 1] for a in template.arrays])
    cols, rows = placement.dxf_to_px(xs, ys)
    return float(np.mean((cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)))


def _cell_contrast(z0, valid, template, placement, W, H):
    """Height contrast (um) = pin-top height - trench-floor height, over a placed cell.

    A real ablated cell has pins raised well above an etched floor (contrast = etch depth),
    whereas an un-ablated flat wafer region is uniformly high (contrast ~ 0). This
    discriminates real cells from flat regions that a global height threshold marks as all
    'pin' (giving false lattice overlap)."""
    xs = np.concatenate([a.centers_um[:, 0] for a in template.arrays])
    ys = np.concatenate([a.centers_um[:, 1] for a in template.arrays])
    cols, rows = placement.dxf_to_px(xs, ys)
    ci = np.round(cols).astype(int); ri = np.round(rows).astype(int)
    m = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    if m.sum() < 10:
        return 0.0
    ci, ri = ci[m], ri[m]
    vv = valid[ri, ci]
    if vv.sum() < 10:
        return 0.0
    pin_h = float(np.median(z0[ri[vv], ci[vv]]))
    r0, r1, c0, c1 = ri.min(), ri.max(), ci.min(), ci.max()
    win = z0[r0:r1 + 1, c0:c1 + 1]; wv = valid[r0:r1 + 1, c0:c1 + 1]
    if wv.sum() < 20:
        return 0.0
    floor_h = float(np.percentile(win[wv], 15))          # ablated trench floor
    return pin_h - floor_h


def _assign_grid_indices(cells, Pxpx, Pypx):
    """Assign design-frame (cell_row, cell_col) by clustering origins (col 1 = min design-x
    = left; row 1 = max design-y = top), given the global reflection stored on each cell."""
    if not cells:
        return
    x_right, y_up = cells[0].x_right, cells[0].y_up
    colkey = [x_right * c.origin_col for c in cells]
    rowkey = [y_up * c.origin_row for c in cells]
    col_groups = _cluster_1d(colkey, 0.5 * Pxpx)
    row_groups = _cluster_1d(rowkey, 0.5 * Pypx)
    col_rank = {idx: r for r, g in enumerate(col_groups, 1) for idx, _ in g}
    Nr = len(row_groups)
    row_rank = {idx: (Nr - r + 1) for r, g in enumerate(row_groups, 1) for idx, _ in g}
    for idx, c in enumerate(cells):
        c.cell_col, c.cell_row = col_rank[idx], row_rank[idx]


def _fit_grid(cells):
    """Least-squares fit origin_scan(row,col) = O + (row-1)*rowvec + (col-1)*colvec from
    cells carrying (cell_row, cell_col). Returns (O, rowvec, colvec) in scan px, or None."""
    if len({(c.cell_row, c.cell_col) for c in cells}) < 3:
        return None
    A = np.array([[1.0, c.cell_row - 1, c.cell_col - 1] for c in cells])
    bx = np.linalg.lstsq(A, np.array([c.origin_col for c in cells]), rcond=None)[0]
    by = np.linalg.lstsq(A, np.array([c.origin_row for c in cells]), rcond=None)[0]
    return ((bx[0], by[0]), (bx[1], by[1]), (bx[2], by[2]))


def _estimate_basis(cells, Ppx_x, Ppx_y):
    """Estimate the two cell-step vectors (col-ish ``a``, row-ish ``b``) in scan px from the
    pairwise origin deltas of confident cells. These capture the true pitch AND any small
    sample rotation, so tiling from an anchor lands on every cell. Falls back to axis-aligned
    pitch vectors if too few confident cells."""
    pts = np.array([[c.origin_col, c.origin_row] for c in cells], float)
    a_c, b_c = [], []
    for i in range(len(pts)):
        for k in range(len(pts)):
            if i == k:
                continue
            d = pts[k] - pts[i]
            if 0.75 * Ppx_x <= abs(d[0]) <= 1.25 * Ppx_x and abs(d[1]) <= 0.3 * Ppx_x:
                a_c.append(d if d[0] > 0 else -d)        # normalise to +col direction
            if 0.75 * Ppx_y <= abs(d[1]) <= 1.25 * Ppx_y and abs(d[0]) <= 0.3 * Ppx_y:
                b_c.append(d if d[1] > 0 else -d)        # normalise to +row direction
    a = np.median(a_c, axis=0) if a_c else np.array([Ppx_x, 0.0])
    b = np.median(b_c, axis=0) if b_c else np.array([0.0, Ppx_y])
    return a, b


def _marker_grid_rotation(peaks, Ppx):
    """Global sample rotation (deg) from the angle of marker-center pairs one cell apart."""
    mc = np.array([[p["col"], p["row"]] for p in peaks], float)
    angs = []
    for i in range(len(mc)):
        for k in range(len(mc)):
            d = mc[k] - mc[i]
            if 0.8 * Ppx <= abs(d[0]) <= 1.2 * Ppx and abs(d[1]) < 0.25 * Ppx:
                v = d if d[0] > 0 else -d
                angs.append(np.arctan2(v[1], v[0]))
    return float(np.degrees(np.median(angs))) if angs else 0.0


# ------------------------------------------- marker-free DXF-pattern fallback #
def _cell_pin_pattern(template, xppx, yppx, x_right, y_up, rot_deg, ds, margin_px=6):
    """Compact binary image of the WHOLE cell's pin disks at downsample ``ds`` for the given
    reflection/rotation, plus the (col,row) pixel in that image where CAD (0,0) sits and the
    image dimensions.

    The full cell pattern -- many arrays, two pitches, a spread of diameters and inter-array
    gaps -- is APERIODIC, so its autocorrelation has a single sharp peak. That is precisely
    what a single uniform pin lattice lacks (its correlation aliases every pitch), and it is
    why a marker-free match against this DXF pattern can pin down a cell origin unambiguously.
    All lengths derive from the DXF geometry (``template``) and the scan's µm/px, so nothing
    about cell size is assumed."""
    xs = np.concatenate([a.centers_um[:, 0] for a in template.arrays])
    ys = np.concatenate([a.centers_um[:, 1] for a in template.arrays])
    dia = np.concatenate([np.full(len(a.centers_um), a.diameter_um) for a in template.arrays])
    th = np.deg2rad(rot_deg); c, s = np.cos(th), np.sin(th)
    cr = x_right * (xs / xppx); rr = y_up * (ys / yppx)
    col = c * cr - s * rr; row = s * cr + c * rr          # full-res px, relative to CAD (0,0)
    offc = col.min() - margin_px; offr = row.min() - margin_px
    Wt = int(np.ceil((col.max() + margin_px - offc) / ds)) + 1
    Ht = int(np.ceil((row.max() + margin_px - offr) / ds)) + 1
    T = np.zeros((Ht, Wt))
    cc = (col - offc) / ds; rrp = (row - offr) / ds
    rad = np.maximum(1, np.round(0.5 * dia / xppx / ds).astype(int))
    for k in range(len(cc)):
        ic, ir, rk = int(round(cc[k])), int(round(rrp[k])), int(rad[k])
        r0, r1 = max(0, ir - rk), min(Ht, ir + rk + 1)
        c0, c1 = max(0, ic - rk), min(Wt, ic + rk + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        yy, xx = np.ogrid[r0:r1, c0:c1]
        T[r0:r1, c0:c1][(xx - ic) ** 2 + (yy - ir) ** 2 <= rk * rk] = 1.0
    t0c = (0.0 - offc) / ds; t0r = (0.0 - offr) / ds       # CAD origin pixel in T (ds coords)
    return T, t0c, t0r, Ht, Wt


def _adaptive_pin_mask(z0, valid, pitch_px, min_prom_um=2.0):
    """Depth-robust pin map for the marker-free fallback: mark pixels raised above the LOCAL
    floor rather than a single global height threshold.

    ``scan_feature``'s ``pin_mask`` (used by the marker path) thresholds the leveled height at a
    whole-scan percentile, so on a sample whose cells span a wide depth range the SHALLOW cells'
    pins never clear the global level and vanish. Here the background is a local (~1.5 pin-pitch)
    mean, so a small fixed prominence catches pins at any absolute etch depth -- exactly the
    cells the global mask drops. The marker path is untouched; this feeds the fallback only."""
    from scipy.ndimage import uniform_filter
    k = int(max(5, round(1.5 * pitch_px)))
    vf = valid.astype(np.float32)
    zf = np.where(valid, z0, 0.0).astype(np.float32)
    bg = (uniform_filter(zf, size=k, mode="nearest")
          / np.maximum(uniform_filter(vf, size=k, mode="nearest"), 1e-6))
    return valid & ((z0 - bg) > min_prom_um)


def _pattern_ncc(img, tpl):
    """Normalized cross-correlation of ``img`` with ``tpl``, FULL mode, in [-1, 1].

    Plain correlation is biased toward dense regions of the (fairly full) local pin mask, which
    pulls coarse peaks onto spurious high-fill spots. NCC divides by the local image energy under
    the template, so a peak reflects PATTERN agreement, not local density -- essential for
    accurately localising every cell. 'full' mode keeps the lag->origin mapping unambiguous: a
    peak at (k,l) means the template's top-left sits at img (k-Ht+1, l-Wt+1)."""
    from scipy.signal import fftconvolve
    t = tpl.astype(np.float64)
    t0 = t - t.mean()
    tnorm = float(np.sqrt((t0 * t0).sum()))
    ones = np.ones_like(t)
    if tnorm < 1e-9:
        Hi, Wi = img.shape; Ht, Wt = t.shape
        return np.zeros((Hi + Ht - 1, Wi + Wt - 1))
    num = fftconvolve(img, t0[::-1, ::-1], mode="full")            # sum (img * zero-mean tpl)
    s1 = fftconvolve(img, ones[::-1, ::-1], mode="full")           # local sum of img
    s2 = fftconvolve(img * img, ones[::-1, ::-1], mode="full")     # local sum of img^2
    n = float(t.size)
    denom = np.sqrt(np.maximum(s2 - s1 * s1 / n, 0.0)) * tnorm
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 1e-9, num / denom, 0.0)


def _register_by_pattern(scan, template, z0, valid, *, cell_pitch_um=None,
                         min_overlap=0.5, min_inbounds=0.8, min_contrast_um=4.0,
                         x_right_options=(-1, 1), y_up_options=(1, -1), angles_deg=None,
                         coarse_cell_px=120.0):
    """Locate and register every unit cell WITHOUT an alignment marker, by coarse-to-fine
    phase correlation of the rasterised DXF pin pattern against the scan's pin map.

    Uses a depth-robust LOCAL pin map (:func:`_adaptive_pin_mask`), not the marker path's global
    threshold, so shallow and deep cells are detected alike -- the fallback finds every cell the
    pin data supports regardless of etch depth, with no assumption about how many cells there are
    or that they tile regularly.

    Coarse: at a downsample chosen so the cell spans ~``coarse_cell_px`` px, the DXF pattern is
    matched-filtered over the whole scan for each reflection/small-rotation option; the option
    with the strongest peak fixes the global mirror + rotation, and its correlation map's
    well-separated peaks become candidate cell origins. The number of candidates is bounded by
    the actual scan extent / DXF cell size, never a fixed cell count.
    Fine: each candidate is snapped by a full-resolution phase-correlation refine
    (``_refine_origin``) and kept only if it clears the same pin-overlap, in-bounds and
    ablation-contrast gates the marker path uses. Returns design-frame-numbered CellPlacements.
    """
    if not template.arrays:
        return []
    xppx, yppx = scan.x_um_per_px, scan.y_um_per_px
    H, W = scan.height_raw.shape
    w_um, h_um = template.size_um                          # cell size from the DXF
    Pxpx = (cell_pitch_um[0] if cell_pitch_um else w_um) / xppx
    Pypx = (cell_pitch_um[1] if cell_pitch_um else h_um) / yppx
    min_sep = 0.5 * min(Pxpx, Pypx)
    if angles_deg is None:
        angles_deg = np.arange(-1.0, 1.0001, 0.25)
    pin_pitch_px = float(np.mean([a.pitch_x_um for a in template.arrays])) / xppx
    pin_mask = _adaptive_pin_mask(z0, valid, pin_pitch_px)   # depth-robust; sees shallow cells
    # the coarse candidate is already accurate to ~one downsample step, so the full-res refine
    # only needs a small correction. Clamp it below half a pin pitch (as the marker path does)
    # so the phase peak cannot alias a whole pitch -- or a whole cell -- away.
    refine_clamp = 0.5 * min(a.pitch_x_um for a in template.arrays) / xppx
    ds = max(1, int(round(min(Pxpx, Pypx) / coarse_cell_px)))
    img = pin_mask[::ds, ::ds].astype(float)
    if img.sum() < 5:
        return []

    # --- coarse: pick the global reflection + rotation by the strongest NCC peak ---
    best = None
    for xr in x_right_options:
        for yu in y_up_options:
            for ang in angles_deg:
                T, t0c, t0r, Ht, Wt = _cell_pin_pattern(template, xppx, yppx, xr, yu, ang, ds)
                if T.sum() < 5:
                    continue
                corr = _pattern_ncc(img, T)
                pv = float(corr.max())
                if best is None or pv > best[0]:
                    best = (pv, xr, yu, ang, corr, t0c, t0r, Ht, Wt)
    if best is None:
        return []
    _, xr, yu, ang, corr, t0c, t0r, Ht, Wt = best

    # --- coarse: peel off well-separated NCC peaks -> candidate origins. Accept a peak only if
    # it is a real pattern match (NCC above an absolute floor AND a fraction of the top peak);
    # count is bounded by scan extent / DXF cell size, never a hardcoded tile count. ---
    n_max = int(np.ceil((W / Pxpx + 1.0) * (H / Pypx + 1.0))) + 2
    sep_ds = max(2, int(round(0.7 * min(Pxpx, Pypx) / ds)))
    flat = corr.copy()
    top = float(flat.max())
    accept = 0.30 * top          # permissive: over-generate candidates, let the gates below filter
    cand = []
    for _ in range(max(1, n_max)):
        idx = int(np.argmax(flat))
        k, l = np.unravel_index(idx, flat.shape)
        if not np.isfinite(flat[k, l]) or flat[k, l] < accept:
            break
        cand.append(((l - Wt + 1 + t0c) * ds, (k - Ht + 1 + t0r) * ds))   # (col,row) full-res
        r0, r1 = max(0, k - sep_ds), min(flat.shape[0], k + sep_ds + 1)
        c0, c1 = max(0, l - sep_ds), min(flat.shape[1], l + sep_ds + 1)
        flat[r0:r1, c0:c1] = -np.inf

    # --- fine: full-res phase-correlation refine of translation, at a given rotation ---
    def _place_all(angle):
        out = []
        for (oc, orow) in cand:
            o, ov = _refine_origin(pin_mask, template, (oc, orow), xppx, yppx, yu, xr, angle,
                                   search_px=int(refine_clamp) + ds + 6, max_shift_px=refine_clamp)
            if np.isfinite(ov):
                out.append((o, float(ov)))
        return out

    placed = _place_all(ang)
    # the coarse peak fixes translation but is nearly blind to a fraction-of-a-degree stage
    # rotation, so refine the ONE global rotation at full resolution on the best-registered
    # cell (its far pins move measurably with angle), then re-place every cell at that angle.
    if placed:
        o_best = max(placed, key=lambda t: t[1])[0]
        best_a, best_ov = ang, -1.0
        for fa in np.arange(ang - 0.5, ang + 0.5001, 0.05):
            _, ov = _refine_origin(pin_mask, template, o_best, xppx, yppx, yu, xr, float(fa),
                                   search_px=int(refine_clamp) + ds + 6, max_shift_px=refine_clamp)
            if ov > best_ov:
                best_ov, best_a = ov, float(fa)
        ang = best_a
        placed = _place_all(ang)

    # --- keep placements clearing the marker path's overlap / in-bounds / contrast gates ---
    cells = []
    for (o, ov) in placed:
        if ov < min_overlap:
            continue
        p = CellPlacement(0, float(o[0]), float(o[1]), xppx, yppx, y_up=yu, x_right=xr,
                          rotation_deg=ang, score=float(ov), method="pattern")
        if _inbounds_frac(p, template, W, H) < min_inbounds:
            continue
        if _cell_contrast(z0, valid, template, p, W, H) < min_contrast_um:
            continue                                       # flat / un-ablated region, not a cell
        cells.append(p)

    cells.sort(key=lambda p: -p.score)                     # dedupe near-coincident placements
    kept = []
    for c in cells:
        if not any((c.origin_col - k.origin_col) ** 2 + (c.origin_row - k.origin_row) ** 2
                   < min_sep ** 2 for k in kept):
            kept.append(c)
    if not kept:
        return []
    _assign_grid_indices(kept, Pxpx, Pypx)                 # design-frame (row,col) numbering
    kept.sort(key=lambda p: (p.cell_row, p.cell_col))
    for i, p in enumerate(kept, 1):
        p.cell_id = i
    return kept


def register_sample(scan, template, cell_pitch_um=None, min_overlap=0.6,
                    min_inbounds=0.8, min_contrast_um=4.0, expect_max=80,
                    mirror_x=True, y_up=1):
    """Detect and register EVERY unit cell tiled across an assembled sample.

    Placement is anchored to the UNIQUE alignment marker (never the periodic pin lattice,
    whose overlap peaks alias a pitch away and give plausible-but-wrong positions). Each
    marker centre gives that cell's translation; one global rotation is taken from the marker
    grid; the DXF is mapped mirrored (``mirror_x=True`` -> x_right=-1) so the array is never
    flipped (present figures flipped instead). A small alias-safe refine polishes translation,
    and false markers are rejected by pin overlap AT the marker-anchored position. Cells whose
    marker was missed are filled from the fitted cell lattice. Survivors are indexed in the
    DESIGN frame with (cell_row=1, cell_col=1) = the DXF top-left. Returns a list of
    :class:`CellPlacement` (``cell_row``/``cell_col`` set), sorted by (row, col).

    When no alignment marker can be used (none detected, or detections yield no valid cell),
    this transparently falls back to :func:`_register_by_pattern`, a marker-free coarse-to-fine
    phase correlation of the DXF pin pattern. The marker path itself is unchanged; the fallback
    is only reached when it would otherwise return nothing.
    """
    x_right = -1 if mirror_x else 1
    xppx, yppx = scan.x_um_per_px, scan.y_um_per_px
    H, W = scan.height_raw.shape
    feat, edge, pin_mask, valid, z0 = scan_feature(scan)

    def _fallback(reason):
        print(f"register_sample: {reason}; falling back to marker-free DXF-pattern "
              f"coarse-to-fine phase correlation.")
        return _register_by_pattern(scan, template, z0, valid,
                                    cell_pitch_um=cell_pitch_um, min_inbounds=min_inbounds,
                                    min_contrast_um=min_contrast_um,
                                    x_right_options=((-1, 1) if mirror_x else (1, -1)))

    marker_um = template.marker_size_um
    if not np.isfinite(marker_um):
        marker_um = 200.0
    mx, my = marker_um / xppx, marker_um / yppx
    w_um, h_um = template.size_um
    Pxpx = (cell_pitch_um[0] if cell_pitch_um else w_um) / xppx
    Pypx = (cell_pitch_um[1] if cell_pitch_um else h_um) / yppx
    min_sep = 0.5 * min(Pxpx, Pypx)
    min_pin_pitch = min(a.pitch_x_um for a in template.arrays) / xppx
    max_shift = 0.3 * min_pin_pitch                      # keep the polish below half a pin pitch

    peaks = detect_markers(edge, mx, my, expect_n=expect_max,
                           min_sep_px=min_sep, rel_snr=0.12)
    if not peaks:
        return _fallback("no alignment marker detected")
    rot_deg = _marker_grid_rotation(peaks, Pxpx)         # one global rotation for all cells
    th = np.deg2rad(rot_deg); cth, sth = np.cos(th), np.sin(th)
    rel = np.array([x_right * (marker_um / 2) / xppx, y_up * (marker_um / 2) / yppx])
    Rrel = np.array([cth * rel[0] - sth * rel[1], sth * rel[0] + cth * rel[1]])

    def _place(oc, orow, floor=0.0, method="marker"):
        """Alias-safe refine of a marker/lattice-anchored origin; None if off-scan/below floor."""
        o, ov = _refine_origin(pin_mask, template, (oc, orow), xppx, yppx, y_up, x_right,
                               rot_deg, search_px=int(max_shift) + 6, max_shift_px=max_shift)
        if not np.isfinite(ov):
            return None
        p = CellPlacement(0, float(o[0]), float(o[1]), xppx, yppx, y_up=y_up, x_right=x_right,
                          rotation_deg=rot_deg, score=float(ov), method=method)
        if _inbounds_frac(p, template, W, H) < min_inbounds or ov < floor:
            return None
        if _cell_contrast(z0, valid, template, p, W, H) < min_contrast_um:
            return None                                  # flat / un-ablated region, not a cell
        return p

    def _dedupe(cands):
        out = []
        for c in sorted(cands, key=lambda p: -p.score):
            if not any((c.origin_col - k.origin_col) ** 2 +
                       (c.origin_row - k.origin_row) ** 2 < min_sep ** 2 for k in out):
                out.append(c)
        return out

    # --- CONFIDENT marker cells only (high overlap + real ablation contrast) define the grid
    # skeleton; low-overlap detections are ignored here so they can't seed a wrong lattice. ---
    good = []
    for pk in peaks:
        oc, orow = pk["col"] - Rrel[0], pk["row"] - Rrel[1]
        p = _place(oc, orow, floor=min_overlap)
        if p is not None:
            good.append(p)
    good = _dedupe(good)
    if not good:
        return _fallback("marker detections yielded no confident cell")

    # --- fit ONE global lattice (least squares) through all confident cells, then enumerate
    # every node. A global fit (vs a single anchor + median step) averages out per-pair noise,
    # so far-corner cells are predicted accurately enough for the small alias-safe refine to
    # lock on. Each node is kept only if it clears the ablation-contrast gate (real cell). ---
    _assign_grid_indices(good, Pxpx, Pypx)
    grid = _fit_grid(good)
    if grid is None:                                     # too few confident cells: anchor+basis
        anchor = max(good, key=lambda p: p.score)
        a_vec, b_vec = _estimate_basis(good, Pxpx, Pypx)   # a=col step, b=row step
        (Ox, Oy) = (anchor.origin_col, anchor.origin_row)
        (Rx, Ry), (Cx, Cy) = (b_vec[0], b_vec[1]), (a_vec[0], a_vec[1])
    else:
        (Ox, Oy), (Rx, Ry), (Cx, Cy) = grid             # O=(row1,col1), rowvec, colvec
    N = int(np.ceil(max(W / Pxpx, H / Pypx))) + 2
    cells = []
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            ocp = Ox + i * Rx + j * Cx
            orowp = Oy + i * Ry + j * Cy
            if not (-Pxpx <= ocp <= W + Pxpx and -Pypx <= orowp <= H + Pypx):
                continue
            if any((ocp - c.origin_col) ** 2 + (orowp - c.origin_row) ** 2 < min_sep ** 2
                   for c in cells):
                continue
            p = _place(ocp, orowp, floor=0.0, method="lattice")   # contrast gate = existence
            if p is not None:
                cells.append(p)
    cells = _dedupe(cells)
    if not cells:
        return _fallback("marker grid produced no valid cells")

    _assign_grid_indices(cells, Pxpx, Pypx)              # final clean design-frame numbering
    cells.sort(key=lambda p: (p.cell_row, p.cell_col))
    for i, p in enumerate(cells, 1):
        p.cell_id = i

    # Reliability score only: re-evaluate each finalised placement's pin overlap against a
    # depth-robust LOCAL pin map, at the SAME position (max_shift_px=0, so registration is
    # unchanged -- still fixed by the marker anchor + global-mask logic above). The global mask
    # scores a correctly-placed but shallow cell ~0, wrongly flagging it unreliable; the local
    # mask sees its pins, so the score reflects true placement quality at any etch depth.
    amask = _adaptive_pin_mask(z0, valid,
                               float(np.mean([a.pitch_x_um for a in template.arrays])) / xppx)
    for p in cells:
        _, ov = _refine_origin(amask, template, (p.origin_col, p.origin_row), xppx, yppx,
                               p.y_up, p.x_right, p.rotation_deg, search_px=6, max_shift_px=0.0)
        if np.isfinite(ov):
            p.score = float(ov)
    return cells


# --------------------------------------------------------------------------- #
if __name__ == "__main__":       # pragma: no cover - manual smoke test
    import sys
    from dxf_geometry import read_design
    from vk4 import read_vk4

    dxf = read_design(sys.argv[1])
    vk = read_vk4(sys.argv[2])
    pls = register_scan(vk, dxf.cells[0], n_cells=1, cell_pitch_mm=dxf.cell_pitch_mm)
    for p in pls:
        print(p)
