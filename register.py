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


def _dealias_origin(pin_mask, template, origin, xppx, yppx, y_up, x_right, rot_deg, margin=0.02):
    """Belt-and-suspenders against off-by-one-PIN aliasing. On a (near-)periodic array -- e.g.
    large pins on a tight pitch, where a full-pitch shift still overlaps neighbours heavily -- a
    whole-cell placement can lock one pin pitch off and still score high overlap, which would
    crop every array one row/column off. Evaluate the DXF-pin overlap at ``origin`` and at +-one
    pitch (each distinct array pitch, both scan axes AND their diagonals) and snap to the highest.
    On a finite/aperiodic cell the true origin wins (its outermost pins fall inside the real
    array while a shifted copy loses an edge row/column), so this corrects an aliased lock and is
    a no-op when the placement is already right. Returns ((col, row), overlap)."""
    def _ov(oc, orow):
        _, ov = _refine_origin(pin_mask, template, (oc, orow), xppx, yppx, y_up, x_right,
                               rot_deg, search_px=4, max_shift_px=0.0)
        return float(ov) if np.isfinite(ov) else -1.0

    th = np.deg2rad(rot_deg); c, s = np.cos(th), np.sin(th)
    offs = {(0.0, 0.0)}
    for a in template.arrays:                              # one-pitch scan-space steps per array
        vx = np.array([c * x_right * a.pitch_x_um / xppx, s * x_right * a.pitch_x_um / xppx])
        vy = np.array([-s * y_up * a.pitch_y_um / yppx, c * y_up * a.pitch_y_um / yppx])
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i or j:
                    off = i * vx + j * vy
                    offs.add((round(float(off[0]), 1), round(float(off[1]), 1)))
    oc0, or0 = float(origin[0]), float(origin[1])
    best = (oc0, or0, _ov(oc0, or0))
    for (dc, dr) in offs:
        ov = _ov(oc0 + dc, or0 + dr)
        if ov > best[2] + margin:                          # only move for a clear improvement
            best = (oc0 + dc, or0 + dr, ov)
    return (best[0], best[1]), best[2]


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

    # --- de-alias each placement (snap off-by-one-pin locks to the true origin), then keep
    # those clearing the overlap / in-bounds / contrast gates ---
    cells = []
    for (o, ov) in placed:
        o, _ = _dealias_origin(pin_mask, template, o, xppx, yppx, yu, xr, ang)
        o, ov = _refine_origin(pin_mask, template, o, xppx, yppx, yu, xr, ang,
                               search_px=int(refine_clamp) + 6, max_shift_px=refine_clamp)
        if not np.isfinite(ov) or ov < min_overlap:
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


def _solid_square_ncc_markers(feat, marker_px, min_sep, max_n=40, thresh=0.30):
    """Locate solid-square alignment markers by |NCC| of a FILLED-square template against the
    feature map. Markers are solid squares (not outlines) and large relative to the pins, so a
    filled-square matched filter is far more discriminative than the old outline-on-edges match.
    |NCC| is polarity-agnostic (marker milled proud OR recessed). Over-detection is harmless --
    the per-cell two-point solve + overlap gate downstream reject false markers -- so this is
    deliberately permissive. Returns [(col, row, score)], strongest first."""
    sq = max(5, int(round(marker_px)))
    marg = max(2, int(round(0.35 * sq)))
    T = np.zeros((sq + 2 * marg, sq + 2 * marg))
    T[marg:marg + sq, marg:marg + sq] = 1.0
    Ht, Wt = T.shape
    t0 = marg + sq / 2.0
    corr = np.abs(_pattern_ncc(feat, T))
    sep = max(2, int(round(min_sep)))
    out = []
    for _ in range(max_n):
        idx = int(np.argmax(corr)); k, l = np.unravel_index(idx, corr.shape)
        v = corr[k, l]
        if not np.isfinite(v) or v < thresh:
            break
        out.append((float(l - Wt + 1 + t0), float(k - Ht + 1 + t0), float(v)))
        r0, r1 = max(0, k - sep), min(corr.shape[0], k + sep + 1)
        c0, c1 = max(0, l - sep), min(corr.shape[1], l + sep + 1)
        corr[r0:r1, c0:c1] = -np.inf
    return out


def _lattice_step_candidates(cells, cell_w, cell_h, ntop=3):
    """Candidate primitive tile-step vectors (dcol, drow) along each axis, smallest magnitude
    first. ``cells`` = list of (origin_col, origin_row, overlap). A horizontal-ish offset between
    two cells (small |drow|, |dcol| > ~half a cell) is a column step; vertical-ish is a row step.
    Smallest first so the primitive (adjacent-cell) step is tried before its multiples; the caller
    scores each candidate lattice by how many nodes actually verify, so a wrong step self-rejects.
    Returns (col_vecs, row_vecs)."""
    col, row = [], []
    for i in range(len(cells)):
        for j in range(len(cells)):
            if i == j:
                continue
            dc = cells[j][0] - cells[i][0]
            dr = cells[j][1] - cells[i][1]
            if dc > 0.4 * cell_w and abs(dr) < 0.3 * cell_h:
                col.append((dc, dr))
            if dr > 0.4 * cell_h and abs(dc) < 0.3 * cell_w:
                row.append((dc, dr))

    def _uniq_small(offs, key):
        out = []
        for d in sorted(offs, key=key):
            if not any(abs(key(d) - key(u)) < 0.15 * (cell_w + cell_h) for u in out):
                out.append(d)
            if len(out) >= ntop:
                break
        return out

    return _uniq_small(col, lambda d: d[0]), _uniq_small(row, lambda d: d[1])


def _global_rotation(amask, template, origin, xppx, yppx, y_up, x_right, max_shift,
                     span=1.2, step=0.1):
    """Best single stage-rotation (deg) for the whole sample, from an overlap sweep on one
    well-registered anchor cell. Stage rotation is global, so one value serves every cell."""
    best_a, best_ov = 0.0, -1.0
    n = int(round(2 * span / step))
    for k in range(n + 1):
        a = -span + k * step
        _, ov = _refine_origin(amask, template, origin, xppx, yppx, y_up, x_right, float(a),
                               search_px=int(max_shift) + 6, max_shift_px=max_shift)
        if np.isfinite(ov) and ov > best_ov:
            best_ov, best_a = ov, float(a)
    return best_a


def _grid_nodes(O0, col_vec, row_vec, W, H, cell_w, cell_h):
    """All lattice-node origins ``O0 + i*col_vec + j*row_vec`` whose cell plausibly lies in-scan."""
    span = max(W, H)
    ni = int(np.ceil(span / max(1.0, float(np.hypot(*col_vec))))) + 2
    nj = int(np.ceil(span / max(1.0, float(np.hypot(*row_vec))))) + 2
    nodes = []
    for i in range(-ni, ni + 1):
        for j in range(-nj, nj + 1):
            oc = O0[0] + i * col_vec[0] + j * row_vec[0]
            orow = O0[1] + i * col_vec[1] + j * row_vec[1]
            if -0.5 * cell_w <= oc <= W + 0.5 * cell_w and -0.5 * cell_h <= orow <= H + 0.5 * cell_h:
                nodes.append((oc, orow))
    return nodes


def _probe_lattice(nodes, template, amask, z0, valid, xppx, yppx, y_up, x_right, rot,
                   max_shift, W, H, min_overlap, min_inbounds, min_contrast):
    """Register every predicted lattice node: a phase-clamped refine, then -- only where a cell is
    actually present (the refine clears the overlap gate) -- an off-by-one-pin de-alias to lock the
    true origin. Nodes far from any real cell (empty gaps, off-scan) fail the gate and are dropped
    without a wasted de-alias. A clamped refine (< half a pin pitch) on an accurate lattice node
    cannot alias, so this stays off-by-one-safe on a dense periodic array. Returns placements."""
    out = []
    for (oc, orow) in nodes:
        o, ov = _refine_origin(amask, template, (oc, orow), xppx, yppx, y_up, x_right, rot,
                               search_px=int(max_shift) + 6, max_shift_px=max_shift)
        if not np.isfinite(ov) or ov < min_overlap:
            continue                                    # no cell at this node -> skip (no de-alias)
        od, _ = _dealias_origin(amask, template, o, xppx, yppx, y_up, x_right, rot)
        od, ovd = _refine_origin(amask, template, od, xppx, yppx, y_up, x_right, rot,
                                 search_px=int(max_shift) + 6, max_shift_px=max_shift)
        if np.isfinite(ovd) and ovd >= ov:              # keep the de-aliased lock only if not worse
            o, ov = od, ovd
        p = CellPlacement(0, float(o[0]), float(o[1]), xppx, yppx, y_up=y_up, x_right=x_right,
                          rotation_deg=rot, score=float(ov), method="lattice")
        if _inbounds_frac(p, template, W, H) < min_inbounds:
            continue
        if _cell_contrast(z0, valid, template, p, W, H) < min_contrast:
            continue
        out.append(p)
    return out


def _dedup_cells(cells, min_sep):
    """Keep the highest-overlap of any cluster of placements within ``min_sep`` px."""
    out = []
    for p in sorted(cells, key=lambda q: -q.score):
        if not any((p.origin_col - k.origin_col) ** 2 + (p.origin_row - k.origin_row) ** 2
                   < min_sep ** 2 for k in out):
            out.append(p)
    return out


def register_sample(scan, template, cell_pitch_um=None, min_overlap=0.6,
                    min_inbounds=0.8, min_contrast_um=4.0, expect_max=80,
                    mirror_x=True, y_up=1, anchor_frac=0.85):
    """Detect and register EVERY unit cell tiled across an assembled sample (Tier 1).

    Marker-anchored **lattice** registration (quality over runtime):

    1. Detect solid alignment-marker candidates (permissive |NCC|). The marker sits in an empty
       corner, OFF the pin lattice, so it is an *absolute* origin reference: a phase-clamped refine
       plus off-by-one-pin de-alias locks each detection to the true cell origin. This is what makes
       a dense, uniform, periodic array (large pins on a tight pitch) resolvable -- an overlap gate
       alone cannot tell a real placement from a one-pin-pitch-shifted one, but the marker can.
    2. Take the highest-overlap detections as anchors, estimate the tile step vectors from their
       geometry (the tile pitch is NOT in the single-cell DXF, so it is measured here), and fix the
       one global stage rotation.
    3. Probe EVERY predicted lattice node and keep those clearing the overlap / in-bounds /
       ablation-contrast gates, then re-fit the lattice from the survivors and re-probe so a large
       grid never drifts a pin off at the far corners. Off-lattice spurious detections (e.g. a pin
       mistaken for the marker) are never on a node, so they are rejected by construction.

    The DXF is mapped mirrored (``mirror_x=True`` -> x_right=-1) so the array is never flipped;
    survivors are indexed in the DESIGN frame ((1,1) = DXF top-left) by ``_assign_grid_indices``.
    Falls back to :func:`_register_by_pattern` (marker-free pin-pattern correlation) when no marker
    anchors a cell -- note a marker-less single uniform periodic array is genuinely off-by-one
    ambiguous, so a corner marker (ideally an asymmetric fiducial) is required for that geometry.
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
    cell_w = (cell_pitch_um[0] if cell_pitch_um else w_um) / xppx
    cell_h = (cell_pitch_um[1] if cell_pitch_um else h_um) / yppx
    min_sep = 0.5 * min(cell_w, cell_h)                  # cells cannot sit closer than ~a cell
    min_pin_pitch = min(a.pitch_x_um for a in template.arrays) / xppx
    max_shift = 0.3 * min_pin_pitch                      # keep every polish below half a pin pitch
    amask = _adaptive_pin_mask(z0, valid,
                               float(np.mean([a.pitch_x_um for a in template.arrays])) / xppx)

    # --- 1. marker candidates -> absolute, off-by-one-safe anchor origins ---
    marker_px = 0.5 * (mx + my)
    markers = _solid_square_ncc_markers(feat, marker_px, max(0.6 * marker_px, 20),
                                        max_n=max(8, 2 * expect_max))
    if not markers:
        return _fallback("no alignment marker detected")

    cand = []
    for (mc, mr, _sc) in markers:
        o0 = (mc - x_right * 0.5 * mx, mr - y_up * 0.5 * my)   # marker centre -> CAD origin (rot~0)
        # First pass: a phase-clamped refine only, NO de-alias. The marker is an absolute,
        # off-pin-lattice anchor, so a <half-pitch refine already locks the true origin without
        # aliasing. De-aliasing raw candidates is pointless work (most are spurious peaks far from
        # any cell) and can only mis-snap them -- off-by-one is resolved later, per lattice node,
        # where "the expected location" is actually defined.
        o, ov = _refine_origin(amask, template, o0, xppx, yppx, y_up, x_right, 0.0,
                               search_px=int(max_shift) + 6, max_shift_px=max_shift)
        if np.isfinite(ov) and ov >= min_overlap:
            cand.append((float(o[0]), float(o[1]), float(ov)))
    cand.sort(key=lambda t: -t[2])
    dedup = []
    for c in cand:
        if not any((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2 < min_sep ** 2 for k in dedup):
            dedup.append(c)
    if not dedup:
        return _fallback("no marker-anchored cell cleared the overlap gate")

    O0 = dedup[0]                                        # highest overlap = surest phase anchor
    rot = _global_rotation(amask, template, (O0[0], O0[1]), xppx, yppx, y_up, x_right, max_shift)

    # --- 2. tile step vectors: prefer high-overlap anchors, fall back to all cells per axis ---
    top = O0[2]
    anchors = [c for c in dedup if c[2] >= max(min_overlap, anchor_frac * top)]
    col_vecs, row_vecs = _lattice_step_candidates(anchors, cell_w, cell_h)
    col_all, row_all = _lattice_step_candidates(dedup, cell_w, cell_h)
    col_vecs = col_vecs or col_all
    row_vecs = row_vecs or row_all

    if not col_vecs or not row_vecs:                     # no 2-D lattice (single cell/row/col)
        kept = [CellPlacement(0, c[0], c[1], xppx, yppx, y_up=y_up, x_right=x_right,
                              rotation_deg=rot, score=c[2], method="marker") for c in dedup]
        _assign_grid_indices(kept, cell_w, cell_h)
        kept.sort(key=lambda p: (p.cell_row, p.cell_col))
        for i, p in enumerate(kept, 1):
            p.cell_id = i
        return kept

    # --- 3. probe each candidate lattice; keep the one that verifies the most nodes ---
    best = None
    for cv in col_vecs:
        for rv in row_vecs:
            nodes = _grid_nodes((O0[0], O0[1]), cv, rv, W, H, cell_w, cell_h)
            placed = _dedup_cells(
                _probe_lattice(nodes, template, amask, z0, valid, xppx, yppx, y_up, x_right, rot,
                               max_shift, W, H, min_overlap, min_inbounds, min_contrast_um),
                min_sep)
            score = (len(placed), -len(nodes))           # most cells, then coarsest (fewest probes)
            if best is None or score > best[0]:
                best = (score, placed)
    kept = best[1]
    if not kept:
        return _fallback("lattice probe verified no cell")

    # --- 4. re-fit the lattice from the survivors and re-probe the full row/col span (tightens far
    #        nodes and fills any interior cell the first pass missed) ---
    if len(kept) >= 4:
        _assign_grid_indices(kept, cell_w, cell_h)
        A = np.array([[p.cell_col, p.cell_row, 1.0] for p in kept])
        kx, *_ = np.linalg.lstsq(A, np.array([p.origin_col for p in kept]), rcond=None)
        ky, *_ = np.linalg.lstsq(A, np.array([p.origin_row for p in kept]), rcond=None)
        rs = [p.cell_row for p in kept]; cs = [p.cell_col for p in kept]
        nodes = [(float(np.array([c, r, 1.0]) @ kx), float(np.array([c, r, 1.0]) @ ky))
                 for r in range(min(rs), max(rs) + 1) for c in range(min(cs), max(cs) + 1)]
        refit = _dedup_cells(
            _probe_lattice(nodes, template, amask, z0, valid, xppx, yppx, y_up, x_right, rot,
                           max_shift, W, H, min_overlap, min_inbounds, min_contrast_um),
            min_sep)
        if len(refit) >= len(kept):
            kept = refit

    _assign_grid_indices(kept, cell_w, cell_h)           # design-frame (row,col) numbering
    kept.sort(key=lambda p: (p.cell_row, p.cell_col))
    for i, p in enumerate(kept, 1):
        p.cell_id = i
    return kept


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
