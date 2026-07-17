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
    origin_col: float            # scan col of the marker min-corner (CAD x = 0)
    origin_row: float            # scan row of the marker origin       (CAD y = 0)
    x_um_per_px: float
    y_um_per_px: float
    y_up: int = 1                # +1: CAD-y up -> row up; -1: flipped
    rotation_deg: float = 0.0
    score: float = float("nan")  # registration quality (higher = better)
    method: str = ""             # "marker+lattice" / "lattice" / "manual"

    def dxf_to_px(self, x_um, y_um):
        """CAD (marker-relative, um) -> scan pixel (col, row). Vectorised."""
        x_um = np.asarray(x_um, float)
        y_um = np.asarray(y_um, float)
        if self.rotation_deg:
            th = np.deg2rad(self.rotation_deg)
            c, s = np.cos(th), np.sin(th)
            xr = c * x_um - s * y_um
            yr = s * x_um + c * y_um
        else:
            xr, yr = x_um, y_um
        col = self.origin_col + xr / self.x_um_per_px
        row = self.origin_row + self.y_up * (yr / self.y_um_per_px)
        return col, row


# --------------------------------------------------------- scan feature maps #
def _level_plane(z, valid):
    """Remove first-order tilt using the low (floor) population; returns z - plane."""
    zz = np.where(valid, z, np.nan)
    lo = np.nanpercentile(zz, 40.0)
    fp = valid & (z <= lo)
    ys, xs = np.nonzero(fp)
    if xs.size < 50:
        return z - np.nanmedian(zz)
    A = np.column_stack([xs.astype(float), ys.astype(float), np.ones(xs.size)])
    coef, *_ = np.linalg.lstsq(A, z[ys, xs], rcond=None)
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    return z - (coef[0] * xx + coef[1] * yy + coef[2])


def scan_feature(scan):
    """Return (feature, edge_mag, pin_mask, valid).

    feature   : pin-bright map (intensity if present, else leveled height), 0..1-ish
    edge_mag  : |gradient| of the feature (marker/pin edges), polarity-agnostic
    pin_mask  : boolean estimate of pin-top pixels (high feature)
    """
    valid = scan.height_raw != 0
    z0 = _level_plane(scan.height_um, valid)

    if getattr(scan, "intensity", None) is not None:
        f = scan.intensity.astype(np.float64)
    else:
        f = np.where(valid, z0, np.nan)
    f = np.where(valid, f, np.nan)
    lo, hi = np.nanpercentile(f, 2), np.nanpercentile(f, 98)
    feat = np.clip((np.nan_to_num(f, nan=lo) - lo) / (hi - lo + 1e-9), 0, 1)

    gy, gx = np.gradient(feat)
    edge = np.hypot(gx, gy)

    # pin tops = high leveled height (robust across dose); combine with feature
    zt = np.where(valid, z0, np.nan)
    thr = np.nanpercentile(zt, 75)
    pin_mask = np.nan_to_num(zt, nan=-1e9) > thr
    return feat, edge, pin_mask, valid


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
def rasterize_cell_pins(cell, x_um_per_px, y_um_per_px, shape, origin, y_up=1):
    """Binary pin-disk mask of one cell placed at ``origin`` (col,row) in a ``shape`` canvas."""
    mask = np.zeros(shape, bool)
    H, W = shape
    oc, orow = origin
    for a in cell.arrays:
        rx = 0.5 * a.diameter_um / x_um_per_px      # per-axis pin radius (ellipse if px non-square)
        ry = 0.5 * a.diameter_um / y_um_per_px
        rrx, rry = int(np.ceil(rx)), int(np.ceil(ry))
        for (x_um, y_um) in a.centers_um:
            ci = int(round(oc + x_um / x_um_per_px))
            ri = int(round(orow + y_up * (y_um / y_um_per_px)))
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
def _refine_origin(pin_mask, cell, origin, xppx, yppx, y_up, search_px=60):
    """Snap the origin to the best design/scan pin-lattice overlap.

    Returns (refined_origin, overlap_score) where overlap_score is in 0..1 (fraction of
    design pins sitting on real pins) so it is directly comparable across y-flip options.
    """
    if not cell.arrays:                         # degenerate template: nothing to refine on
        return origin, float("nan")
    cxmin, cxmax, cymin, cymax = _cell_pin_extent_px(cell, xppx, yppx)
    oc, orow = origin
    rA, rB = orow + y_up * cymin, orow + y_up * cymax
    r_lo, r_hi = int(min(rA, rB) - search_px), int(max(rA, rB) + search_px + 1)
    c_lo, c_hi = int(oc + cxmin - search_px), int(oc + cxmax + search_px + 1)
    r0, r1 = max(0, r_lo), min(pin_mask.shape[0], r_hi)
    c0, c1 = max(0, c_lo), min(pin_mask.shape[1], c_hi)
    if r1 - r0 < 16 or c1 - c0 < 16:
        return origin, float("nan")
    win = pin_mask[r0:r1, c0:c1].astype(float)
    design = rasterize_cell_pins(cell, xppx, yppx, win.shape,
                                 (oc - c0, orow - r0), y_up).astype(float)
    if design.sum() < 5 or win.sum() < 5:
        return origin, float("nan")
    drow, dcol, _pk = _phase_shift(win, design)
    new_origin = (oc + dcol, orow + drow)
    design2 = rasterize_cell_pins(cell, xppx, yppx, win.shape,
                                  (new_origin[0] - c0, new_origin[1] - r0), y_up).astype(bool)
    return new_origin, _overlap_score(win.astype(bool), design2)


def register_scan(scan, template, n_cells=1, overrides=None, refine=True,
                  resolve_yflip=True, cell_pitch_mm=(float("nan"),) * 2, min_overlap=0.12):
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
    feat, edge, pin_mask, valid = scan_feature(scan)

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
                cid, float(o["origin_col"]), float(o["origin_row"]),
                xppx, yppx, int(o.get("y_up", 1)), float(o.get("rotation_deg", 0.0)),
                score=float("nan"), method="manual"))
            continue

        pk = peaks[pk_i] if pk_i < len(peaks) else None
        pk_i += 1
        if pk is None:
            placements.append(CellPlacement(cid, np.nan, np.nan, xppx, yppx,
                                            method="failed"))
            continue

        base_oc = pk["col"] - 0.5 * marker_px_x
        best = None
        yflips = (1, -1) if resolve_yflip else (1,)
        for yf in yflips:
            orow = pk["row"] - 0.5 * marker_px_y if yf == 1 else pk["row"] + 0.5 * marker_px_y
            origin = (base_oc, orow)
            if refine:
                origin, ref = _refine_origin(pin_mask, template, origin, xppx, yppx, yf)
                if np.isfinite(ref):
                    score, method = ref, "marker+lattice"   # overlap 0..1, comparable
                else:
                    score, method = -1.0, "marker"          # refine failed this flip
            else:
                score, method = pk.get("snr", 0.0), "marker"
            if best is None or score > best[0]:
                best = (score, origin, yf, method)
        score, origin, yf, method = best
        placements.append(CellPlacement(
            cid, float(origin[0]), float(origin[1]), xppx, yppx,
            y_up=yf, score=float(score), method=method))

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
