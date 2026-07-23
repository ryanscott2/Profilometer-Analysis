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
from types import SimpleNamespace

import numpy as np

import accel                       # optional GPU/CPU FFT-NCC backend (see accel.py)


# --------------------------------------------------------------------------- #
class RegistrationAmbiguityError(RuntimeError):
    """Automatic registration cannot establish a unique absolute design origin."""


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
    method: str = ""             # includes "uniform-edge" / "uniform-phase" for markerless grids
    cell_row: int = 0            # design-frame unit-cell index (1 = top), set by register_sample
    cell_col: int = 0            # design-frame unit-cell index (1 = left)
    absolute_origin: bool = True # False for a lattice-phase lock with unresolved integer index
    ambiguous_axes: str = ""     # any unresolved design axes, e.g. "xy" for an interior crop

    def dxf_to_px(self, x_um, y_um):
        """CAD (marker-relative, um) -> scan pixel (col, row). Vectorised.

        Supports a left-right reflection (``x_right`` = -1, common: the scan is X-mirrored
        vs the DXF so the bottom-left design marker images at the cell's bottom-right) and a
        top-bottom reflection (``y_up`` = -1), plus a small rotation."""
        x_um = np.asarray(x_um, float)
        y_um = np.asarray(y_um, float)
        xr = self.x_right * x_um                             # reflect in PHYSICAL um...
        yr = self.y_up * y_um
        if self.rotation_deg:                                # ...rotate in um, THEN scale per axis.
            th = np.deg2rad(self.rotation_deg)               # (Rotating before the per-axis um->px
            c, s = np.cos(th), np.sin(th)                    # scale is correct for non-square pixels;
            xr, yr = c * xr - s * yr, s * xr + c * yr        # identical to the old px-space rotation
        col = self.origin_col + xr / self.x_um_per_px        # when x_um_per_px == y_um_per_px.)
        row = self.origin_row + yr / self.y_um_per_px
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
    with np.errstate(divide="ignore", invalid="ignore"):
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
        centers = np.asarray(a.centers_um, float)
        if centers.size == 0:
            continue
        # Vectorised equivalent of the former per-pin/per-pixel double loop, byte-identical to it on
        # every tested geometry (see selftest [18]): reflect + rotate in um then scale, np.rint ==
        # Python round() (both round-half-to-even), the same ellipse test on the same integer offset
        # grid, the same in-bounds guard, and the same idempotent OR into the mask.  (Do NOT swap
        # np.rint for (x+0.5).astype(int): that truncates toward zero and would mis-round negative /
        # exact-half centres.)
        xr = x_right * centers[:, 0]                          # reflect
        yr = y_up * centers[:, 1]
        xr, yr = c * xr - s * yr, s * xr + c * yr             # rotate (matches CellPlacement.dxf_to_px)
        ci = np.rint(oc + xr / x_um_per_px).astype(np.intp)   # pin-centre pixel (col, row)
        ri = np.rint(orow + yr / y_um_per_px).astype(np.intp)
        dcc, drr = np.arange(-rrx, rrx + 1), np.arange(-rry, rry + 1)
        DC, DR = np.meshgrid(dcc, drr)                        # disk stencil (identical for every pin)
        keep = (DC / rx) ** 2 + (DR / ry) ** 2 <= 1.0
        drel, crel = DR[keep], DC[keep]
        if drel.size == 0:
            continue
        yy = ri[:, None] + drel[None, :]                     # (n_pins, n_stencil) stamped pixels
        xx = ci[:, None] + crel[None, :]
        inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        mask[yy[inb], xx[inb]] = True
    return mask


def _cell_pin_extent_um(cell):
    """Pin-centre extent from the marker origin in physical coordinates (um)."""
    xs = np.concatenate([a.centers_um[:, 0] for a in cell.arrays])
    ys = np.concatenate([a.centers_um[:, 1] for a in cell.arrays])
    return xs.min(), xs.max(), ys.min(), ys.max()


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
    xmin, xmax, ymin, ymax = _cell_pin_extent_um(cell)
    oc, orow = origin
    th = np.deg2rad(rot_deg); cth, sth = np.cos(th), np.sin(th)
    rr_v, cc_v = [], []                         # tight window around the footprint corners
    for x_um in (xmin, xmax):
        for y_um in (ymin, ymax):
            # Rotate in physical space, exactly as dxf_to_px/rasterize_cell_pins do, and only
            # then convert each axis to pixels. Rotating already-scaled pixel coordinates is
            # geometrically wrong whenever x/y pixel sizes differ.
            x_rel, y_rel = x_right * x_um, y_up * y_um
            x_rot = cth * x_rel - sth * y_rel
            y_rot = sth * x_rel + cth * y_rel
            cc_v.append(oc + x_rot / xppx)
            rr_v.append(orow + y_rot / yppx)
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

    # Refine on a depth-robust LOCAL pin mask with the shift clamped below half a pin pitch. The
    # marker is an absolute, off-lattice anchor, so we trust its origin and only let the refine
    # snap a little -- otherwise a dense periodic array (large pins, tight pitch) can pull the
    # phase peak a fraction of a pitch off the true origin. Matches register_sample.
    if template.arrays:
        amask = _adaptive_pin_mask(
            z0, valid, float(np.mean([a.pitch_x_um for a in template.arrays])) / xppx)
        max_shift = 0.3 * min(a.pitch_x_um for a in template.arrays) / xppx
    else:
        amask, max_shift = pin_mask, None

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
    if (refine and n_detect > 0 and template.marker_shape == "L"
            and template.marker_polygon_um is not None
            and len(template.marker_polygon_um) >= 3 and template.arrays):
        auto = _register_rotated_marker_cells(
            feat, z0, valid, amask, template, xppx, yppx,
            ((1, -1) if resolve_xflip else (1,)),
            ((1, -1) if resolve_yflip else (1,)),
            min_sep, max_shift, min_overlap, n_detect)
        if auto:
            placements = []
            auto_i = 0
            for cid in range(1, n_cells + 1):
                if cid in overrides:
                    o = overrides[cid]
                    placements.append(CellPlacement(
                        cid, float(o["origin_col"]), float(o["origin_row"]), xppx, yppx,
                        y_up=int(o.get("y_up", 1)), x_right=int(o.get("x_right", 1)),
                        rotation_deg=float(o.get("rotation_deg", 0.0)),
                        score=float("nan"), method="manual"))
                elif auto_i < len(auto):
                    p = auto[auto_i]
                    p.cell_id = cid
                    placements.append(p)
                    auto_i += 1
                else:
                    placements.append(CellPlacement(cid, np.nan, np.nan, xppx, yppx,
                                                    method="failed"))
            return _finalize_placements(placements, min_sep, min_overlap,
                                        template.size_um[1] / yppx)

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
                    origin, ref = _refine_origin(
                        amask, template, origin, xppx, yppx, yf, xf,
                        search_px=(int(max_shift) + 6) if max_shift else 60,
                        max_shift_px=max_shift)
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
    = left; row 1 = max design-y = top), given the global reflection stored on each cell.

    Indices are ABSOLUTE: each successive cluster advances by the origin gap measured in tile
    pitches (the pitch unit is taken from the survivors' own median adjacent-cluster spacing, which
    is robust to Pxpx/Pypx being the pin-cluster size rather than the true tile pitch). So a MISSING
    interior row/column leaves a HOLE in the numbering instead of contiguously renumbering the later
    cells -- which would silently shift their cell_params (passes/speed) assignment. NOTE: a missing
    EDGE row/column still cannot be recovered here (no anchor); run_sample warns on that via the
    reverse CSV check. For a complete grid this is identical to the old dense ranking."""
    if not cells:
        return
    x_right, y_up = cells[0].x_right, cells[0].y_up
    colkey = [x_right * c.origin_col for c in cells]
    rowkey = [y_up * c.origin_row for c in cells]

    def _abs_index(groups, expected):
        reps = [float(np.mean([v for _, v in g])) for g in groups]      # cluster mean, ascending
        gaps = np.diff(reps)
        pos = gaps[gaps > 1e-6]
        min_gap = float(pos.min()) if pos.size else 0.0
        # unit = one tile pitch. Normally the closest two present cells are one pitch apart, so the
        # SMALLEST adjacent gap is the pitch (robust to a single missing interior cell). BUT under an
        # alternating / multi-cell dropout (survivors at design cols 1,3,5) NO two survivors are one
        # pitch apart, so the min gap is 2+ pitches -> using it renumbers 1,3,5 as 1,2,3 and silently
        # mis-maps every cell's laser params. Fall back to the DESIGN tile pitch (``expected``, in px,
        # from cell_pitch_um) when the survivors' min gap disagrees with it by more than ~half a pitch.
        if expected and expected > 1e-6 and (min_gap <= 1e-6 or min_gap > 1.5 * expected):
            if min_gap > 1.5 * expected:
                print(f"WARNING: grid indexing: min survivor spacing {min_gap:.0f}px exceeds 1.5x the "
                      f"design tile pitch {expected:.0f}px (alternating/edge dropout?) -> using the "
                      f"design pitch so cell numbering / laser-param mapping is not shifted.")
            unit = float(expected)
        elif min_gap > 1e-6:
            unit = min_gap
        else:
            unit = float(expected) if (expected and expected > 1e-6) else 1.0
        idx = [1]
        for k in range(1, len(reps)):
            idx.append(idx[-1] + max(1, int(round((reps[k] - reps[k - 1]) / unit))))
        return {mi: idx[gi] for gi, g in enumerate(groups) for mi, _ in g}

    col_idx = _abs_index(_cluster_1d(colkey, 0.5 * Pxpx), Pxpx)
    row_asc = _abs_index(_cluster_1d(rowkey, 0.5 * Pypx), Pypx)         # ascending design-y = bottom
    top = max(row_asc.values())                                        # flip so row 1 = top
    for idx, c in enumerate(cells):
        c.cell_col, c.cell_row = col_idx[idx], top - row_asc[idx] + 1


# ------------------------------------------- marker-free DXF-pattern fallback #
def _uniform_lattice_alias_reason(template):
    """Describe a complete uniform lattice that needs finite-edge registration.

    A single complete rectangular pin array has no internal absolute-phase feature: translating
    it by one pitch retains the same interior.  Such arrays bypass the generic whole-pattern
    matcher and are registered by :func:`_register_uniform_lattice`, which accepts an absolute
    origin only when observed pin terminations identify both finite array indices.  Multi-array or
    intentionally incomplete patterns are aperiodic and remain eligible for whole-pattern NCC.
    """
    if len(template.arrays) != 1:
        return None
    a = template.arrays[0]
    if a.nx < 2 or a.ny < 2 or a.n_pins != a.nx * a.ny:
        return None
    return (f"markerless DXF is a complete uniform {a.nx}x{a.ny} lattice "
            f"(D{a.diameter_um:g} um, pitch {a.pitch_x_um:g}x{a.pitch_y_um:g} um); "
            "its interior has pitch-equivalent origins")


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
    xr = x_right * xs; yr = y_up * ys                     # reflect + rotate in um, then scale per axis
    xr, yr = c * xr - s * yr, s * xr + c * yr             # (matches dxf_to_px; correct for non-square)
    col = xr / xppx; row = yr / yppx                      # full-res px, relative to CAD (0,0)
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


def _uniform_component_centers(pin_mask, array, xppx, yppx):
    """Return physical ``(x,y)`` centroids of plausible individual pins.

    Connected components are used only to estimate lattice angle and phase.  Final finite-array
    scoring samples every expected lattice node directly, so a damaged or bridged component cannot
    by itself create edge evidence.  The broad area window deliberately retains partial/eroded pin
    tops while rejecting the many one-pixel prominence speckles seen in real VK4 height maps.
    """
    from scipy.ndimage import center_of_mass, label

    labels, n_labels = label(pin_mask)
    if n_labels == 0:
        return np.empty((0, 2), float)
    areas = np.bincount(labels.ravel())[1:]
    expected = np.pi * (0.5 * array.diameter_um / xppx) * (0.5 * array.diameter_um / yppx)
    keep = np.flatnonzero((areas >= max(3.0, 0.04 * expected))
                          & (areas <= 1.6 * expected)) + 1
    if keep.size == 0:
        return np.empty((0, 2), float)
    rc = np.asarray(center_of_mass(pin_mask, labels, keep), float)
    return np.column_stack([rc[:, 1] * xppx, rc[:, 0] * yppx])


def _fit_uniform_lattice(centers_xy_um, pitch_um, angle_limits=(-5.0, 5.0),
                         min_components=9):
    """Fit square-lattice angle and phase, robustly, without assigning absolute indices.

    Circular coherence supplies exactly the information a periodic crop contains: orientation and
    phase modulo one pitch.  Absolute integer indices are intentionally left to finite-edge scoring.
    Returns ``(angle_deg, phase_u_um, phase_v_um, inlier_centers)``.
    """
    from scipy.optimize import minimize_scalar

    xy = np.asarray(centers_xy_um, float)
    min_components = max(6, int(min_components))
    if len(xy) < min_components:
        raise RegistrationAmbiguityError(
            f"Only {len(xy)} isolated pin components were found; at least {min_components} are "
            "required to fit a markerless lattice reliably.")
    lo, hi = map(float, angle_limits)
    if not np.isfinite(lo + hi) or hi <= lo:
        raise ValueError("uniform-lattice angle limits must be finite and increasing")

    def _coherence(angle_deg, points=xy):
        th = np.deg2rad(angle_deg); c, s = np.cos(th), np.sin(th)
        u = c * points[:, 0] + s * points[:, 1]
        v = -s * points[:, 0] + c * points[:, 1]
        zu = np.mean(np.exp(2j * np.pi * u / pitch_um))
        zv = np.mean(np.exp(2j * np.pi * v / pitch_um))
        return 0.5 * (abs(zu) + abs(zv))

    coarse = np.linspace(lo, hi, max(41, int(np.ceil((hi - lo) / 0.05)) + 1))
    coarse_score = np.asarray([_coherence(a) for a in coarse])
    a0 = float(coarse[int(np.argmax(coarse_score))])
    half = max(0.06, 1.5 * (coarse[1] - coarse[0]))
    bounds = (max(lo, a0 - half), min(hi, a0 + half))
    opt = minimize_scalar(lambda a: -_coherence(float(a)), bounds=bounds,
                          method="bounded", options={"xatol": 1e-5})
    angle = float(opt.x)

    def _phase_and_residual(points, angle_deg):
        th = np.deg2rad(angle_deg); c, s = np.cos(th), np.sin(th)
        u = c * points[:, 0] + s * points[:, 1]
        v = -s * points[:, 0] + c * points[:, 1]
        zu = np.mean(np.exp(2j * np.pi * u / pitch_um))
        zv = np.mean(np.exp(2j * np.pi * v / pitch_um))
        pu = float(np.angle(zu) * pitch_um / (2 * np.pi)) % pitch_um
        pv = float(np.angle(zv) * pitch_um / (2 * np.pi)) % pitch_um
        ru = (u - pu + 0.5 * pitch_um) % pitch_um - 0.5 * pitch_um
        rv = (v - pv + 0.5 * pitch_um) % pitch_um - 0.5 * pitch_um
        return pu, pv, np.hypot(ru, rv), 0.5 * (abs(zu) + abs(zv))

    pu, pv, residual, coherence = _phase_and_residual(xy, angle)
    inlier = residual <= 0.24 * pitch_um
    if inlier.sum() < min_components or inlier.mean() < 0.55 or coherence < 0.55:
        raise RegistrationAmbiguityError(
            f"Pin components do not support one reliable {pitch_um:g} um lattice "
            f"({int(inlier.sum())}/{len(xy)} inliers, coherence {coherence:.2f}). Check the DXF "
            "pitch, tile stitching, and whether the scan contains more than one lattice phase.")

    # One robust refit after removing debris/partial-component centroids.
    pts = xy[inlier]
    opt = minimize_scalar(lambda a: -_coherence(float(a), pts), bounds=bounds,
                          method="bounded", options={"xatol": 1e-6})
    angle = float(opt.x)
    pu, pv, residual, coherence = _phase_and_residual(pts, angle)
    pts = pts[residual <= 0.18 * pitch_um]
    # A single D300 VK4 frame contains only ~3x4 pins; after excluding border-clipped components it
    # can legitimately leave 6-8 clean centroids.  Six high-coherence inliers still overdetermine
    # angle plus two phase coordinates, while the independent node-quality gates below remain in
    # force.  Larger/noisier images still need coherence >= 0.70.
    min_final = 6 if coherence >= 0.80 else 9
    if len(pts) < min_final or coherence < 0.70:
        raise RegistrationAmbiguityError(
            f"Lattice phase remained unstable after robust fitting ({len(pts)} inliers, "
            f"coherence {coherence:.2f}).")
    pu, pv, _residual, _coherence = _phase_and_residual(pts, angle)
    return angle, pu, pv, pts


def _uniform_node_grid(pin_mask, valid, array, xppx, yppx, angle_deg, phase_u, phase_v):
    """Sample observed presence/absence at every testable anonymous lattice node.

    Rows/columns in the returned matrices are global *phase* indices, not DXF indices.  A node is
    testable only when the full nominal pin neighbourhood is valid and in frame.  Pin presence is
    a nine-point inner-disk vote, which remains usable when neighbouring prominence components are
    bridged and avoids treating every image pixel as independent evidence.
    """
    H, W = pin_mask.shape
    th = np.deg2rad(angle_deg); c, s = np.cos(th), np.sin(th)
    corners = np.array([[0.0, 0.0], [W * xppx, 0.0],
                        [0.0, H * yppx], [W * xppx, H * yppx]])
    cu = c * corners[:, 0] + s * corners[:, 1]
    cv = -s * corners[:, 0] + c * corners[:, 1]
    p = float(array.pitch_x_um)
    qu = np.arange(int(np.floor((cu.min() - phase_u) / p)) - 1,
                   int(np.ceil((cu.max() - phase_u) / p)) + 2)
    qv = np.arange(int(np.floor((cv.min() - phase_v) / p)) - 1,
                   int(np.ceil((cv.max() - phase_v) / p)) + 2)
    Q_u, Q_v = np.meshgrid(qu, qv)
    U = phase_u + Q_u * p; V = phase_v + Q_v * p
    X = c * U - s * V; Y = s * U + c * V
    col = X / xppx; row = Y / yppx

    radius = 0.5 * float(array.diameter_um)
    # Full-neighbourhood validity: centre plus an outer physical ring.
    outer = radius * np.array([[0.0, 0.0], [1.05, 0.0], [-1.05, 0.0],
                               [0.0, 1.05], [0.0, -1.05], [0.74, 0.74],
                               [0.74, -0.74], [-0.74, 0.74], [-0.74, -0.74]])
    testable = np.ones(Q_u.shape, bool)
    for dx, dy in outer:
        cc = np.rint(col + dx / xppx).astype(int)
        rr = np.rint(row + dy / yppx).astype(int)
        inside = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
        sample = np.zeros(Q_u.shape, bool)
        sample[inside] = valid[rr[inside], cc[inside]]
        testable &= sample

    # Inner-disk votes classify the anonymous lattice node as pin or floor.
    inner = radius * np.array([[0.0, 0.0], [0.48, 0.0], [-0.48, 0.0],
                               [0.0, 0.48], [0.0, -0.48], [0.34, 0.34],
                               [0.34, -0.34], [-0.34, 0.34], [-0.34, -0.34]])
    votes = np.zeros(Q_u.shape, np.int16)
    for dx, dy in inner:
        cc = np.rint(col + dx / xppx).astype(int)
        rr = np.rint(row + dy / yppx).astype(int)
        inside = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
        hit = np.zeros(Q_u.shape, bool)
        hit[inside] = pin_mask[rr[inside], cc[inside]]
        votes += hit
    present = testable & (votes >= 5)
    return qu, qv, testable, present


def _rect_count(prefix, u0, u1, v0, v1, qu, qv):
    """Integral-image count over inclusive global-index rectangle, clipped to the node grid."""
    iu0 = max(0, int(u0 - qu[0])); iu1 = min(len(qu) - 1, int(u1 - qu[0]))
    iv0 = max(0, int(v0 - qv[0])); iv1 = min(len(qv) - 1, int(v1 - qv[0]))
    if iu1 < iu0 or iv1 < iv0:
        return 0
    return int(prefix[iv1 + 1, iu1 + 1] - prefix[iv0, iu1 + 1]
               - prefix[iv1 + 1, iu0] + prefix[iv0, iu0])


def _uniform_prediction(qu, qv, testable, lo_u, lo_v, nx, ny):
    """Testable-node mask predicted to contain pins for one finite-index hypothesis."""
    return (testable & (qu[None, :] >= lo_u) & (qu[None, :] < lo_u + nx)
            & (qv[:, None] >= lo_v) & (qv[:, None] < lo_v + ny))


def _block_bootstrap_margin(present, testable, best_pred, alt_pred, *, seed=20260722,
                            block=3, n_boot=2000):
    """One-percent lower bound for alternative-minus-best node loss.

    Adjacent 3x3 lattice nodes are resampled as blocks, limiting false confidence from spatially
    correlated threshold/dropout errors.  Pixels are never treated as independent observations.
    """
    diff = ((present != alt_pred).astype(np.int16)
            - (present != best_pred).astype(np.int16))
    vals = []
    H, W = diff.shape
    for r0 in range(0, H, block):
        for c0 in range(0, W, block):
            t = testable[r0:r0 + block, c0:c0 + block]
            if t.any():
                vals.append(int(diff[r0:r0 + block, c0:c0 + block][t].sum()))
    if not vals:
        return float("-inf")
    vals = np.asarray(vals, np.int16)
    rng = np.random.default_rng(seed)
    margins = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].sum(axis=1)
    return float(np.percentile(margins, 1.0))


def _register_uniform_lattice(scan, template, z0, valid, *, x_right_options=(-1, 1),
                              y_up_options=(1, -1), angles_deg=None,
                              allow_phase_only=False):
    """Register one finite, markerless uniform array using termination evidence.

    The lattice fit determines rotation and phase only modulo pitch.  Every finite integer-index
    hypothesis is then scored on independent lattice nodes, including valid floor nodes where a
    shifted hypothesis predicts a nonexistent edge pin.  Both axes must beat their nearest alias by
    a block-bootstrap lower bound greater than zero.  When ``allow_phase_only`` is true, an
    unresolved finite index returns a centred pitch-equivalent placement explicitly labelled
    ``uniform-phase`` instead of claiming an absolute origin.
    """
    a = template.arrays[0]
    if not np.isclose(a.pitch_x_um, a.pitch_y_um, rtol=1e-6, atol=1e-6):
        raise RegistrationAmbiguityError(
            "Finite-edge markerless registration currently requires equal X/Y lattice pitch.")
    xr_options = tuple(x_right_options)
    yu_options = tuple(y_up_options)
    if not xr_options or not yu_options:
        raise ValueError("uniform-lattice registration requires at least one X and Y orientation")
    xppx, yppx = float(scan.x_um_per_px), float(scan.y_um_per_px)
    pitch_px = 0.5 * (a.pitch_x_um / xppx + a.pitch_y_um / yppx)
    pin_mask = _adaptive_pin_mask(z0, valid, pitch_px)
    centers = _uniform_component_centers(pin_mask, a, xppx, yppx)
    if angles_deg is None:
        limits = (-5.0, 5.0)
    else:
        av = np.asarray(tuple(angles_deg), float)
        if av.size < 2:
            centre = float(av[0]) if av.size else 0.0
            limits = (centre - 0.1, centre + 0.1)
        else:
            limits = (float(av.min()), float(av.max()))
    angle, phase_u, phase_v, _inliers = _fit_uniform_lattice(
        centers, float(a.pitch_x_um), limits,
        min_components=(6 if allow_phase_only else 9))
    qu, qv, testable, present = _uniform_node_grid(
        pin_mask, valid, a, xppx, yppx, angle, phase_u, phase_v)
    n_present = int(present.sum())
    present_u = np.flatnonzero(present.any(axis=0))
    present_v = np.flatnonzero(present.any(axis=1))
    phase_patch_ok = (n_present >= 6 and min(len(present_u), len(present_v)) >= 2
                      and max(len(present_u), len(present_v)) >= 3)
    absolute_patch_ok = (n_present >= 9 and len(present_u) >= 3 and len(present_v) >= 3)
    if not (absolute_patch_ok or (allow_phase_only and phase_patch_ok)):
        raise RegistrationAmbiguityError(
            f"Only {n_present} testable pins spanning {len(present_u)}x{len(present_v)} lattice "
            "lines were found; absolute finite-edge registration needs a 3x3 patch, while "
            "phase-only registration needs at least a 2x3 patch.")

    test_prefix = np.pad(testable.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    pres_prefix = np.pad(present.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    candidates = []
    for lo_u in range(int(qu[0]) - a.nx + 1, int(qu[-1]) + 1):
        for lo_v in range(int(qv[0]) - a.ny + 1, int(qv[-1]) + 1):
            total = _rect_count(test_prefix, lo_u, lo_u + a.nx - 1,
                                lo_v, lo_v + a.ny - 1, qu, qv)
            matched = _rect_count(pres_prefix, lo_u, lo_u + a.nx - 1,
                                  lo_v, lo_v + a.ny - 1, qu, qv)
            missing = total - matched
            extra = n_present - matched
            candidates.append((missing + extra, -matched, lo_u, lo_v,
                               total, matched, missing, extra))
    candidates.sort()
    best = candidates[0]
    _err, _negmatch, best_u, best_v, total, matched, missing, extra = best
    recall = matched / max(1, n_present)
    precision = matched / max(1, total)
    min_matched = 6 if allow_phase_only and not absolute_patch_ok else 9
    if matched < min_matched or recall < 0.75 or precision < 0.70:
        raise RegistrationAmbiguityError(
            f"The best finite-array hypothesis explains only {matched}/{n_present} observed and "
            f"{matched}/{total} predicted testable pins (recall {recall:.2f}, precision "
            f"{precision:.2f}); the scan does not support this DXF reliably.")

    best_pred = _uniform_prediction(qu, qv, testable, best_u, best_v, a.nx, a.ny)
    evidence = {}
    for axis, pos in (("x", 2), ("y", 3)):
        alt = next((cnd for cnd in candidates if cnd[pos] != best[pos]), None)
        if alt is None:
            evidence[axis] = (float("inf"), float("inf"), None)
            continue
        alt_pred = _uniform_prediction(qu, qv, testable, alt[2], alt[3], a.nx, a.ny)
        raw_margin = float(alt[0] - best[0])
        boot_margin = _block_bootstrap_margin(
            present, testable, best_pred, alt_pred,
            seed=20260722 + (0 if axis == "x" else 1))
        evidence[axis] = (raw_margin, boot_margin, alt)

    # A one-pitch alias of a large N x N array differs along O(N) edge nodes while the interior
    # contains O(N^2) matches, so a fraction-of-all-pins threshold would perversely get stricter as
    # the array grows.  One percent plus the block-bootstrap guard retains ample independent edge
    # support for the 100x100 fabrication grid without rejecting its mathematically expected ~2N
    # margin after rotated-frame clipping.
    min_raw = max(5, int(np.ceil(0.01 * matched)))
    ambiguous = [axis for axis, (raw, boot, _alt) in evidence.items()
                 if raw < min_raw or boot <= 0.0]
    if ambiguous:
        detail = ", ".join(
            f"{axis}: next alias +{evidence[axis][0]:g} node loss, "
            f"bootstrap 1% bound {evidence[axis][1]:g}"
            for axis in ambiguous)
        if not allow_phase_only:
            raise RegistrationAmbiguityError(
                "Uniform markerless lattice phase was found, but the finite array index is not "
                f"identifiable along {','.join(ambiguous)} ({detail}). Capture at least one physical "
                "pin-termination edge plus roughly one pitch of valid floor in both lattice "
                "directions, enable phase-only registration, or provide a trusted manual origin.")

        # Pick a deterministic representative of the pitch-equivalent family.  Preserve every axis
        # that finite-edge evidence actually resolved; on each ambiguous axis centre the observed
        # lattice span inside the DXF array.  This maximises useful partial-image coverage while the
        # metadata below makes clear that the chosen integer index is only a coordinate convention.
        obs_u = qu[present.any(axis=0)]
        obs_v = qv[present.any(axis=1)]
        target_u = int(round(0.5 * (obs_u[0] + obs_u[-1] - (a.nx - 1))))
        target_v = int(round(0.5 * (obs_v[0] + obs_v[-1] - (a.ny - 1))))
        pool = [cnd for cnd in candidates
                if ("x" in ambiguous or cnd[2] == best_u)
                and ("y" in ambiguous or cnd[3] == best_v)]
        optimum = min((cnd[0], cnd[1]) for cnd in pool)
        pool = [cnd for cnd in pool if (cnd[0], cnd[1]) == optimum]
        best = min(pool, key=lambda cnd:
                   ((cnd[2] - target_u) ** 2 if "x" in ambiguous else 0)
                   + ((cnd[3] - target_v) ** 2 if "y" in ambiguous else 0))
        _err, _negmatch, best_u, best_v, total, matched, missing, extra = best

    xr = int(xr_options[0]); yu = int(yu_options[0])
    origin_qu = best_u if xr > 0 else best_u + a.nx - 1
    origin_qv = best_v if yu > 0 else best_v + a.ny - 1
    ou = phase_u + origin_qu * a.pitch_x_um
    ov = phase_v + origin_qv * a.pitch_y_um
    th = np.deg2rad(angle); c, s = np.cos(th), np.sin(th)
    origin_x = c * ou - s * ov; origin_y = s * ou + c * ov
    score = 2.0 * matched / max(1, total + n_present)
    return [CellPlacement(
        1, float(origin_x / xppx), float(origin_y / yppx), xppx, yppx,
        y_up=yu, x_right=xr, rotation_deg=float(angle), score=float(score),
        method=("uniform-phase" if ambiguous else "uniform-edge"), cell_row=1, cell_col=1,
        absolute_origin=not bool(ambiguous), ambiguous_axes="".join(ambiguous))]


def _marker_feature(z0, valid, pitch_px):
    """Feature map for marker matched-filtering: LOCAL height prominence ``z0 − local-background``.

    The alignment marker is detected on HEIGHT, not intensity: on the real scans the L barely shows
    in intensity (its reflectivity ~ the surround) but stands out clearly in height. Local
    prominence (background = mean over ~1.5 pin pitch) has a second benefit -- the L sits ISOLATED
    in an empty corner (background = floor -> full prominence), whereas each pin is surrounded by
    pins (background raised by neighbours -> smaller prominence), so the marker stands out even more
    than the big pins. Returns a 0..1 map (|NCC| downstream is polarity-agnostic if the L is
    recessed rather than proud)."""
    from scipy.ndimage import uniform_filter
    k = int(max(5, round(1.5 * pitch_px)))
    vf = valid.astype(np.float32)
    zf = np.where(valid, z0, 0.0).astype(np.float32)
    bg = (uniform_filter(zf, size=k, mode="nearest")
          / np.maximum(uniform_filter(vf, size=k, mode="nearest"), 1e-6))
    prom = np.where(valid, z0 - bg, 0.0)
    if not valid.any():
        return np.zeros_like(prom)
    lo, hi = np.percentile(prom[valid], 2), np.percentile(prom[valid], 99.5)
    return np.clip((prom - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _pattern_ncc(img, tpl, decisive=False):
    """Normalized cross-correlation of ``img`` with ``tpl``, FULL mode, in [-1, 1].

    Plain correlation is biased toward dense regions of the (fairly full) local pin mask, which
    pulls coarse peaks onto spurious high-fill spots. NCC divides by the local image energy under
    the template, so a peak reflects PATTERN agreement, not local density -- essential for
    accurately localising every cell. 'full' mode keeps the lag->origin mapping unambiguous: a
    peak at (k,l) means the template's top-left sits at img (k-Ht+1, l-Wt+1).

    Delegates to :func:`accel.pattern_ncc`; the CPU/scipy path is byte-identical to the historical
    implementation.  Pass ``decisive=True`` when the peak VALUE (not just its position) selects a
    reflection/rotation, so that comparison always runs on deterministic CPU float64 rather than a
    float32 GPU result."""
    return accel.pattern_ncc(img, tpl, decisive=decisive)


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
        # rotate-in-um-then-scale-PER-AXIS (matches dxf_to_px): the col component divides by xppx and
        # the row component by yppx. The rotation-coupled term must use the OTHER axis' scale -- on
        # non-square pixels the old (both-same-axis) form mis-sized the +-1-pitch de-alias probe.
        vx = np.array([c * x_right * a.pitch_x_um / xppx, s * x_right * a.pitch_x_um / yppx])
        vy = np.array([-s * y_up * a.pitch_y_um / xppx, c * y_up * a.pitch_y_um / yppx])
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
                         coarse_cell_px=120.0, allow_uniform_phase_only=False):
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
    alias_reason = _uniform_lattice_alias_reason(template)
    if alias_reason:
        if scan is None or z0 is None or valid is None:
            raise RegistrationAmbiguityError(
                f"Cannot test finite-edge evidence: {alias_reason}. Supply scan data that includes "
                "physical pin terminations in both directions, add/detect an asymmetric alignment "
                "fiducial, or provide a trusted manual origin.")
        return _register_uniform_lattice(
            scan, template, z0, valid, x_right_options=x_right_options,
            y_up_options=y_up_options, angles_deg=angles_deg,
            allow_phase_only=allow_uniform_phase_only)
    xppx, yppx = scan.x_um_per_px, scan.y_um_per_px
    H, W = scan.height_raw.shape
    w_um, h_um = template.size_um                          # cell size from the DXF
    Pxpx = (cell_pitch_um[0] if cell_pitch_um else w_um) / xppx
    Pypx = (cell_pitch_um[1] if cell_pitch_um else h_um) / yppx
    min_sep = 0.5 * min(Pxpx, Pypx)
    auto_angles = angles_deg is None
    if auto_angles:
        # Broad enough for realistic stage placement, but still coarse here; only the winning
        # reflection/angle gets the second downsampled refinement and full-resolution polish.
        angles_deg = np.arange(-5.0, 5.0001, 0.5)
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
                corr = _pattern_ncc(img, T, decisive=True)   # peak value selects reflection+angle
                pv = float(corr.max())
                if best is None or pv > best[0]:
                    best = (pv, xr, yu, ang, corr, t0c, t0r, Ht, Wt)
    if best is None:
        return []
    if auto_angles:
        # Refine only the winning reflection around its coarse angle before extracting peaks.
        _, best_xr, best_yu, best_ang, *_ = best
        for ang2 in np.arange(max(-5.0, best_ang - 0.5),
                              min(5.0, best_ang + 0.5) + 0.0501, 0.1):
            T, t0c, t0r, Ht, Wt = _cell_pin_pattern(
                template, xppx, yppx, best_xr, best_yu, float(ang2), ds)
            if T.sum() < 5:
                continue
            corr2 = _pattern_ncc(img, T, decisive=True)      # peak value selects the fine angle
            pv2 = float(corr2.max())
            if pv2 > best[0]:
                best = (pv2, best_xr, best_yu, float(ang2), corr2,
                        t0c, t0r, Ht, Wt)
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


def _rasterize_marker(cell, xppx, yppx, x_right, y_up, rot_deg, pad_px=4):
    """Filled binary raster of the cell's alignment-marker polygon (``marker_polygon_um`` -- verts
    in um relative to the DESIGN origin) under the given reflection/rotation, matching dxf_to_px.

    This is the matched-filter TEMPLATE: the actual marker SHAPE (an asymmetric L, or the deprecated
    square). Using the real shape is what discriminates the marker from the round pins -- a filled
    square latched onto the large D300 pins. Returns (T, t0c, t0r) with (t0c, t0r) = the pixel of
    CAD origin (0,0) inside T, so a detection maps straight to the cell origin even for an inset L."""
    from matplotlib.path import Path as MplPath
    poly = np.asarray(cell.marker_polygon_um, float)
    th = np.deg2rad(rot_deg); c, s = np.cos(th), np.sin(th)
    xr = x_right * poly[:, 0]; yr = y_up * poly[:, 1]      # reflect + rotate in um, then scale per axis
    xr, yr = c * xr - s * yr, s * xr + c * yr
    col = xr / xppx; row = yr / yppx                       # px relative to CAD (0,0)
    offc = float(col.min()) - pad_px; offr = float(row.min()) - pad_px
    Wt = int(np.ceil(float(col.max()) + pad_px - offc)) + 1
    Ht = int(np.ceil(float(row.max()) + pad_px - offr)) + 1
    gx, gy = np.meshgrid(np.arange(Wt), np.arange(Ht))
    pts = np.column_stack([gx.ravel() + offc, gy.ravel() + offr])
    inside = MplPath(np.column_stack([col, row])).contains_points(pts).reshape(Ht, Wt)
    return inside.astype(float), -offc, -offr


def _solid_square_ncc_markers(feat, marker_px, min_sep, max_n=40, thresh=0.30):
    """Locate the deprecated SOLID-SQUARE alignment marker by |NCC| of a filled-square template
    against the (intensity) feature map. This is the original, proven square detector -- kept for
    square-marker samples; the L fiducial uses ``_detect_marker_origins`` on the height map instead.
    Returns [(col, row, score)] marker CENTRES, strongest first (the caller maps centre -> origin)."""
    sq = max(5, int(round(marker_px)))
    marg = max(2, int(round(0.35 * sq)))
    T = np.zeros((sq + 2 * marg, sq + 2 * marg))
    T[marg:marg + sq, marg:marg + sq] = 1.0
    Ht, Wt = T.shape
    t0 = marg + sq / 2.0
    corr = np.abs(_pattern_ncc(feat, T, decisive=True))   # peak value feeds count/reflection/accept
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


def _detect_marker_origins(feat, cell, xppx, yppx, x_right, y_up, rot, min_sep,
                           max_n=60, thresh=0.2):
    """Locate alignment markers by |NCC| of the rasterised marker-polygon template against the
    feature map, returning each as a CELL ORIGIN (col, row) -- the template carries the origin
    reference (``t0``), so an asymmetric/inset L maps to the true origin directly (no square-centre
    assumption). |NCC| is polarity-agnostic (marker milled proud OR recessed). Over-detection is
    harmless (the overlap/lattice gates downstream reject false peaks). Strongest first."""
    T, t0c, t0r = _rasterize_marker(cell, xppx, yppx, x_right, y_up, rot)
    Ht, Wt = T.shape
    if T.sum() < 3:
        return []
    corr = np.abs(_pattern_ncc(feat, T, decisive=True))   # peak value feeds count/reflection/accept
    sep = max(2, int(round(min_sep)))
    out = []
    for _ in range(max_n):
        idx = int(np.argmax(corr)); k, l = np.unravel_index(idx, corr.shape)
        v = corr[k, l]
        if not np.isfinite(v) or v < thresh:
            break
        out.append(((l - Wt + 1) + t0c, (k - Ht + 1) + t0r, float(v)))
        r0, r1 = max(0, k - sep), min(corr.shape[0], k + sep + 1)
        c0, c1 = max(0, l - sep), min(corr.shape[1], l + sep + 1)
        corr[r0:r1, c0:c1] = -np.inf
    return out


def _detect_marker_origins_rotated(features, cell, xppx, yppx, x_right, y_up,
                                   min_sep, max_n=60, span=5.0, coarse_step=1.0,
                                   fine_step=0.25, thresh=0.2, n_angle_hypotheses=3):
    """Rotation-aware marker search with a downsampled coarse-to-fine angle sweep.

    Stage rotation is global. The expensive angle sweep runs on a reduced feature map (marker long
    side ~=20 px), then only a small shortlist (including zero, to protect near-aligned real scans
    from border-driven maxima) gets full-resolution NCC. Downstream pin-pattern validation chooses
    among the hypotheses. Returns
    ``[(origin_col, origin_row, ncc, angle_deg), ...]``.
    """
    features = [np.asarray(f, float) for f in features if f is not None]
    if not features:
        return []
    poly = np.asarray(cell.marker_polygon_um, float)
    marker_span_px = max(np.ptp(poly[:, 0]) / xppx, np.ptp(poly[:, 1]) / yppx)
    ds = max(1, int(round(marker_span_px / 20.0)))
    coarse_features = [f[::ds, ::ds] for f in features]

    def _angle_score(angle):
        T, _, _ = _rasterize_marker(
            cell, xppx * ds, yppx * ds, x_right, y_up, float(angle))
        if T.sum() < 3:
            return -1.0
        score = -1.0
        for f in coarse_features:
            corr = np.abs(_pattern_ncc(f, T, decisive=True))  # peak value scores this angle
            if corr.size:
                score = max(score, float(np.max(corr)))
        return score

    coarse_angles = np.arange(-span, span + 0.5 * coarse_step, coarse_step)
    scored = [(_angle_score(a), float(a)) for a in coarse_angles]
    if max(scored, key=lambda t: t[0])[0] < thresh:
        return []
    centres = []
    for score, angle in sorted(scored, reverse=True):
        if score >= thresh and not any(abs(angle - a) < 0.75 * coarse_step for a in centres):
            centres.append(angle)
        if len(centres) >= max(1, n_angle_hypotheses - 1):
            break
    if not any(abs(a) < 0.5 * coarse_step for a in centres):
        centres.append(0.0)

    angles = []
    for centre in centres[:n_angle_hypotheses]:
        best_score, best_angle = _angle_score(centre), float(centre)
        fine_lo = max(-span, centre - coarse_step)
        fine_hi = min(span, centre + coarse_step)
        for angle in np.arange(fine_lo, fine_hi + 0.5 * fine_step, fine_step):
            score = _angle_score(angle)
            if score > best_score:
                best_score, best_angle = score, float(angle)
        if not any(abs(best_angle - a) < 0.5 * fine_step for a in angles):
            angles.append(best_angle)

    # Keep feature channels *and angle hypotheses* independent until pin-overlap refinement.  On the
    # real 72026 scan, a slightly stronger -1-degree marker NCC lies within one marker-width of the
    # genuine zero-degree candidate, while a height false peak similarly crowds an intensity marker.
    # Deduplicating either dimension here discards the right absolute anchor before the pin lattice
    # gets a vote.  ``_detect_marker_origins`` already bounds and peels each feature/angle search, so
    # this remains bounded by max_n * n_features * n_angle_hypotheses.
    out = []
    for angle in angles:
        for f in features:
            out.extend((*hit, angle) for hit in _detect_marker_origins(
                f, cell, xppx, yppx, x_right, y_up, angle, min_sep,
                max_n=max_n, thresh=thresh))
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
                     span=5.0, step=0.1, coarse_step=0.25):
    """Best single stage-rotation (deg) for the whole sample, from an overlap sweep on one
    well-registered anchor cell. Stage rotation is global, so one value serves every cell.

    Coarse-to-fine so a several-degree stage rotation is recoverable without a fine sweep over the
    whole ±span: a coarse pass (``coarse_step``) finds the ballpark, then a fine pass (``step``)
    refines within ±coarse_step of it. (The previous flat ±1.2° sweep could not reach a typical
    multi-degree rotation and clamped to the window edge — see review finding #19.) The final
    resolution is still ``step``, so small-rotation results match the old behaviour."""
    def _score(a):
        _, ov = _refine_origin(amask, template, origin, xppx, yppx, y_up, x_right, float(a),
                               search_px=int(max_shift) + 6, max_shift_px=max_shift)
        return ov if np.isfinite(ov) else -1.0

    best_a, best_ov = 0.0, _score(0.0)
    nc = int(round(2 * span / coarse_step))                # coarse pass over the full ±span
    for k in range(nc + 1):
        ov = _score(-span + k * coarse_step)
        if ov > best_ov:
            best_ov, best_a = ov, float(-span + k * coarse_step)
    nf = int(round(2 * coarse_step / step))                # fine pass within ±coarse_step of the best
    lo = best_a - coarse_step
    for k in range(nf + 1):
        a = lo + k * step
        if abs(a) <= span:
            ov = _score(a)
            if ov > best_ov:
                best_ov, best_a = ov, float(a)
    return best_a


def _register_rotated_marker_cells(feat, z0, valid, amask, template, xppx, yppx,
                                   x_right_options, y_up_options, min_sep, max_shift,
                                   min_overlap, max_n):
    """Rotation-aware marker registration shared by ``register_scan``'s auto path.

    Every reflection option gets its own L-marker angle search and pin-overlap polish. The winning
    option must explain the requested number of cells, then wins on marker and pin evidence.
    Returns only finite marker-anchored placements.
    """
    poly = np.asarray(template.marker_polygon_um, float)
    marker_span = max(np.ptp(poly[:, 0]) / xppx, np.ptp(poly[:, 1]) / yppx)
    marker_sep = max(0.6 * marker_span, 20.0)
    mean_pitch_px = float(np.mean([a.pitch_x_um for a in template.arrays])) / xppx
    mfeat = _marker_feature(z0, valid, mean_pitch_px)
    options = []

    for xr in x_right_options:
        for yu in y_up_options:
            markers = _detect_marker_origins_rotated(
                (mfeat, feat), template, xppx, yppx, xr, yu, marker_sep,
                max_n=max(8, 2 * max_n))
            if not markers:
                continue

            cand = []
            for oc, orow, marker_sc, marker_rot in markers:
                o, ov = _refine_origin(
                    amask, template, (oc, orow), xppx, yppx, yu, xr, marker_rot,
                    search_px=int(max_shift) + 6, max_shift_px=max_shift)
                if np.isfinite(ov) and ov >= min_overlap:
                    cand.append((float(o[0]), float(o[1]), float(ov), float(marker_sc),
                                 float(marker_rot)))
            if not cand:
                continue
            # Pin-pattern overlap is the registration score and the most stable discriminator;
            # marker NCC breaks ties but must not pull a good origin toward a nearby visual peak.
            cand.sort(key=lambda t: (-t[2], -t[3]))
            dedup = []
            for c in cand:
                if not any((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2 < min_sep ** 2
                           for k in dedup):
                    dedup.append(c)
            if not dedup:
                continue

            prelim = dedup[:max_n]
            mean_ov = float(np.mean([c[2] for c in prelim]))
            marker_top = prelim[0][3]
            quality = (len(prelim), marker_top + 2.0 * mean_ov, mean_ov)
            options.append((quality, xr, yu, prelim))

    if not options:
        return []
    _, xr, yu, chosen = max(options, key=lambda t: t[0])
    anchor = max(chosen, key=lambda t: (t[2], t[3]))
    rot = _global_rotation(
        amask, template, (anchor[0], anchor[1]), xppx, yppx, yu, xr, max_shift)
    angle_consistent = [c for c in chosen if abs(c[4] - rot) <= 0.6]
    if angle_consistent:
        chosen = angle_consistent
    polished = []
    for c in chosen:
        o, ov = _refine_origin(
            amask, template, (c[0], c[1]), xppx, yppx, yu, xr, rot,
            search_px=int(max_shift) + 6, max_shift_px=max_shift)
        if np.isfinite(ov) and ov >= min_overlap:
            polished.append((float(o[0]), float(o[1]), float(ov), c[3]))
    polished.sort(key=lambda t: (-t[3], -t[2]))
    return [CellPlacement(
        0, c[0], c[1], xppx, yppx, y_up=yu, x_right=xr,
        rotation_deg=rot, score=c[2], method="marker+lattice") for c in polished[:max_n]]


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


def _pitch_reps(template):
    """A reduced template carrying ONE array per distinct (pitch_x, pitch_y). De-aliasing against
    this tests off-by-one on every lattice period the cell contains while rasterizing only a
    representative array per pitch -- not all of them, which is the cost driver on many-array cells
    (a 14-array cell took ~40 min otherwise). Off-by-one still shows up because each representative
    array is finite: a one-pitch shift drops an edge row/column and lowers the overlap."""
    seen, reps = set(), []
    for a in template.arrays:
        key = (round(a.pitch_x_um, 1), round(a.pitch_y_um, 1))
        if key not in seen:
            seen.add(key)
            reps.append(a)
    return SimpleNamespace(arrays=reps)


def _probe_lattice(nodes, template, amask, z0, valid, xppx, yppx, y_up, x_right, rot,
                   max_shift, W, H, min_overlap, min_inbounds, min_contrast, dealias_tpl=None):
    """Register every predicted lattice node: a phase-clamped refine, then -- only where a cell is
    actually present (the refine clears the overlap gate) AND ``dealias_tpl`` is given -- an
    off-by-one-pin de-alias to lock the true origin. ``dealias_tpl`` is a reduced template (one
    array per distinct pitch, from :func:`_pitch_reps`), evaluated at its FULL window, so the
    de-alias tests every lattice period without rasterizing all arrays; pass None for the cheap
    combo-selection probes. Nodes far from any real cell fail the gate and are dropped."""
    out = []
    for (oc, orow) in nodes:
        o, ov = _refine_origin(amask, template, (oc, orow), xppx, yppx, y_up, x_right, rot,
                               search_px=int(max_shift) + 6, max_shift_px=max_shift)
        if not np.isfinite(ov) or ov < min_overlap:
            continue                                    # no cell at this node -> skip (no de-alias)
        if dealias_tpl is not None:
            od, _ = _dealias_origin(amask, dealias_tpl, o, xppx, yppx, y_up, x_right, rot)
            od, ovd = _refine_origin(amask, template, od, xppx, yppx, y_up, x_right, rot,
                                     search_px=int(max_shift) + 6, max_shift_px=max_shift)
            if np.isfinite(ovd) and ovd >= ov:          # keep the de-aliased lock only if not worse
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
                    mirror_x=True, y_up=1, anchor_frac=0.85,
                    allow_uniform_phase_only=True):
    """Detect and register EVERY unit cell tiled across an assembled sample (Tier 1).

    Marker-anchored **lattice** registration (quality over runtime):

    1. Detect alignment markers by matched-filtering the marker's own POLYGON shape
       (``marker_polygon_um``, rasterised) against the feature map -- NOT a filled square, which on
       a large-pin sample (D300) correlates more strongly with the round pins than with the thin L
       and mislocks. The L fiducial is detected on the HEIGHT-prominence map (it barely shows in
       intensity); the deprecated square on intensity. The marker sits in an empty corner, OFF the
       pin lattice, so it is an *absolute* origin reference (the template carries the origin, so an
       inset/asymmetric L maps straight to the cell origin); a phase-clamped refine plus off-by-one
       de-alias then locks each to the true origin. This is what makes a dense uniform periodic
       array resolvable -- an overlap gate alone cannot tell a real placement from a
       one-pin-pitch-shifted one, but the marker can.
    2. Take the highest-overlap detections as anchors, estimate the tile step vectors from their
       geometry (the tile pitch is NOT in the single-cell DXF, so it is measured here), and fix the
       one global stage rotation.
    3. Probe EVERY predicted lattice node and keep those clearing the overlap / in-bounds /
       ablation-contrast gates, then re-fit the lattice from the survivors and re-probe so a large
       grid never drifts a pin off at the far corners. Off-lattice spurious detections (e.g. a pin
       mistaken for the marker) are never on a node, so they are rejected by construction.

    The reflection is the known Keyence X-mirror (``mirror_x`` -> x_right=-1) so the array is never
    flipped; survivors are indexed in the DESIGN frame ((1,1) = DXF top-left) by
    ``_assign_grid_indices``. Falls back to :func:`_register_by_pattern` when no marker polygon is
    present or none anchors a cell. Aperiodic layouts use pin-pattern correlation; a markerless
    complete uniform array uses finite pin-termination evidence. If its absolute indices remain
    ambiguous, ``allow_uniform_phase_only`` permits a visibly labelled lattice-phase placement so a
    single subsection can still be measured; set it false wherever absolute pin identity is needed.
    """
    xppx, yppx = scan.x_um_per_px, scan.y_um_per_px   # x_right/y_up are resolved from the marker below
    H, W = scan.height_raw.shape
    feat, edge, pin_mask, valid, z0 = scan_feature(scan)

    def _fallback(reason):
        print(f"register_sample: {reason}; falling back to marker-free registration.")
        found = _register_by_pattern(
            scan, template, z0, valid, cell_pitch_um=cell_pitch_um,
            min_inbounds=min_inbounds, min_contrast_um=min_contrast_um,
            x_right_options=((-1, 1) if mirror_x else (1, -1)),
            allow_uniform_phase_only=allow_uniform_phase_only)
        for p in found:
            if not p.absolute_origin:
                print("WARNING: markerless uniform subsection registered by lattice phase only; "
                      f"absolute pin index is unresolved on {p.ambiguous_axes.upper()} axis/axes. "
                      "Geometry measurements are aligned, but do not interpret the reported DXF "
                      "origin or individual pin indices as absolute.")
        return found

    w_um, h_um = template.size_um
    cell_w = (cell_pitch_um[0] if cell_pitch_um else w_um) / xppx
    cell_h = (cell_pitch_um[1] if cell_pitch_um else h_um) / yppx
    min_sep = 0.5 * min(cell_w, cell_h)                  # cells cannot sit closer than ~a cell
    min_pin_pitch = min(a.pitch_x_um for a in template.arrays) / xppx
    max_shift = 0.3 * min_pin_pitch                      # marker-anchored polish: well below half pitch
    # Lattice NODES are predicted from an averaged step, not marker-anchored, so on a large grid
    # with a non-uniform tile step they drift; allow the node refine a larger (still sub-half-pitch,
    # so alias-safe) snap so far cells are not stranded outside the clamp and under-counted.
    probe_shift = 0.45 * min_pin_pitch
    mean_pitch_px = float(np.mean([a.pitch_x_um for a in template.arrays])) / xppx
    amask = _adaptive_pin_mask(z0, valid, mean_pitch_px)

    # --- 1. detect every alignment marker as a cell ORIGIN, BY MARKER TYPE (read from the DXF), each
    #        with its proven method. The reflection is the known, fixed Keyence X-mirror
    #        (``mirror_x``): the pin lattice is reflection-symmetric and the marker |NCC| does not
    #        reliably resolve the flip on these pin-dominated feature maps, so we trust the hardware
    #        mirror rather than a fragile asymmetry vote. False peaks (chip border, pins) are rejected
    #        by the pin-overlap + lattice gates below. ---
    if template.marker_polygon_um is None or len(template.marker_polygon_um) < 3:
        return _fallback("template has no alignment-marker polygon")
    x_right, y_up = (-1 if mirror_x else 1), int(y_up)
    ex = max(8, 2 * expect_max)
    poly = np.asarray(template.marker_polygon_um, float)
    if template.marker_shape == "L":
        # L fiducial: rasterised-polygon matched filter -> the detection IS the cell origin (the
        # template carries the origin, correct for the inset/asymmetric L). Search the HEIGHT-
        # prominence map (the L barely shows in intensity -- reflectivity ~ the surround) AND
        # intensity, unioned (a given sample's Ls may show in one or the other). Peel at MARKER
        # scale -- a cell-scale sep merged adjacent cells' markers and lost them.
        mfeat = _marker_feature(z0, valid, mean_pitch_px)
        marker_sep = max(0.6 * max(np.ptp(poly[:, 0]) / xppx, np.ptp(poly[:, 1]) / yppx), 20.0)
        markers = _detect_marker_origins_rotated(
            (mfeat, feat), template, xppx, yppx, x_right, y_up, marker_sep, max_n=ex)
    else:
        # Deprecated square marker: preserve the proven filled-square intensity detector exactly.
        # A square carries no orientation evidence, and rotating its outline changed the established
        # 718 small-angle phase. Multi-degree automatic recovery is supported by asymmetric L
        # fiducials (and aperiodic marker-free patterns), not by this legacy symmetric marker.
        marker_um = template.marker_size_um if np.isfinite(template.marker_size_um) else 200.0
        mx, my = marker_um / xppx, marker_um / yppx
        marker_px = 0.5 * (mx + my)
        centres = _solid_square_ncc_markers(
            feat, marker_px, max(0.6 * marker_px, 20), max_n=ex)
        markers = [(mc - x_right * 0.5 * mx, mr - y_up * 0.5 * my, sc, 0.0)
                   for (mc, mr, sc) in centres]
    if not markers:
        return _fallback("no alignment marker detected")

    cand = []
    for (oc, orow, marker_sc, marker_rot) in markers:
        # Each detection is already a CELL ORIGIN. A phase-clamped refine (<half a pin pitch) polishes
        # it against the pin lattice without aliasing; off-by-one is resolved later, per lattice node.
        o, ov = _refine_origin(amask, template, (oc, orow), xppx, yppx, y_up, x_right, marker_rot,
                               search_px=int(max_shift) + 6, max_shift_px=max_shift)
        if np.isfinite(ov) and ov >= min_overlap:
            cand.append((float(o[0]), float(o[1]), float(ov), float(marker_sc),
                         float(marker_rot)))
    # Joint evidence is essential: marker NCC supplies absolute phase while pin overlap rejects
    # borders and round pins that resemble the fiducial.  Feature channels and angle hypotheses
    # have deliberately not been deduplicated yet, so a strong false marker cannot crowd out the
    # correct lower-NCC channel before this combined score is available.
    if template.marker_shape == "L":
        cand.sort(key=lambda t: -(t[2] + 0.5 * t[3]))
    else:
        cand.sort(key=lambda t: -t[2])
    dedup = []
    for c in cand:
        if not any((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2 < min_sep ** 2 for k in dedup):
            dedup.append(c)
    if not dedup:
        return _fallback("no marker-anchored cell cleared the overlap gate")

    O0 = dedup[0]                                        # highest overlap = surest phase anchor
    rot = _global_rotation(amask, template, (O0[0], O0[1]), xppx, yppx, y_up, x_right, max_shift)

    # The sample has one physical stage angle.  Candidate marker hypotheses far from the recovered
    # pin-pattern angle are visual false positives; letting them vote on the cell-to-cell step can
    # create a pitch-aliased lattice even though the absolute marker anchors are correct.  The
    # 0.6-degree window is deliberately wider than the 0.25-degree marker and 0.1-degree pin-angle
    # refinements, while separating adjacent coarse hypotheses on the real 72026 scan.
    angle_consistent = [c for c in dedup if abs(c[4] - rot) <= 0.6]
    if angle_consistent:
        dedup = angle_consistent
    O0 = (max(dedup, key=lambda t: t[2] + 0.5 * t[3])
          if template.marker_shape == "L" else max(dedup, key=lambda t: t[2]))

    # --- 2. tile step vectors: prefer high-overlap anchors, fall back to all cells per axis ---
    top = O0[2]
    anchors = [c for c in dedup if c[2] >= max(min_overlap, anchor_frac * top)]
    col_vecs, row_vecs = _lattice_step_candidates(anchors, cell_w, cell_h)
    col_all, row_all = _lattice_step_candidates(dedup, cell_w, cell_h)
    col_vecs = col_vecs or col_all
    row_vecs = row_vecs or row_all

    if not col_vecs or not row_vecs:                     # no 2-D lattice (single cell/row/col)
        # A several-degree marker sweep leaves a small angle-dependent origin error.  Polish the
        # surviving one-dimensional anchors at the recovered global angle before returning them.
        # The 2-D path deliberately keeps its original marker seeds for step estimation (that is
        # what preserves the established 72026 phase), then its lattice probe performs the polish.
        polished = []
        for c in dedup:
            o, ov = _refine_origin(
                amask, template, (c[0], c[1]), xppx, yppx, y_up, x_right, rot,
                search_px=int(max_shift) + 6, max_shift_px=max_shift)
            if np.isfinite(ov) and ov >= min_overlap:
                polished.append((float(o[0]), float(o[1]), float(ov), c[3], c[4]))
        if polished:
            dedup = polished
        # Pitch-valid false origins can form a parallel row on a periodic design. Choose the
        # longest collinear family; joint marker+pin evidence breaks equal-length ties.  Ranking
        # mean evidence first lets a single excellent false origin beat a complete tiled row.
        families = []
        for vals, tol in (([c[1] for c in dedup], 0.30 * cell_h),
                          ([c[0] for c in dedup], 0.30 * cell_w)):
            for group in _cluster_1d(vals, max(tol, 1.0)):
                families.append([dedup[i] for i, _ in group])
        if families:
            def _family_evidence(fam):
                return float(np.mean([
                    c[2] + (0.5 * c[3] if template.marker_shape == "L" else 0.0)
                    for c in fam]))
            dedup = max(families, key=lambda fam: (len(fam), _family_evidence(fam)))
        # One-dimensional layouts used to return the marker candidates directly. That skipped
        # the same de-alias, in-bounds, overlap, and ablation-contrast gates used for 2-D scans.
        # Run every surviving anchor through the shared probe so single rows/columns cannot admit
        # a periodic false lock or a flat/unablated apparent marker.
        kept = _dedup_cells(
            _probe_lattice(
                [(c[0], c[1]) for c in dedup], template, amask, z0, valid,
                xppx, yppx, y_up, x_right, rot, probe_shift, W, H,
                min_overlap, min_inbounds, min_contrast_um,
                dealias_tpl=_pitch_reps(template),
            ),
            min_sep,
        )
        if not kept:
            return _fallback("one-dimensional marker anchors failed the cell-quality gates")
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
                               probe_shift, W, H, min_overlap, min_inbounds, min_contrast_um),
                min_sep)
            score = (len(placed), len(nodes))            # most cells, then finest -- a doubled step
            if best is None or score > best[0]:          # ties the count but under-probes columns
                best = (score, placed)
    kept = best[1] if best else []

    # --- 4. Re-fit the lattice from the step-3 survivors and re-probe the FULL scan extent WITH the
    #        off-by-one de-alias (one representative array per pitch, full window). The affine
    #        (row,col)->origin fit averages out the tile-step non-uniformity, and re-deriving the
    #        step vectors from it then probing every in-scan node (via _grid_nodes, not just the
    #        survivors' bbox) recovers far cells the rigid step-3 lattice drifted off -- with the
    #        larger probe_shift clamp absorbing the residual. Off-lattice / off-scan nodes fail the
    #        overlap / in-bounds / contrast gates, so spurious marker hits are never resurrected. ---
    reps = _pitch_reps(template)
    if len(kept) >= 4:
        _assign_grid_indices(kept, cell_w, cell_h)
        A = np.array([[p.cell_col, p.cell_row, 1.0] for p in kept])
        kx, *_ = np.linalg.lstsq(A, np.array([p.origin_col for p in kept]), rcond=None)
        ky, *_ = np.linalg.lstsq(A, np.array([p.origin_row for p in kept]), rcond=None)
        col_vec = (float(kx[0]), float(ky[0]))               # d origin / d cell_col
        row_vec = (float(kx[1]), float(ky[1]))               # d origin / d cell_row
        node11 = (float(np.array([1.0, 1.0, 1.0]) @ kx), float(np.array([1.0, 1.0, 1.0]) @ ky))
        nodes = _grid_nodes(node11, col_vec, row_vec, W, H, cell_w, cell_h)
    else:
        nodes = [(p.origin_col, p.origin_row) for p in kept]   # too few cells for an affine fit
    locked = _dedup_cells(
        _probe_lattice(nodes, template, amask, z0, valid, xppx, yppx, y_up, x_right, rot,
                       probe_shift, W, H, min_overlap, min_inbounds, min_contrast_um,
                       dealias_tpl=reps),
        min_sep)
    if len(locked) >= len(kept):
        kept = locked
    if not kept:
        return _fallback("lattice probe verified no cell")

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
