"""
Per-array pin-fin geometry from a registered VK4 scan (v2).

v1 re-derived the pin lattice from each field of view by 2-D autocorrelation, because it
had no independent knowledge of where the pins were. v2 reads the design from the DXF and
registers it onto the scan (see ``register.py``), so for every array we already know the
exact pitch and the pin phase. We use that **known lattice** as the primary path:

  * folding bins pixels on the *known* pitch (no autocorrelation guess needed);
  * the pin/floor/debris classification marks pins at their *known* centres, so depth is
    measured from the true trench floor even when debris buries the periodic signal that
    autocorrelation relies on.

The autocorrelation is still run to report a *measured* pitch and a lattice-strength QC
number, and it is the fallback when a scan is analysed without registration.

The heavy lifting (levelling, folding, radial profile, clean-floor depth) is the same
maths as v1 ``extract.py``; here it is driven per-array off a crop of the big scan.
One :class:`PinFinResult` is produced per array per unit cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch


# ----------------------------------------------------------- lightweight VK4 #
@dataclass
class ScanCrop:
    """A rectangular crop of a VK4 scan, quacking like ``vk4.VK4`` for the maths below."""

    height_raw: np.ndarray
    intensity: np.ndarray | None
    x_um_per_px: float
    y_um_per_px: float
    z_um_per_digit: float

    @property
    def height_um(self) -> np.ndarray:
        return self.height_raw.astype(np.float64) * self.z_um_per_digit

    @property
    def width(self) -> int:
        return self.height_raw.shape[1]

    @property
    def height(self) -> int:
        return self.height_raw.shape[0]


@dataclass
class ArraySample:
    """Design metadata for one array (the v2 analogue of v1's ``SampleFile``)."""

    filename: str                # "<vk4stem>_cell{c}_b{band}c{col}_D{dia}"
    vk4_stem: str
    cell_id: int
    array_id: int
    band: int
    col: int
    passes: int                  # laser passes for this cell (user CSV)
    speed: float                 # scan speed mm/s for this cell (user CSV)
    nominal_diameter_um: float   # DRAWN diameter from the DXF
    target_diameter_um: float    # band reference diameter (median of the band)
    nominal_pitch_um: float
    nominal_pitch_x_um: float
    nominal_pitch_y_um: float
    nx: int
    ny: int
    cx_um: float                 # array centroid in the cell (marker-relative um)
    cy_um: float
    cell_label: str = ""         # optional user label for the cell

    @property
    def sample(self) -> str:
        return f"P{self.passes}_S{self.speed:g}"

    @property
    def dose_ratio(self) -> float:
        return self.passes / self.speed if self.speed else float("nan")


@dataclass
class PinFinResult:
    filename: str
    # design / provenance
    vk4_stem: str = ""
    cell_id: int = 0
    array_id: int = 0
    band: int = 0
    col: int = 0
    passes: int = 0
    speed: float = float("nan")
    cx_um: float = float("nan")
    cy_um: float = float("nan")
    # measured geometry
    pitch_um: float = float("nan")
    pitch_x_um: float = float("nan")
    pitch_y_um: float = float("nan")
    diameter_um: float = float("nan")        # mid-height (0.50 * depth)
    base_diameter_um: float = float("nan")   # near floor (0.15 * depth) -> widest
    top_diameter_um: float = float("nan")    # near plateau (0.85 * depth) -> narrowest
    depth_um: float = float("nan")
    floor_um: float = float("nan")
    top_um: float = float("nan")
    lattice_strength: float = float("nan")
    coverage: float = float("nan")
    n_cells: float = float("nan")
    # design references
    nominal_diameter_um: float = float("nan")
    target_diameter_um: float = float("nan")
    nominal_pitch_um: float = float("nan")
    reg_score: float = float("nan")
    reg_method: str = ""
    flags: str = ""
    floor_flatness_um: float = float("nan")
    debris_fraction: float = float("nan")
    pin_sat_frac: float = float("nan")
    # grid + thumbnail for the alignment-overlay montage (not written to CSV)
    grid_phase: tuple = (0.0, 0.0)
    grid_px_px: float = 0.0
    grid_py_px: float = 0.0
    thumb: object = None
    thumb_down: int = 1

    def as_row(self) -> dict:
        d = self.__dict__.copy()
        for k in ("grid_phase", "thumb", "thumb_down", "grid_px_px", "grid_py_px"):
            d.pop(k, None)
        return d


# ================================ core maths (ported from v1 extract.py) ==== #
def _level_floor(z, valid):
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


def _measure_pitch_px(z0, valid, nominal_px):
    """First autocorrelation side-peak lag (px) along x and y + periodicity strength."""
    m = valid & np.isfinite(z0)
    if m.sum() < 50:
        return nominal_px, nominal_px, 0.0
    zc = np.where(m, z0 - np.nanmedian(z0[m]), 0.0)
    F = np.fft.fft2(zc)
    ac = np.fft.fftshift(np.fft.ifft2(F * np.conj(F)).real)
    cy, cx = np.array(ac.shape) // 2
    center_val = ac[cy, cx]

    def side_peak(line, center):
        lo = int(0.5 * nominal_px)
        hi = int(min(1.8 * nominal_px, len(line) - center - 1))
        if hi <= lo:
            return nominal_px, 0.0
        seg = line[center + lo:center + hi]
        k = int(np.argmax(seg))
        return lo + k, float(seg[k] / center_val if center_val else 0.0)

    px_lag, sx = side_peak(ac[cy, :], cx)
    py_lag, sy = side_peak(ac[:, cx], cy)
    return px_lag, py_lag, 0.5 * (sx + sy)


def _stack_pins(field, valid, centers_local, px_px, py_px, reach=0.55):
    """Mean pin by STACKING the patches centred on each known pin (from the DXF lattice).

    Superior to a modulo-fold for a tightly-cropped array: there is no wrap seam and no
    edge bias, and the mean pin is exactly centred (radial profile is taken about the patch
    centre). Each pin contributes a (2*reach*pitch)-wide window; only valid pixels count.
    Returns (mean_patch, count_patch); mean_patch centre = (hwy, hwx).
    """
    hwx = int(np.ceil(reach * px_px))
    hwy = int(np.ceil(reach * py_px))
    ph, pw = 2 * hwy + 1, 2 * hwx + 1
    acc = np.zeros((ph, pw)); cnt = np.zeros((ph, pw))
    H, W = field.shape
    for (col, row) in centers_local:
        ci, ri = int(round(col)), int(round(row))
        y0, y1, x0, x1 = ri - hwy, ri + hwy + 1, ci - hwx, ci + hwx + 1
        yy0, yy1 = max(0, y0), min(H, y1)
        xx0, xx1 = max(0, x0), min(W, x1)
        if yy1 <= yy0 or xx1 <= xx0:
            continue
        py0, py1 = yy0 - y0, ph - (y1 - yy1)
        px0, px1 = xx0 - x0, pw - (x1 - xx1)
        sub = field[yy0:yy1, xx0:xx1]
        vsub = valid[yy0:yy1, xx0:xx1]
        acc[py0:py1, px0:px1] += np.where(vsub, sub, 0.0)
        cnt[py0:py1, px0:px1] += vsub
    mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return mean, cnt


def _radial_profile(cell, pyu, pxu, rmax_um, nbin=60):
    ny, nx = cell.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot((xx - cx) * pxu, (yy - cy) * pyu)
    edges = np.linspace(0, rmax_um, nbin + 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    prof = np.array([
        np.nanmedian(cell[(r >= edges[i]) & (r < edges[i + 1])]) for i in range(nbin)
    ])
    good = np.isfinite(prof)
    if good.sum() >= 2:
        prof = np.interp(rc, rc[good], prof[good])
    return rc, prof


def _profile_floor_top(prof):
    """Self-contained floor/top/depth from a mean-pin radial profile (fallback only,
    used when the DXF classification could not measure the heights)."""
    if not np.isfinite(prof).any():
        return np.nan, np.nan, np.nan
    it = int(np.argmin(prof))
    floor = float(prof[it])
    inner = prof[:it + 1] if it >= 1 else prof
    peak = float(np.max(inner))
    plateau = inner[inner >= floor + 0.70 * (peak - floor)]
    top = float(np.median(plateau)) if plateau.size else peak
    return floor, top, top - floor


def _diameters_from_profile(rc, prof, floor, depth):
    """Base/mid/top diameters from the folded mean-pin radial profile.

    The crossing LEVELS are anchored to the ``floor`` and ``depth`` measured by the DXF-driven
    pin/floor/debris classification (same z0 units as the folded profile), so the pin-top and
    trench-floor references are the robust, debris-free values rather than the folded cell's
    own (noise-prone) min/max. If the classification did not yield heights, fall back to the
    profile's own shape. Crossings are the first outward descent from the pin centre (immune
    to far debris bumps); a crossing that reaches the cell edge is returned as NaN.
    """
    if not np.isfinite(prof).any():
        return np.nan, np.nan, np.nan
    if not (np.isfinite(floor) and np.isfinite(depth)) or depth <= 0:
        n = len(prof)                                     # fallback: profile shape
        top = float(np.nanmedian(prof[:max(2, n // 12)]))
        floor = float(np.nanmin(prof))
        depth = top - floor
        if depth <= 0:
            return np.nan, np.nan, np.nan
    edge = 0.98 * rc[-1]

    def first_cross(level):
        below = np.nonzero(prof < level)[0]
        if below.size == 0 or below[0] == 0:
            return np.nan                     # never drops (fills cell) / no pin at centre
        k = int(below[0])
        f = (prof[k - 1] - level) / (prof[k - 1] - prof[k] + 1e-9)
        r = rc[k - 1] + f * (rc[k] - rc[k - 1])
        return r if r < edge else np.nan

    d_base = 2.0 * first_cross(floor + 0.15 * depth)
    d_mid = 2.0 * first_cross(floor + 0.50 * depth)
    d_top = 2.0 * first_cross(floor + 0.85 * depth)
    return d_base, d_mid, d_top


def _classify_floor_depth(z0, valid, centers_local, pxu, pyu, d_nom_um, pitch_um):
    """Classify pins vs clean floor vs debris from the KNOWN pin centres, and measure depth
    from the low, flat (debris-free) trench floor.

    Distance-to-nearest-pin is computed in PHYSICAL micrometres from the actual registered
    pin centres (``centers_local`` = (col,row) px), so it is correct on non-square pixels and
    honours any registration rotation (the centres are already rotated by ``dxf_to_px``); it
    also avoids the drift of a modulo lattice extrapolated from a single corner node.
    ``rr`` in the returned dict is in micrometres and ``r_pin_um`` is the physical pin radius.
    """
    H, W = z0.shape
    r_pin_um = float(min(0.5 * d_nom_um, 0.46 * pitch_um))
    yy, xx = np.mgrid[0:H, 0:W]
    px_um = (xx * pxu).ravel()
    py_um = (yy * pyu).ravel()
    from scipy.spatial import cKDTree
    centers_um = centers_local * np.array([pxu, pyu])          # (N,2) col*pxu, row*pyu
    tree = cKDTree(centers_um)
    rr = tree.query(np.column_stack([px_um, py_um]))[0].reshape(H, W)   # um to nearest pin

    pin_core = (rr <= 0.55 * r_pin_um) & valid
    pin_mask = (rr <= r_pin_um) & valid
    floor_region = valid & (rr >= r_pin_um + max(2.0 * max(pxu, pyu), 0.05 * r_pin_um))

    out = dict(pin_mask=pin_mask, floor_region=floor_region, rr=rr, r_pin_um=r_pin_um,
               pin_top=np.nan, clean_floor=np.nan, depth=np.nan,
               flatness=np.nan, debris_frac=np.nan, debris_thresh=np.nan)
    if pin_core.sum() < 20 or floor_region.sum() < 50:
        return out

    pin_top = float(np.median(z0[pin_core]))
    fv = z0[floor_region]
    anchor = float(np.percentile(fv, 15.0))
    clean = fv[np.abs(fv - anchor) <= 3.0]
    if clean.size < 20:
        clean = fv[fv <= np.percentile(fv, 25.0)]
    clean_floor = float(np.median(clean))
    flatness = float(np.std(clean))
    depth = pin_top - clean_floor
    debris_thresh = clean_floor + max(5.0, 6.0 * flatness)
    debris_frac = float(np.mean(fv > debris_thresh))

    out.update(pin_top=pin_top, clean_floor=clean_floor, depth=depth,
               flatness=flatness, debris_frac=debris_frac, debris_thresh=debris_thresh)
    return out


# ============================================================ cropping ====== #
def crop_for_array(scan, placement, array, margin_frac=0.75):
    """Crop the scan to one array's footprint (+ margin). Returns (crop, local_lattice)
    or (None, None) if the array falls outside the scan / registration failed.

    local_lattice = dict(px_px, py_px, phase=(row,col), centers_local=(N,2 col,row))
    """
    if not np.isfinite(placement.origin_col) or not np.isfinite(placement.origin_row):
        return None, None
    xppx, yppx = scan.x_um_per_px, scan.y_um_per_px
    cols, rows = placement.dxf_to_px(array.centers_um[:, 0], array.centers_um[:, 1])
    px_px = array.pitch_x_um / xppx
    py_px = array.pitch_y_um / yppx
    mcol = margin_frac * px_px
    mrow = margin_frac * py_px

    c0 = int(np.floor(cols.min() - mcol)); c1 = int(np.ceil(cols.max() + mcol))
    r0 = int(np.floor(rows.min() - mrow)); r1 = int(np.ceil(rows.max() + mrow))
    H, W = scan.height_raw.shape
    c0c, c1c = max(0, c0), min(W, c1)
    r0c, r1c = max(0, r0), min(H, r1)
    if c1c - c0c < 8 or r1c - r0c < 8:
        return None, None

    hr = scan.height_raw[r0c:r1c, c0c:c1c].copy()
    inten = (scan.intensity[r0c:r1c, c0c:c1c].copy()
             if getattr(scan, "intensity", None) is not None else None)
    crop = ScanCrop(height_raw=hr, intensity=inten,
                    x_um_per_px=xppx, y_um_per_px=yppx,
                    z_um_per_digit=scan.z_um_per_digit)

    cl = np.column_stack([cols - c0c, rows - r0c])
    phase = (float(cl[0, 1]), float(cl[0, 0]))     # (row, col) of the first pin
    lattice = dict(px_px=px_px, py_px=py_px, phase=phase, centers_local=cl,
                   coverage_frac=((c1c - c0c) * (r1c - r0c)) /
                                 max(1, (c1 - c0) * (r1 - r0)))
    return crop, lattice


# ============================================================ extraction ==== #
def extract_array(scan, placement, array, sample, *,
                  make_qc=True, qc_path=None) -> PinFinResult:
    """Measure one array. Uses the known (registered) lattice; autocorrelation is QC/fallback."""
    crop, lattice = crop_for_array(scan, placement, array)
    base = dict(
        filename=sample.filename, vk4_stem=sample.vk4_stem,
        cell_id=sample.cell_id, array_id=sample.array_id,
        band=sample.band, col=sample.col, passes=sample.passes, speed=sample.speed,
        cx_um=sample.cx_um, cy_um=sample.cy_um,
        nominal_diameter_um=sample.nominal_diameter_um,
        target_diameter_um=sample.target_diameter_um,
        nominal_pitch_um=sample.nominal_pitch_um,
        reg_score=placement.score, reg_method=placement.method,
    )
    if crop is None:
        return PinFinResult(flags="off-scan (registration/crop failed)", **base)

    pxu, pyu = crop.x_um_per_px, crop.y_um_per_px
    z = crop.height_um
    valid = crop.height_raw != 0
    d_nom = sample.nominal_diameter_um
    flags = []

    z0 = _level_floor(z, valid)

    # --- known lattice (primary) vs measured autocorrelation (QC) ---
    known_px, known_py = lattice["px_px"], lattice["py_px"]
    meas_px, meas_py, periodicity = _measure_pitch_px(
        z0, valid, sample.nominal_pitch_um / pxu)
    meas_pitch_um = 0.5 * (meas_px * pxu + meas_py * pyu)
    pitch_x_um, pitch_y_um = known_px * pxu, known_py * pyu
    pitch_um = 0.5 * (pitch_x_um + pitch_y_um)
    if abs(meas_pitch_um - pitch_um) > 0.25 * pitch_um and periodicity > 0.15:
        flags.append(f"pitch meas {meas_pitch_um:.0f} vs design {pitch_um:.0f}")

    # --- mean pin by stacking the patches centred on each KNOWN pin (from the DXF lattice).
    # Avoids the wrap/edge artifacts of a modulo-fold on a tightly-cropped array and keeps
    # the radial profile exactly centred — an off-centre or seamed mean pin smears the edge
    # and corrupts the near-top diameter. ---
    cell_c, cnt = _stack_pins(z0, valid, lattice["centers_local"], known_px, known_py)
    coverage = float((cnt > 0).mean())

    rmax = 0.5 * pitch_um * 1.05
    rc, prof = _radial_profile(cell_c, pyu, pxu, rmax)

    n_cells = sample.nx * sample.ny * lattice.get("coverage_frac", 1.0)

    # --- PRIMARY: classify pin / clean-floor / debris from the KNOWN DXF pin centres, and
    # take the floor and pin-top HEIGHTS from those classified pixel populations (robust to
    # debris; the folded profile is not used to define the heights). ---
    cls = _classify_floor_depth(z0, valid, lattice["centers_local"], pxu, pyu,
                                d_nom, pitch_um)
    depth, floor, top = cls["depth"], cls["clean_floor"], cls["pin_top"]
    floor_flatness, debris_fraction = cls["flatness"], cls["debris_frac"]
    if not np.isfinite(depth):                      # fallback: derive heights from the profile
        floor, top, depth = _profile_floor_top(prof)

    # diameters = radial-profile edge, crossings anchored to the classified floor+depth
    d_base, d_mid, d_top = _diameters_from_profile(rc, prof, floor, depth)

    pin_sat_frac = float("nan")
    if crop.intensity is not None and np.isfinite(cls.get("r_pin_um", float("nan"))):
        core = (cls["rr"] <= 0.55 * cls["r_pin_um"]) & valid
        if int(core.sum()) > 10:
            pin_sat_frac = float(np.mean(
                crop.intensity[core].astype(np.float64) >= 0.98 * 65535.0))

    # --- reliability flags (same policy as v1) ---
    if not np.isfinite(depth) or depth < 1.5:
        if (np.isfinite(depth) and depth < -1.0 and np.isfinite(pin_sat_frac)
                and pin_sat_frac > 0.3):
            flags.append("no relief - pin below floor (specular, height suspect)")
        else:
            flags.append("no relief (<1.5um)")
    elif depth < 3.0:
        flags.append(f"shallow ({depth:.1f}um)")
    if np.isfinite(periodicity) and periodicity < 0.12:
        flags.append(f"weak lattice ({periodicity:.2f})")
    if np.isfinite(d_mid) and d_mid > 1.4 * d_nom:
        flags.append("wide-D (debris?)")

    down = max(1, int(round(min(known_px, known_py) / 4)))
    thumb = np.where(valid, z0, np.nan)[::down, ::down]

    res = PinFinResult(
        pitch_um=pitch_um, pitch_x_um=pitch_x_um, pitch_y_um=pitch_y_um,
        diameter_um=float(d_mid), base_diameter_um=float(d_base),
        top_diameter_um=float(d_top), depth_um=float(depth),
        floor_um=float(floor), top_um=float(top),
        lattice_strength=float(periodicity), coverage=float(coverage),
        n_cells=float(n_cells), flags=";".join(flags),
        floor_flatness_um=float(floor_flatness), debris_fraction=float(debris_fraction),
        pin_sat_frac=pin_sat_frac,
        grid_phase=(lattice["phase"][0], lattice["phase"][1]),
        grid_px_px=float(known_px), grid_py_px=float(known_py),
        thumb=thumb, thumb_down=down, **base,
    )
    if make_qc:
        _qc(crop, z0, valid, cell_c, rc, prof, lattice, known_px, known_py,
            res, qc_path, cls)
    return res


# ================================================================= QC plot == #
def _qc(crop, z0, valid, cell_c, rc, prof, lattice, px_px, py_px, res, qc_path, cls):
    pxu, pyu = crop.x_um_per_px, crop.y_um_per_px
    H, W = z0.shape
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    cf = res.floor_um if np.isfinite(res.floor_um) else 0.0
    disp = np.where(valid, z0 - cf, np.nan)
    finite = np.isfinite(disp)
    if finite.any():
        vmin, vmax = np.nanpercentile(disp, 2), np.nanpercentile(disp, 98)
    else:
        vmin, vmax = 0.0, 1.0
    good_pin = (np.isfinite(res.depth_um) and res.depth_um > 0.5
                and np.isfinite(res.top_diameter_um)
                and res.top_diameter_um < 1.03 * res.pitch_um)

    # (0,0) height referenced to clean floor + KNOWN pin centres
    im0 = ax[0, 0].imshow(disp, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im0, ax=ax[0, 0], shrink=0.8, label="height above clean floor (µm)")
    cl = lattice["centers_local"]
    ax[0, 0].plot(cl[:, 0], cl[:, 1], "r+", ms=9, mew=1.2)
    ax[0, 0].set_title(f"{res.filename}\nknown pins (pitch {res.pitch_um:.1f} µm, "
                       f"reg {res.reg_method} {res.reg_score:.2f})")
    ax[0, 0].set_xlabel("px"); ax[0, 0].set_ylabel("px")

    # (0,1) mean pin
    ny, nx = cell_c.shape
    cx, cy = nx * pxu / 2, ny * pyu / 2
    ax[0, 1].imshow(cell_c, origin="lower", cmap="viridis",
                    extent=[0, nx * pxu, 0, ny * pyu])
    ax[0, 1].add_patch(Circle((cx, cy), res.nominal_diameter_um / 2, fill=False,
                              color="w", lw=1.2, ls=":"))
    if good_pin:
        ax[0, 1].add_patch(Circle((cx, cy), res.base_diameter_um / 2, fill=False,
                                  color="orange", lw=1.5, ls="--"))
        ax[0, 1].add_patch(Circle((cx, cy), res.top_diameter_um / 2, fill=False,
                                  color="r", lw=1.5))
        ax[0, 1].set_title(f"mean pin (~{res.n_cells:.0f})  base={res.base_diameter_um:.0f}"
                           f" / top={res.top_diameter_um:.0f} µm "
                           f"(drawn {res.nominal_diameter_um:g}, white)")
    else:
        ax[0, 1].set_title(f"mean pin (~{res.n_cells:.0f})  drawn Ø={res.nominal_diameter_um:g}"
                           f" (white)\ndiameter unreliable")
    ax[0, 1].set_xlabel("µm"); ax[0, 1].set_ylabel("µm")

    # (1,0) classification
    ax[1, 0].imshow(disp, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ov = np.zeros((H, W, 4))
    pin_mask, floor_region = cls["pin_mask"], cls["floor_region"]
    dth = cls["debris_thresh"]
    ov[pin_mask] = (0.86, 0.08, 0.08, 0.40)
    if np.isfinite(dth):
        ov[floor_region & (z0 <= dth)] = (0.11, 0.35, 0.93, 0.38)
        ov[floor_region & (z0 > dth)] = (1.0, 0.60, 0.0, 0.42)
    ax[1, 0].imshow(ov, origin="lower")
    ax[1, 0].legend(handles=[Patch(color="#DC1414", label="pins"),
                             Patch(color="#1C59ED", label="clean floor"),
                             Patch(color="#FF9900", label="debris")],
                    fontsize=8, loc="upper right")
    ax[1, 0].set_title(f"classification  depth={res.depth_um:.1f} µm · "
                       f"floor flat={res.floor_flatness_um:.2f} µm · "
                       f"debris {100*res.debris_fraction:.0f}%")
    ax[1, 0].set_xlabel("px"); ax[1, 0].set_ylabel("px")

    # (1,1) radial profile
    ax[1, 1].plot(rc, prof, "-o", ms=3, color="0.3")
    if good_pin:
        for r_, c_, lab in [(res.base_diameter_um / 2, "orange", "base"),
                            (res.diameter_um / 2, "purple", "mid"),
                            (res.top_diameter_um / 2, "red", "top")]:
            ax[1, 1].axvline(r_, color=c_, ls="--", lw=1, label=f"{lab} Ø={2*r_:.0f}")
    ax[1, 1].axvline(res.nominal_diameter_um / 2, color="g", ls=":",
                     label=f"drawn r={res.nominal_diameter_um/2:g}")
    ax[1, 1].set_xlabel("radius (µm)"); ax[1, 1].set_ylabel("mean-cell height (µm)")
    ax[1, 1].legend(fontsize=7)
    sat = f" · pin-sat {100*res.pin_sat_frac:.0f}%" if np.isfinite(res.pin_sat_frac) else ""
    ax[1, 1].set_title(f"radial profile  meas-latt={res.lattice_strength:.2f}{sat}"
                       + (f"\nFLAGS: {res.flags}" if res.flags else ""))

    plt.tight_layout()
    if qc_path is None:
        qc_path = Path(res.filename).with_suffix(".qc.png")
    Path(qc_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(qc_path, dpi=150)
    plt.close(fig)


# ============================================================ full-scan map = #
def save_height_map(scan, path, *, title=None, placements=None, design=None):
    """Colour-gradient height map of the RAW scan (µm axes). If placements+design are
    given, overlay each cell's marker origin and pin lattice so the registration can be
    eyeballed."""
    z = scan.height_um
    valid = scan.height_raw != 0
    disp = np.where(valid, z, np.nan)
    x1 = scan.width * scan.x_um_per_px
    y1 = scan.height * scan.y_um_per_px
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(disp, origin="lower", cmap="viridis", extent=[0, x1, 0, y1],
                   vmin=np.nanmin(disp) if valid.any() else 0,
                   vmax=np.nanmax(disp) if valid.any() else 1)
    if placements and design:
        for pl, cell in zip(placements, design.cells):
            if not np.isfinite(pl.origin_col):
                continue
            ax.plot(pl.origin_col * scan.x_um_per_px, pl.origin_row * scan.y_um_per_px,
                    "ws", ms=9, mfc="none", mew=1.6)
            for a in cell.arrays:
                cols, rows = pl.dxf_to_px(a.centers_um[:, 0], a.centers_um[:, 1])
                ax.plot(cols * scan.x_um_per_px, rows * scan.y_um_per_px,
                        "r+", ms=3, mew=0.5)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(title or Path(path).stem)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, shrink=0.85, label="raw height (µm)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
