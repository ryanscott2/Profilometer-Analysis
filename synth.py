"""
Synthetic VK4-like scans generated from the DXF geometry.

The real unit-cell VK4 scans do not exist yet, so this module fabricates a scan whose pins,
alignment marker, tilt, noise and (optional) debris are placed from the *known* DXF
coordinates at a chosen pixel scale and origin. It is used by ``selftest.py`` to prove the
registration recovers the known origin and the extraction recovers the known diameter /
pitch / depth, and it doubles as a demo data source (``synth.py`` writes a preview PNG).

A :class:`SynthScan` quacks like ``vk4.VK4`` (``height_raw``, ``intensity``,
``x_um_per_px``, ``y_um_per_px``, ``z_um_per_digit``, ``height_um``, ``width``, ``height``),
so it flows straight through ``register.py`` and ``extract.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynthScan:
    height_raw: np.ndarray
    intensity: np.ndarray
    x_um_per_px: float
    y_um_per_px: float
    z_um_per_digit: float

    @property
    def height_um(self):
        return self.height_raw.astype(np.float64) * self.z_um_per_digit

    @property
    def width(self):
        return self.height_raw.shape[1]

    @property
    def height(self):
        return self.height_raw.shape[0]


def _stamp_disk(arr, cx, cy, rx, value, mode="set", ry=None):
    """Stamp an ellipse of pixel half-axes (rx, ry) -- a physical circle on non-square pixels
    when rx=radius_um/x_um_per_px and ry=radius_um/y_um_per_px. ry defaults to rx (circle)."""
    ry = rx if ry is None else ry
    H, W = arr.shape
    rix, riy = int(np.ceil(rx)), int(np.ceil(ry))
    x0, x1 = max(0, int(cx) - rix), min(W, int(cx) + rix + 1)
    y0, y1 = max(0, int(cy) - riy), min(H, int(cy) + riy + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    sub = arr[y0:y1, x0:x1]
    if mode == "set":
        sub[m] = value
    else:
        sub[m] += value
    arr[y0:y1, x0:x1] = sub


def synth_scan(template, *, x_um_per_px=2.0, y_um_per_px=2.0,
               origin_px=(120.0, 90.0), n_cells=1, cell_gap_um=300.0,
               depth_um=30.0, floor_um=60.0, taper_frac=0.0,
               noise_um=0.4, tilt=(0.0015, -0.0008), debris_frac=0.0,
               marker=True, y_up=1, x_right=1, rotation_deg=0.0, seed=0):
    """Build a synthetic scan of ``n_cells`` copies of ``template``.

    ``rotation_deg`` rotates each cell's pins about its marker origin (to exercise the
    rotation-aware measurement path; the marker itself is left axis-aligned).
    Returns (scan, truth) where truth = dict with per-cell origins (col,row) and the
    scalar ground-truth depth/floor used, so the caller can score recovery.
    """
    rng = np.random.default_rng(seed)
    xppx, yppx = x_um_per_px, y_um_per_px
    th = np.deg2rad(rotation_deg)
    cth, sth = np.cos(th), np.sin(th)

    # cell pin span (um) from marker origin
    allc = template.all_centers_um()
    span_x_um = allc[:, 0].max()
    span_y_um = allc[:, 1].max()
    cell_w_px = span_x_um / xppx
    cell_h_px = span_y_um / yppx
    tile_dx_px = (span_x_um + cell_gap_um) / xppx

    oc0, orow0 = origin_px
    pad = 60
    W = int(oc0 + (n_cells - 1) * tile_dx_px + cell_w_px + pad + (cell_w_px if x_right < 0 else 0))
    H = int(orow0 + cell_h_px + pad + (abs(span_y_um) / yppx if y_up < 0 else 0))
    W = max(W, 64); H = max(H, 64)

    # base surface: floor + tilt plane + noise
    yy, xx = np.mgrid[0:H, 0:W]
    z = floor_um + tilt[0] * xx * xppx / 1000.0 + tilt[1] * yy * yppx / 1000.0
    z = z + rng.normal(0, noise_um, size=z.shape)
    inten = np.full((H, W), 8000.0)                # dark floor
    inten += rng.normal(0, 300, size=inten.shape)

    truth_origins = []
    for k in range(n_cells):
        oc = oc0 + k * tile_dx_px
        orow = orow0
        truth_origins.append((float(oc), float(orow)))

        # alignment marker: the ACTUAL fiducial shape at the cell origin (CAD (0,0) -> (oc,orow)),
        # under the same x_right/y_up reflection as the pins. The L polygon exercises the PRIMARY
        # production detector (register_sample's marker branch); the square ring is the legacy
        # fallback for square-marker templates (which register_scan resolves).
        _mshape = getattr(template, "marker_shape", "")
        _mpoly = getattr(template, "marker_polygon_um", None)
        if marker and _mshape == "L" and _mpoly is not None and len(_mpoly) >= 3:
            from matplotlib.path import Path as _MplPath          # fill the true L polygon (proud)
            poly = np.asarray(_mpoly, float)
            _pxr = cth * poly[:, 0] - sth * poly[:, 1]           # rotate the marker about the origin,
            _pyr = sth * poly[:, 0] + cth * poly[:, 1]           # consistent with the pins (real stage rot)
            vc = oc + x_right * _pxr / xppx
            vr = orow + y_up * _pyr / yppx
            gc0, gc1 = int(np.floor(vc.min())), int(np.ceil(vc.max()))
            gr0, gr1 = int(np.floor(vr.min())), int(np.ceil(vr.max()))
            gx, gy = np.meshgrid(np.arange(max(0, gc0), min(W, gc1 + 1)),
                                 np.arange(max(0, gr0), min(H, gr1 + 1)))
            if gx.size:
                inside = _MplPath(np.column_stack([vc, vr])).contains_points(
                    np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
                z[gy[inside], gx[inside]] = floor_um + depth_um
                inten[gy[inside], gx[inside]] = 55000.0
        elif marker and np.isfinite(template.marker_size_um):    # legacy raised square ring
            ms_px_x = template.marker_size_um / xppx
            ms_px_y = template.marker_size_um / yppx
            thick = max(3, int(round(16.0 / xppx)))              # ~16 um ring thickness
            c0 = int(round(oc)); r0 = int(round(orow))
            c1 = int(round(oc + x_right * ms_px_x)); r1 = int(round(orow + y_up * ms_px_y))
            rr0, rr1 = min(r0, r1), max(r0, r1)
            cc0, cc1 = min(c0, c1), max(c0, c1)
            for (a0, a1, b0, b1) in [(rr0, rr0 + thick, cc0, cc1),        # bottom edge
                                     (rr1 - thick, rr1, cc0, cc1),        # top edge
                                     (rr0, rr1, cc0, cc0 + thick),        # left edge
                                     (rr0, rr1, cc1 - thick, cc1)]:       # right edge
                a0c, a1c = max(0, a0), min(H, a1)
                b0c, b1c = max(0, b0), min(W, b1)
                z[a0c:a1c, b0c:b1c] = floor_um + depth_um
                inten[a0c:a1c, b0c:b1c] = 55000.0

        # pins
        for a in template.arrays:
            r_top_um = 0.5 * a.diameter_um * (1.0 - taper_frac)
            for (x_um, y_um) in a.centers_um:
                xr = cth * x_um - sth * y_um            # rotate about the marker origin
                yr = sth * x_um + cth * y_um
                cx = oc + x_right * (xr / xppx)
                cy = orow + y_up * (yr / yppx)
                rx, ry = 0.5 * a.diameter_um / xppx, 0.5 * a.diameter_um / yppx
                _stamp_disk(z, cx, cy, rx, floor_um + depth_um, "set", ry=ry)
                _stamp_disk(inten, cx, cy, rx, 60000.0, "set", ry=ry)
                if taper_frac > 0:                       # narrower bright plateau on top
                    _stamp_disk(inten, cx, cy, r_top_um / xppx, 63000.0, "set",
                                ry=r_top_um / yppx)

        # optional debris: random bumps on the floor between pins
        if debris_frac > 0:
            n_deb = int(debris_frac * (cell_w_px * cell_h_px) / 50)
            for _ in range(n_deb):
                dc = oc + x_right * rng.uniform(0, cell_w_px)
                dr = orow + y_up * rng.uniform(0, cell_h_px)
                _stamp_disk(z, dc, dr, rng.uniform(2, 5),
                            floor_um + rng.uniform(3, depth_um * 0.9), "set")
                _stamp_disk(inten, dc, dr, rng.uniform(2, 5), 15000.0, "set")

    z_um_per_digit = 0.001
    raw = np.clip(np.round(z / z_um_per_digit), 1, None).astype(np.uint32)
    inten = np.clip(inten, 0, 65535).astype(np.uint16)
    scan = SynthScan(raw, inten, xppx, yppx, z_um_per_digit)
    truth = dict(origins=truth_origins, depth_um=depth_um, floor_um=floor_um,
                 x_um_per_px=xppx, y_um_per_px=yppx, y_up=y_up, x_right=x_right,
                 rotation_deg=rotation_deg)
    return scan, truth


if __name__ == "__main__":       # pragma: no cover
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dxf_geometry import read_design

    here = Path(__file__).parent
    design = read_design(next((here / "DXF").glob("*.dxf")))
    scan, truth = synth_scan(design.cells[0])
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(scan.height_um, origin="lower", cmap="viridis"); ax[0].set_title("height")
    ax[1].imshow(scan.intensity, origin="lower", cmap="gray"); ax[1].set_title("intensity")
    out = here / "Results" / "synth_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120); print("wrote", out, "truth origin", truth["origins"])
