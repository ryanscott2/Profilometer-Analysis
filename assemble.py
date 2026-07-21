"""
Assemble a grid of VK4 tiles (filenames ``..._Y{row}_X{col}.vk4``) into one continuous scan.

A full-sample profilometry run is captured as an overlapping raster of field-of-view tiles.
The ``Y{n}_X{m}`` indices give the raster position; the physical **step** between tiles
(FOV minus overlap) is measured here by cross-correlating adjacent tiles, then every tile is
stamped into one big height + intensity array. The result quacks like ``vk4.VK4`` so it flows
straight through ``register.py`` / ``extract.py`` in a single global coordinate system.

Space is not a concern (the caller opts into holding the whole sample in memory); this favours
a simple, fast single-array assembly over tiled/lazy access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vk4 import read_vk4

_TILE_RE = re.compile(r"_Y(\d+)_X(\d+)\.vk4$", re.IGNORECASE)


@dataclass
class AssembledScan:
    """A stitched multi-tile scan, presenting the ``vk4.VK4`` interface."""

    height_raw: np.ndarray
    intensity: np.ndarray | None
    x_um_per_px: float
    y_um_per_px: float
    z_um_per_digit: float
    step_col: int
    step_row: int
    grid_rows: int              # number of tile rows (Y)
    grid_cols: int              # number of tile cols (X)
    tile_shape: tuple           # (h, w) of one tile
    y_indices: tuple = ()       # sorted Y indices present
    x_indices: tuple = ()       # sorted X indices present

    @property
    def height_um(self) -> np.ndarray:
        return self.height_raw.astype(np.float64) * self.z_um_per_digit

    @property
    def width(self) -> int:
        return self.height_raw.shape[1]

    @property
    def height(self) -> int:
        return self.height_raw.shape[0]


# --------------------------------------------------------- step estimation #
def _feat(vk):
    """Normalised feature (intensity if present, else height) for correlation."""
    v = vk.height_raw != 0
    f = vk.intensity.astype(np.float64) if vk.intensity is not None else vk.height_um
    f = np.where(v, f, np.nan)
    lo, hi = np.nanpercentile(f, 2), np.nanpercentile(f, 98)
    return np.nan_to_num((f - lo) / (hi - lo + 1e-9), nan=0.0)


def _best_shift(A, B, drange, axis, perp=30):
    """B is to the right of (axis='x') / below (axis='y') A. Return (step, perp, ncc)."""
    H, W = A.shape
    best = (drange[0], 0, -1.0)
    for d in drange:
        for p in range(-perp, perp + 1):
            if axis == "x":
                a, b = A[:, d:], B[:, :W - d]
                if p >= 0:
                    a, b = a[p:], b[:b.shape[0] - p]
                else:
                    a, b = a[:a.shape[0] + p], b[-p:]
            else:
                a, b = A[d:, :], B[:H - d, :]
                if p >= 0:
                    a, b = a[:, p:], b[:, :b.shape[1] - p]
                else:
                    a, b = a[:, :a.shape[1] + p], b[:, -p:]
            if a.size < 500:
                continue
            a = a - a.mean(); b = b - b.mean()
            denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9
            ncc = float((a * b).sum() / denom)
            if ncc > best[2]:
                best = (d, p, ncc)
    return best


def _estimate_step(tiles, ys, xs, tile_hw, ds=4, max_pairs=6):
    """Median tile step (step_col, step_row) in px, from several adjacent pairs."""
    th, tw = tile_hw
    dxs, dys = [], []
    # horizontal pairs
    hpairs = [(y, x) for y in ys for x in xs[:-1] if (y, x) in tiles and (y, x + 1) in tiles]
    for (y, x) in hpairs[:: max(1, len(hpairs) // max_pairs)][:max_pairs]:
        A = _feat(tiles[(y, x)])[::ds, ::ds]; B = _feat(tiles[(y, x + 1)])[::ds, ::ds]
        W = A.shape[1]
        d, _p, ncc = _best_shift(A, B, range(int(0.5 * W), int(0.98 * W)), "x")
        if ncc > 0.3:
            dxs.append(d * ds)
    vpairs = [(y, x) for y in ys[:-1] for x in xs if (y, x) in tiles and (y + 1, x) in tiles]
    for (y, x) in vpairs[:: max(1, len(vpairs) // max_pairs)][:max_pairs]:
        A = _feat(tiles[(y, x)])[::ds, ::ds]; B = _feat(tiles[(y + 1, x)])[::ds, ::ds]
        H = A.shape[0]
        d, _p, ncc = _best_shift(A, B, range(int(0.5 * H), int(0.98 * H)), "y")
        if ncc > 0.3:
            dys.append(d * ds)
    step_col = int(np.median(dxs)) if dxs else tw
    step_row = int(np.median(dys)) if dys else th
    return step_col, step_row


# --------------------------------------------------------------- assembly #
def assemble_tiles(vk4_dir, step_col=None, step_row=None, verbose=True) -> AssembledScan:
    """Stitch every ``_Y*_X*.vk4`` tile in ``vk4_dir`` into one :class:`AssembledScan`.

    Tile (Y,X) is placed with its top-left at ((X-Xmin)*step_col, (Y-Ymin)*step_row); in
    overlaps, valid pixels of later-placed tiles win. Step defaults to a measured estimate.

    Heights are stamped RAW -- no tile-to-tile Z normalisation. A sample scanned with consistent
    microscope settings shares one Z reference across its tiles, so reconciling per-tile offsets
    (which we used to do) only injected noise; the real, unmodified heights are kept instead.
    """
    vk4_dir = Path(vk4_dir)
    tiles = {}
    for f in sorted(vk4_dir.glob("*.vk4")):
        m = _TILE_RE.search(f.name)
        if m:
            tiles[(int(m.group(1)), int(m.group(2)))] = read_vk4(f)
    if not tiles:
        raise SystemExit(f"No '_Y*_X*.vk4' tiles found in {vk4_dir}")

    ys = sorted({k[0] for k in tiles}); xs = sorted({k[1] for k in tiles})
    vk0 = tiles[(ys[0], xs[0])]
    xppx, yppx, zpd = vk0.x_um_per_px, vk0.y_um_per_px, vk0.z_um_per_digit
    th, tw = vk0.height, vk0.width
    has_int0 = vk0.intensity is not None
    for (y, x), vk in tiles.items():                     # fail closed on any tile inconsistency
        if (abs(vk.x_um_per_px - xppx) > 1e-6 or abs(vk.y_um_per_px - yppx) > 1e-6
                or abs(vk.z_um_per_digit - zpd) > 1e-9):
            raise ValueError(f"tile Y{y}_X{x} has inconsistent x/y/z calibration; cannot assemble")
        if (vk.height, vk.width) != (th, tw):
            raise ValueError(f"tile Y{y}_X{x} shape {vk.width}x{vk.height} != {tw}x{th}; "
                             f"cannot assemble")
        if vk.height_raw.dtype != vk0.height_raw.dtype:
            raise ValueError(f"tile Y{y}_X{x} height dtype {vk.height_raw.dtype} != "
                             f"{vk0.height_raw.dtype}; cannot assemble")
        if (vk.intensity is not None) != has_int0:
            raise ValueError(f"tile Y{y}_X{x} intensity presence differs from the first tile; "
                             f"cannot assemble")

    if step_col is None or step_row is None:
        sc, sr = _estimate_step(tiles, ys, xs, (th, tw))
        step_col = step_col or sc
        step_row = step_row or sr
    if verbose:
        print(f"assembling {len(tiles)} tiles ({len(ys)}Y x {len(xs)}X), "
              f"tile {tw}x{th}px, step ({step_col},{step_row})px  "
              f"overlap ({100*(1-step_col/tw):.0f}%,{100*(1-step_row/th):.0f}%)")

    W = (max(xs) - min(xs)) * step_col + tw
    H = (max(ys) - min(ys)) * step_row + th
    height = np.zeros((H, W), np.uint32)
    has_int = vk0.intensity is not None
    inten = np.zeros((H, W), np.uint16) if has_int else None
    for (y, x), vk in tiles.items():           # stamp each tile's raw height (no Z normalisation)
        r0 = (y - min(ys)) * step_row
        c0 = (x - min(xs)) * step_col
        v = vk.height_raw != 0
        hs = height[r0:r0 + th, c0:c0 + tw]; hs[v] = vk.height_raw[v]
        height[r0:r0 + th, c0:c0 + tw] = hs
        if has_int and vk.intensity is not None:
            isl = inten[r0:r0 + th, c0:c0 + tw]; isl[v] = vk.intensity[v]
            inten[r0:r0 + th, c0:c0 + tw] = isl

    if verbose:
        print(f"  -> {W}x{H}px = {W*xppx/1000:.2f} x {H*yppx/1000:.2f} mm")
    return AssembledScan(height_raw=height, intensity=inten,
                         x_um_per_px=xppx, y_um_per_px=yppx, z_um_per_digit=zpd,
                         step_col=step_col, step_row=step_row,
                         grid_rows=len(ys), grid_cols=len(xs), tile_shape=(th, tw),
                         y_indices=tuple(ys), x_indices=tuple(xs))


if __name__ == "__main__":       # pragma: no cover
    import sys
    s = assemble_tiles(sys.argv[1] if len(sys.argv) > 1 else
                       Path(__file__).parent / "VK4")
    print(s.width, s.height, s.step_col, s.step_row)
