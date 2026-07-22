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
    step_col_row: int = 0       # row drift when advancing one tile column
    step_row_col: int = 0       # column drift when advancing one tile row
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


def _dominant_vector(vectors, tol):
    """Consensus 2-D tile-origin vector, or ``None`` when adjacent pairs do not agree.

    Both the primary step and perpendicular stage drift matter.  A majority cluster is required so
    a periodic texture cannot win by an arbitrary plurality, and the component-wise median keeps a
    single noisy pair from skewing the mosaic.
    """
    if not vectors:
        return None
    best, best_resid = [], float("inf")
    for s in vectors:
        c = [t for t in vectors if np.hypot(t[0] - s[0], t[1] - s[1]) <= tol]
        centre = np.median(np.asarray(c, float), axis=0)
        resid = float(np.mean([np.hypot(t[0] - centre[0], t[1] - centre[1]) for t in c]))
        if len(c) > len(best) or (len(c) == len(best) and resid < best_resid):
            best = c
            best_resid = resid
    if len(best) < (len(vectors) // 2 + 1 if len(vectors) > 1 else 1):
        return None
    centre = np.median(np.asarray(best, float), axis=0)
    return int(round(centre[0])), int(round(centre[1]))


def _estimate_step(tiles, ys, xs, tile_hw, ds=4):
    """Global 2-D column/row step vectors from the consensus of every adjacent pair.

    Using all pairs + a dominant-cluster vote (not a few-pair plain median) is what makes this
    alias-safe on periodic-pin samples: the earlier code sampled ~6 pairs and took the median, so
    when a minority of pairs aliased by one pin pitch (large pins on a tight pitch, e.g. D300) the
    median could land on the aliased step and compress the whole mosaic (cells then overlap)."""
    th, tw = tile_hw
    tol = max(3 * ds, 8)                                 # group near-identical shifts; keeps pin-pitch aliases apart
    col_vectors, row_vectors = [], []
    xpairs = [(y, x) for y in ys for x in xs[:-1] if (y, x) in tiles and (y, x + 1) in tiles]
    ypairs = [(y, x) for y in ys[:-1] for x in xs if (y, x) in tiles and (y + 1, x) in tiles]
    edge = 0                                             # pairs whose best shift pinned at the 50% edge
    for (y, x) in xpairs:
        A = _feat(tiles[(y, x)])[::ds, ::ds]; B = _feat(tiles[(y, x + 1)])[::ds, ::ds]
        W = A.shape[1]; lo = int(0.5 * W)
        d, p, ncc = _best_shift(A, B, range(lo, int(0.98 * W)), "x")
        if ncc > 0.3:
            if d > lo:
                col_vectors.append((d * ds, p * ds))
            else:                                        # peak pinned at the 50%-overlap search limit
                edge += 1
    for (y, x) in ypairs:
        A = _feat(tiles[(y, x)])[::ds, ::ds]; B = _feat(tiles[(y + 1, x)])[::ds, ::ds]
        H = A.shape[0]; lo = int(0.5 * H)
        d, p, ncc = _best_shift(A, B, range(lo, int(0.98 * H)), "y")
        if ncc > 0.3:
            if d > lo:
                row_vectors.append((p * ds, d * ds))
            else:
                edge += 1
    if edge:                                             # true overlap likely exceeds the 50% window
        print(f"WARNING: {edge} tile pair(s) correlated best at the 50%-overlap search limit -> the "
              f"true overlap may exceed 50% (the step search covers only ~2-50% overlap); those "
              f"pairs are ignored. If assembly then fails, pass explicit step_col/step_row.")
    # Return None (NOT the tile size) when adjacent pairs exist on an axis but NONE yielded a usable
    # shift: the caller must fail closed rather than silently assemble at 0% overlap (which mis-places
    # every tile). Fall back to the tile size only when there are no adjacent pairs at all (single
    # row/column -- that axis has no overlap to measure).
    col_step = (_dominant_vector(col_vectors, tol) if xpairs else (tw, 0))
    row_step = (_dominant_vector(row_vectors, tol) if ypairs else (0, th))
    return col_step, row_step


# --------------------------------------------------------------- assembly #
def assemble_tiles(vk4_dir, step_col=None, step_row=None, verbose=True) -> AssembledScan:
    """Stitch every ``_Y*_X*.vk4`` tile in ``vk4_dir`` into one :class:`AssembledScan`.

    Tile origins follow two measured 2-D vectors (advance one tile column / one tile row), preserving
    consistent cross-axis stage drift instead of forcing the mosaic onto an axis-aligned grid. In
    overlaps, valid pixels of later-placed tiles win. Primary steps may be overridden explicitly.

    Heights are stamped RAW -- no tile-to-tile Z normalisation. A sample scanned with consistent
    microscope settings shares one Z reference across its tiles, so reconciling per-tile offsets
    (which we used to do) only injected noise; the real, unmodified heights are kept instead.
    """
    vk4_dir = Path(vk4_dir)
    tiles = {}
    sources = {}
    for f in sorted(vk4_dir.glob("*.vk4")):
        m = _TILE_RE.search(f.name)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            if key in sources:
                raise ValueError(
                    f"duplicate tile coordinates Y{key[0]} X{key[1]}: "
                    f"'{sources[key].name}' and '{f.name}'")
            sources[key] = f
            tiles[key] = read_vk4(f)
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
        if has_int0 and (vk.intensity.shape != (th, tw)
                         or vk.intensity.dtype != vk0.intensity.dtype):
            raise ValueError(f"tile Y{y}_X{x} intensity {vk.intensity.shape}/{vk.intensity.dtype} "
                             f"!= expected {(th, tw)}/{vk0.intensity.dtype} (must match the height "
                             f"raster); cannot assemble")

    if step_col is None or step_row is None:
        col_est, row_est = _estimate_step(tiles, ys, xs, (th, tw))
        if step_col is None:
            if col_est is None:
                raise ValueError(
                    "tile-step estimation failed on the column (x) axis: adjacent tiles exist but "
                    "none exceeded the correlation gate (NCC>0.3). Refusing to assume 0% overlap "
                    "(= full tile width), which would misplace every tile and silently corrupt all "
                    "downstream coordinates. Pass an explicit step_col, or check tile overlap/order.")
            step_col = col_est[0]
        if step_row is None:
            if row_est is None:
                raise ValueError(
                    "tile-step estimation failed on the row (y) axis: adjacent tiles exist but none "
                    "exceeded the correlation gate (NCC>0.3). Refusing to assume 0% overlap (= full "
                    "tile height). Pass an explicit step_row, or check tile overlap/order.")
            step_row = row_est[1]
        col_cross = col_est[1] if col_est is not None else 0
        row_cross = row_est[0] if row_est is not None else 0
    else:
        col_cross = row_cross = 0
    col_step = (int(step_col), int(col_cross))
    row_step = (int(row_cross), int(step_row))
    if verbose:
        print(f"assembling {len(tiles)} tiles ({len(ys)}Y x {len(xs)}X), "
              f"tile {tw}x{th}px, col-step {col_step}px, row-step {row_step}px  "
              f"overlap ({100*(1-step_col/tw):.0f}%,{100*(1-step_row/th):.0f}%)")

    raw_origins = {
        key: ((key[1] - min(xs)) * col_step[0] + (key[0] - min(ys)) * row_step[0],
              (key[1] - min(xs)) * col_step[1] + (key[0] - min(ys)) * row_step[1])
        for key in tiles
    }
    min_c = min(c for c, _ in raw_origins.values())
    min_r = min(r for _, r in raw_origins.values())
    origins = {key: (c - min_c, r - min_r) for key, (c, r) in raw_origins.items()}
    W = max(c + tw for c, _ in origins.values())
    H = max(r + th for _, r in origins.values())
    height = np.zeros((H, W), vk0.height_raw.dtype)
    has_int = vk0.intensity is not None
    inten = np.zeros((H, W), vk0.intensity.dtype) if has_int else None  # carry full precision (not uint16)
    for (y, x), vk in tiles.items():           # stamp each tile's raw height (no Z normalisation)
        c0, r0 = origins[(y, x)]
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
                         step_col_row=col_step[1], step_row_col=row_step[0],
                         y_indices=tuple(ys), x_indices=tuple(xs))


if __name__ == "__main__":       # pragma: no cover
    import sys
    s = assemble_tiles(sys.argv[1] if len(sys.argv) > 1 else
                       Path(__file__).parent / "VK4")
    print(s.width, s.height, s.step_col, s.step_row)
