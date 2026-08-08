"""
Geometry extraction from the fabrication DXF for the UV-laser pin-fin study (v2).

Unlike v1 (where every array's nominal diameter/pitch was hand-transcribed into a
Python table), v2 reads the *actual drawn geometry* straight out of the DXF, so the
design never has to be re-typed and any layout change is picked up automatically.

What we read
------------
* **Pins**  — every ``CIRCLE`` is one pin. Its centre is the pin location and 2*radius
  is the drawn diameter.
* **Alignment marker** — a small (~200 um) closed square ``LWPOLYLINE`` at the bottom-left
  of a unit cell. Its presence marks the drawing (or a tile of it) as a *unit cell*, and
  its min corner is the origin the profilometer aligns to. Pin coordinates are reported
  relative to this origin so they can be matched to a VK4 scan (see ``register.py``).
* **Cell boundary** — the large closed square ``LWPOLYLINE`` enclosing a cell (optional).

Units
-----
DXF ``$INSUNITS = 4`` -> millimetres. All *stored* lengths in this module are converted
to **micrometres** to match the VK4 pipeline; absolute DXF coordinates are kept in mm
only where noted (``marker_origin_mm``, ``bbox_mm``).

Structure produced
-------------------
``read_design(path) -> DXFDesign``
    ``.cells``            list[UnitCell]           one per alignment marker (>=1)
    ``UnitCell.arrays``   list[PinArray]           the pin-fin arrays in that cell
    ``PinArray``          diameter / pitch / nx,ny / centres (um, marker-relative) / band

An "array" is a contiguous rectangular block of same-diameter pins on a regular grid.
Arrays are found by grouping pins by drawn diameter and then splitting each diameter group
into spatially-connected components, so two arrays that share a diameter (placed apart) are
still separated, while neighbouring arrays of *different* diameter never merge even when the
inter-array gap happens to equal the intra-array pitch (which it does in this design).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import ezdxf
except ImportError as e:  # pragma: no cover - dependency check
    raise SystemExit(
        "ezdxf is required for the DXF reader.  pip install ezdxf"
    ) from e

# ------------------------------------------------------------------ constants #
MM_TO_UM = 1000.0

# alignment-marker acceptance windows. Two marker styles are supported:
#   * square (DEPRECATED) — a ~200 um filled square whose bottom-left corner IS the cell origin.
#   * L      — an asymmetric "L" fiducial (short 50 um-wide arms). Its corner is inset from the
#              cell origin (offset derived from the cell-boundary square), and its asymmetry lets
#              registration resolve the sample's mirror/rotation (see register.py).
MARKER_NOMINAL_UM = 200.0
MARKER_TOL_UM = 40.0            # accept 160..240 um squares as the (deprecated) square marker
SQUARE_ASPECT_TOL = 0.15        # |w-h|/max(w,h) below this counts as "square"
SQUARE_FILL_MIN = 0.85         # enclosed-area / bbox-area at/above which a square counts as filled
L_AREA_RATIO = (0.35, 0.80)    # enclosed-area / bbox-area band for an L (a filled square is ~1.0)
L_MARKER_UM = (80.0, 300.0)    # accept L fiducials whose larger bbox side is in this range

# clustering / grid tolerances
DIAM_ROUND_UM = 0.1             # pins whose diameters agree to this share a "diameter"
PITCH_MERGE_FRAC = 0.40         # unique-coordinate merge tol as a fraction of pitch
CC_LINK_FRAC = 1.6             # connect pins within this * (min NN spacing) of a diameter
BAND_GAP_FRAC = 0.6             # y-gap (fraction of array height) that starts a new band


# --------------------------------------------------------------------------- #
@dataclass
class PinArray:
    """One rectangular block of identical, regularly-spaced pins."""

    array_id: int                 # unique within the cell (ordered band-major, then x)
    band: int                     # 1-based row-of-arrays index, 1 = bottom
    col: int                      # 1-based column within the band, 1 = left
    diameter_um: float            # drawn diameter (2 * circle radius), mean over the block
    diameter_std_um: float        # spread of drawn diameters in the block (should be ~0)
    pitch_x_um: float
    pitch_y_um: float
    nx: int
    ny: int
    n_pins: int
    centers_um: np.ndarray        # (n,2) pin (x,y) in um, relative to the marker origin
    cx_um: float                  # block centroid x (um, marker-relative)
    cy_um: float
    x0_um: float                  # block bounding box (um, marker-relative)
    y0_um: float
    x1_um: float
    y1_um: float
    lattice_vectors: "np.ndarray | None" = None
    #   (2,2) primitive translation vectors [a1; a2] (um) of the pin lattice, so every pin sits at
    #   origin + i*a1 + j*a2 for integers i,j.  Axis-aligned ([[px,0],[0,py]]) for the usual
    #   rectangular grid; genuinely oblique for a triangular/hex lattice.  ``nx``/``ny`` are the
    #   integer index extents along a1/a2 (== the unique-line counts for a rectangular grid).  This
    #   lets registration/extraction treat any Bravais lattice -- not just a square one -- correctly.

    @property
    def width_um(self) -> float:
        return self.x1_um - self.x0_um

    @property
    def height_um(self) -> float:
        return self.y1_um - self.y0_um

    @property
    def pitch_um(self) -> float:
        return 0.5 * (self.pitch_x_um + self.pitch_y_um)

    def __repr__(self) -> str:
        return (f"PinArray(id={self.array_id} band{self.band}c{self.col} "
                f"D={self.diameter_um:.1f}um {self.nx}x{self.ny} "
                f"pitch=({self.pitch_x_um:.1f},{self.pitch_y_um:.1f})um "
                f"@({self.cx_um:.0f},{self.cy_um:.0f})um)")


@dataclass
class UnitCell:
    """One instance of the tiled unit-cell pattern (anchored by its alignment marker)."""

    cell_id: int                       # 1-based, ordered left->right, bottom->top
    marker_origin_mm: tuple            # (x,y) min corner of the marker square, absolute DXF mm
    marker_center_mm: tuple            # (x,y) centre of the marker square, absolute DXF mm
    marker_size_um: float              # measured marker side length (um)
    bbox_mm: tuple                     # (x0,y0,x1,y1) cell bounding box, absolute DXF mm
    arrays: list                       # list[PinArray]
    n_pins: int
    marker_shape: str = ""             # "square" (deprecated) | "L" | "" (no marker)
    marker_polygon_um: np.ndarray | None = None   # marker vertices (N,2 um) relative to the cell
    #                                               origin -- the scan rasterises this to locate the
    #                                               cell and (for the asymmetric L) resolve mirror/rot

    @property
    def size_um(self) -> tuple:
        x0, y0, x1, y1 = self.bbox_mm
        return ((x1 - x0) * MM_TO_UM, (y1 - y0) * MM_TO_UM)

    @property
    def n_arrays(self) -> int:
        return len(self.arrays)

    @property
    def n_bands(self) -> int:
        return max((a.band for a in self.arrays), default=0)

    def all_centers_um(self) -> np.ndarray:
        """Every pin centre in the cell, (N,2) um, relative to the marker origin."""
        if not self.arrays:
            return np.empty((0, 2))
        return np.vstack([a.centers_um for a in self.arrays])

    def __repr__(self) -> str:
        w, h = self.size_um
        return (f"UnitCell(id={self.cell_id} {self.n_arrays} arrays / {self.n_pins} pins, "
                f"{w:.0f}x{h:.0f} um, marker@{self.marker_origin_mm})")


@dataclass
class DXFDesign:
    """Everything read out of one DXF file."""

    path: str
    units: str
    is_unit_cell: bool                 # True if >=1 alignment marker was found
    cells: list                        # list[UnitCell]
    n_markers: int
    cell_pitch_mm: tuple               # (dx,dy) tile pitch for a tiled grid, or (nan,nan)
    boundary_bbox_mm: tuple | None     # bbox of the largest square, if any
    raw_bbox_mm: tuple                 # bbox of all pins

    @property
    def n_pins(self) -> int:
        return sum(c.n_pins for c in self.cells)

    def summary(self) -> str:
        lines = [f"DXFDesign({Path(self.path).name}, units={self.units}, "
                 f"{'unit-cell' if self.is_unit_cell else 'raw-design'}, "
                 f"{len(self.cells)} cell(s), {self.n_pins} pins)"]
        for c in self.cells:
            lines.append(f"  {c!r}")
            for a in c.arrays:
                lines.append(f"      {a!r}")
        return "\n".join(lines)


def validate_equivalent_cells(design: DXFDesign, atol_um=1e-3) -> None:
    """Reject a tiled DXF whose parsed unit cells are not geometrically identical.

    The production analyzer registers repeated copies of one template. Silently selecting
    ``cells[0]`` for a heterogeneous drawing would measure later cells with the wrong pin geometry.
    Coordinates are already marker-relative, so a direct tolerance comparison is translation
    invariant.
    """
    if len(design.cells) <= 1:
        return
    ref = design.cells[0]

    def fail(cell, detail):
        raise ValueError(
            f"{Path(design.path).name}: tiled DXF cells are not geometrically equivalent; "
            f"cell {ref.cell_id} vs cell {cell.cell_id}: {detail}. Production analysis requires "
            "one repeated unit-cell geometry; split heterogeneous designs or add per-cell support.")

    for cell in design.cells[1:]:
        if cell.marker_shape != ref.marker_shape:
            fail(cell, f"marker shape {ref.marker_shape!r} != {cell.marker_shape!r}")
        if not np.allclose(cell.size_um, ref.size_um, atol=atol_um, rtol=0.0):
            fail(cell, f"cell size {ref.size_um} != {cell.size_um}")
        rp, cp = ref.marker_polygon_um, cell.marker_polygon_um
        if (rp is None) != (cp is None):
            fail(cell, "marker polygon presence differs")
        if rp is not None and (rp.shape != cp.shape
                               or not np.allclose(rp, cp, atol=atol_um, rtol=0.0)):
            fail(cell, "marker polygon differs")
        if len(cell.arrays) != len(ref.arrays) or cell.n_pins != ref.n_pins:
            fail(cell, f"array/pin counts {(len(ref.arrays), ref.n_pins)} != "
                 f"{(len(cell.arrays), cell.n_pins)}")
        for i, (a, b) in enumerate(zip(ref.arrays, cell.arrays), 1):
            discrete_a = (a.band, a.col, a.nx, a.ny, a.n_pins)
            discrete_b = (b.band, b.col, b.nx, b.ny, b.n_pins)
            if discrete_a != discrete_b:
                fail(cell, f"array {i} topology {discrete_a} != {discrete_b}")
            numeric_a = (a.diameter_um, a.pitch_x_um, a.pitch_y_um)
            numeric_b = (b.diameter_um, b.pitch_x_um, b.pitch_y_um)
            if not np.allclose(numeric_a, numeric_b, atol=atol_um, rtol=0.0):
                fail(cell, f"array {i} diameter/pitch {numeric_a} != {numeric_b}")
            if (a.centers_um.shape != b.centers_um.shape
                    or not np.allclose(a.centers_um, b.centers_um, atol=atol_um, rtol=0.0)):
                fail(cell, f"array {i} pin centers differ")


# ---------------------------------------------------------------- DXF reading #
def _poly_area(verts):
    """Enclosed (shoelace) area of a closed polygon, given its vertices (N,2). Sign-agnostic."""
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _read_entities(path):
    """Return (pins Nx3 [x,y,r] in mm, polys list, units str).

    Each ``poly`` is a dict describing one closed LWPOLYLINE/POLYLINE: ``verts`` (N,2 mm),
    ``bbox`` (x0,y0,x1,y1 mm), ``area_mm2`` (enclosed area), ``n_verts``. Marker/boundary
    classification (square vs L vs cell outline) happens downstream from these."""
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    insunits = doc.header.get("$INSUNITS", 4)
    units = {0: "unitless", 1: "in", 4: "mm", 5: "cm", 6: "m", 13: "um"}.get(insunits, "mm")

    pins = []
    polys = []
    for e in msp:
        t = e.dxftype()
        if t == "CIRCLE":
            c = e.dxf.center
            pins.append((float(c.x), float(c.y), float(e.dxf.radius)))
        elif t in ("LWPOLYLINE", "POLYLINE"):
            try:
                if t == "LWPOLYLINE":
                    pts = np.array([(p[0], p[1]) for p in e.get_points()], float)
                else:  # old-style POLYLINE
                    pts = np.array([(v.dxf.location.x, v.dxf.location.y)
                                    for v in e.vertices], float)
            except Exception:
                continue
            if len(pts) < 4:
                continue
            x0, y0 = pts[:, 0].min(), pts[:, 1].min()
            x1, y1 = pts[:, 0].max(), pts[:, 1].max()
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            polys.append(dict(verts=pts, bbox=(x0, y0, x1, y1),
                              area_mm2=_poly_area(pts), n_verts=len(pts)))

    return np.array(pins, float).reshape(-1, 3), polys, units


# ----------------------------------------------------------- marker handling #
def _marker_shape(poly, to_um):
    """Classify a closed polyline as an alignment marker: 'square' (deprecated), 'L', or None.

    Square: ~square bbox, near-filled, ~200 um side. L: a 6-vertex outline whose enclosed area
    is well below its bounding box (an L fills ~0.5-0.6 of its bbox; a filled square fills ~1.0),
    with an arm-scale bbox. Everything else (pins are CIRCLEs; the big cell outline is too large)
    falls through to None."""
    x0, y0, x1, y1 = poly["bbox"]
    w_um, h_um = (x1 - x0) * to_um, (y1 - y0) * to_um
    bbox_area = (x1 - x0) * (y1 - y0)
    ratio = poly["area_mm2"] / bbox_area if bbox_area > 0 else 0.0
    aspect = abs(w_um - h_um) / max(w_um, h_um)
    side_um = 0.5 * (w_um + h_um)
    if (aspect <= SQUARE_ASPECT_TOL and ratio >= SQUARE_FILL_MIN
            and abs(side_um - MARKER_NOMINAL_UM) <= MARKER_TOL_UM):
        return "square"
    if (poly["n_verts"] == 6 and L_AREA_RATIO[0] <= ratio <= L_AREA_RATIO[1]
            and L_MARKER_UM[0] <= max(w_um, h_um) <= L_MARKER_UM[1]):
        return "L"
    return None


def _find_markers(polys, to_um):
    """Return alignment markers (square [deprecated] or L) as dicts.

    Each marker carries ``ref_mm`` (bbox-min corner, the detectable anchor), ``center_mm``,
    ``size_um`` (mean bbox side), ``shape`` ('square'|'L') and ``verts_mm`` (the polygon). The
    DESIGN origin a cell's pins are relative to is resolved later (``read_design``): the square
    sits AT the origin so its corner IS the origin; the L is inset, so the origin comes from the
    cell-boundary corner and the L's offset is derived from the geometry."""
    markers = []
    for poly in polys:
        shape = _marker_shape(poly, to_um)
        if shape is None:
            continue
        x0, y0, x1, y1 = poly["bbox"]
        markers.append(dict(
            ref_mm=(x0, y0),
            center_mm=(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
            size_um=0.5 * ((x1 - x0) + (y1 - y0)) * to_um,
            shape=shape,
            verts_mm=poly["verts"],
        ))
    # order tiled markers left->right, bottom->top (row-major with a y tolerance)
    if markers:
        ys = np.array([m["ref_mm"][1] for m in markers])
        ytol = 0.25 * np.ptp(ys) / max(1, len(set(np.round(ys, 3)))) if np.ptp(ys) else 1.0
        markers.sort(key=lambda m: (round(m["ref_mm"][1] / max(ytol, 1e-6)), m["ref_mm"][0]))
    return markers


def _boundary_square(polys, markers, to_um):
    """Largest ~square polyline that is not an alignment marker -> the cell/design boundary.

    A marker is identified by its FULL bounding box, not just its lower-left corner: a cell
    boundary square commonly shares the (0,0) corner with the origin-anchored square marker, and
    matching on the corner alone would discard the real boundary (leaving cell size_um = the pin
    cluster span instead of the true cell size)."""
    def _bbox(v):
        return (round(float(v[:, 0].min()), 4), round(float(v[:, 1].min()), 4),
                round(float(v[:, 0].max()), 4), round(float(v[:, 1].max()), 4))
    marker_bboxes = {_bbox(m["verts_mm"]) for m in markers}
    best, best_area = None, -1.0
    for poly in polys:
        x0, y0, x1, y1 = poly["bbox"]
        if (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)) in marker_bboxes:
            continue                                       # this polyline IS a marker
        w, h = x1 - x0, y1 - y0
        if abs(w - h) / max(w, h) > SQUARE_ASPECT_TOL:     # the boundary is a ~square outline
            continue
        area = w * h
        if area > best_area:
            best, best_area = (x0, y0, x1, y1), area
    return best


def _assign_cells(pins_xy_mm, markers, dx, dy, span_x, span_y):
    """Assign each pin to its bottom-left anchor marker.

    Returns a list of boolean masks (one per marker). Robust to a degenerate 1xN row or Nx1
    column of cells, where one of ``dx``/``dy`` is NaN (all markers share that coordinate):
    the finite axis separates the cells and the degenerate axis uses the full pin span, so
    pins are never assigned to every cell (the failure mode of a naive isfinite(dx&dy) gate).
    """
    origins = np.array([m["ref_mm"] for m in markers], float)        # (M,2) marker anchor corners
    wx = dx if np.isfinite(dx) else (span_x + 1.0)                   # window width per axis
    wy = dy if np.isfinite(dy) else (span_y + 1.0)
    tol = 0.02 * min(wx, wy)
    px = pins_xy_mm[:, 0][:, None]; py = pins_xy_mm[:, 1][:, None]   # (N,1)
    ox = origins[:, 0][None, :]; oy = origins[:, 1][None, :]         # (1,M)
    offx, offy = px - ox, py - oy                                    # (N,M)
    valid = (offx >= -tol) & (offx < wx) & (offy >= -tol) & (offy < wy)
    key = np.where(valid, offx + offy, np.inf)                       # nearest bottom-left
    best = np.argmin(key, axis=1)
    has = np.isfinite(np.take_along_axis(key, best[:, None], axis=1)).ravel()
    return [(best == j) & has for j in range(len(markers))], (wx, wy)


def _cell_pitch(markers):
    """Estimate the tile pitch (dx,dy) of a tiled marker grid."""
    if len(markers) < 2:
        return (float("nan"), float("nan"))
    ox = np.array([m["ref_mm"][0] for m in markers])
    oy = np.array([m["ref_mm"][1] for m in markers])

    def min_positive_gap(v):
        u = np.array(sorted(set(np.round(v, 4))))
        d = np.diff(u)
        d = d[d > 1e-6]
        return float(np.min(d)) if d.size else float("nan")

    return (min_positive_gap(ox), min_positive_gap(oy))


# --------------------------------------------------------- array clustering #
def _connected_components(xy, link_dist):
    """Label points into connected components where an edge links points within link_dist.
    Uses a KD-tree; falls back to a simple union-find so scipy is optional here."""
    n = len(xy)
    if n == 0:
        return np.array([], int), 0
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=link_dist, output_type="ndarray")

    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels, labels.max() + 1 if n else 0


def _unique_axis(v, pitch):
    """Collapse coordinates to grid lines (values within PITCH_MERGE_FRAC*pitch merge)."""
    v = np.sort(v)
    tol = max(PITCH_MERGE_FRAC * pitch, 1e-6)
    lines = [v[0]]
    for x in v[1:]:
        if x - lines[-1] > tol:
            lines.append(x)
        else:
            lines[-1] = 0.5 * (lines[-1] + x)   # running merge
    return np.array(lines)


def _nn_spacing(xy):
    """Median nearest-neighbour distance (the array pitch estimate)."""
    if len(xy) < 2:
        return float("nan")
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


# ------------------------------------------------- general lattice basis ---- #
# Not every pin array is a square (axis-aligned rectangular) grid: a triangular / hexagonal layout
# packs alternate rows a half-pitch apart, so its two primitive translation vectors are NOT
# orthogonal.  Projecting such a lattice onto independent x/y grid lines (``_unique_axis``) mis-reads
# it as a half-filled rectangle (wrong pitch, ``n_pins != nx*ny``).  We therefore fit the actual
# primitive basis (a1,a2) and only keep the axis-aligned unique-line path when the array really IS a
# filled rectangular grid, so existing square layouts are unchanged bit-for-bit.
RECT_FILL_MIN = 0.85        # filled rectangular grid: n_pins ~ nx*ny (a triangular grid fills ~0.5)
AXIS_ALIGNED_TOL = 0.03     # min(|vx|,|vy|)/|v| below this => an axis-aligned (rectangular) basis vec
LATTICE_RESID_FRAC = 0.12   # max per-pin index residual (fraction of a lattice step) for a clean fit


def _gauss_reduce(a, b):
    """Lagrange/Gauss lattice reduction -> the two shortest primitive vectors of the same lattice."""
    a = np.array(a, float); b = np.array(b, float)
    for _ in range(100):
        if a @ a > b @ b:
            a, b = b, a
        m = round((a @ b) / (a @ a)) if (a @ a) > 0 else 0
        b2 = b - m * a
        if np.allclose(b2, b):
            break
        b = b2
    if a @ a > b @ b:
        a, b = b, a
    return a, b


def _axis_aligned_basis(a1, a2, tol=AXIS_ALIGNED_TOL):
    """True when (a1,a2) is an axis-aligned rectangular basis (one ~x, one ~y)."""
    def axisness(v):
        L = float(np.hypot(*v))
        return (min(abs(float(v[0])), abs(float(v[1]))) / L) if L > 0 else 1.0
    return axisness(a1) < tol and axisness(a2) < tol


def _fit_lattice_basis(xy):
    """Detect the primitive lattice vectors of a regular 2-D point set (ANY Bravais lattice).

    Returns ``dict(a1, a2, indices, resid_max)`` where ``indices`` are the integer per-pin lattice
    indices (``xy = origin + i*a1 + j*a2``), or ``None`` when the points are not one clean lattice.
    Robust to square, rectangular and oblique/triangular layouts alike.
    """
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, float)
    if len(xy) < 4:
        return None
    tree = cKDTree(xy)
    k = min(len(xy), 7)
    d, idx = tree.query(xy, k=k)
    nn = float(np.median(d[:, 1]))
    if not (np.isfinite(nn) and nn > 0):
        return None
    vecs = []                                        # short neighbour displacements, sign-canonical
    for i in range(len(xy)):
        for jj in range(1, k):
            v = xy[idx[i, jj]] - xy[i]
            if np.hypot(v[0], v[1]) <= 1.35 * nn:
                if v[1] < -1e-9 or (abs(v[1]) <= 1e-9 and v[0] < 0):
                    v = -v                           # fold +v and -v into one cluster
                vecs.append(v)
    if len(vecs) < 2:
        return None
    vecs = np.asarray(vecs, float)
    q = np.round(vecs / (nn / 4.0)).astype(int)      # cluster near-identical short vectors
    _uniq, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    means = np.array([vecs[inv == u].mean(axis=0) for u in range(len(_uniq))])
    order = np.argsort(-cnt)
    means = means[order]
    a1 = means[0]                                     # most common short vector
    a2 = None
    for m in means[1:]:                               # shortest common vector not parallel to a1
        cross = abs(float(a1[0] * m[1] - a1[1] * m[0]))
        if cross > 0.2 * float(np.hypot(*a1) * np.hypot(*m)):     # angle from a1 > ~11.5 deg
            a2 = m
            break
    if a2 is None:
        return None
    a1, a2 = _gauss_reduce(a1, a2)
    M = np.column_stack([a1, a2])                     # xy - origin = M @ [i, j]^T
    if abs(float(np.linalg.det(M))) < 1e-9:
        return None
    o = xy[np.lexsort((xy[:, 0], xy[:, 1]))[0]]       # bottom-left pin as the index origin
    frac = (xy - o) @ np.linalg.inv(M).T              # fractional lattice indices (step = 1.0)
    ij = np.round(frac)
    resid_max = float(np.abs(frac - ij).max())        # deviation from a node, in lattice-step units
    if resid_max > LATTICE_RESID_FRAC:
        return None
    ij = ij.astype(int)
    if len(np.unique(ij, axis=0)) != len(ij):        # two pins share a node -> not a clean lattice
        return None
    return dict(a1=a1, a2=a2, indices=ij, resid_max=resid_max)


def _build_arrays(pins_um, origin_um):
    """pins_um: (N,3) [x,y,r] in um (absolute). origin_um: (ox,oy) marker origin um.
    Return list[PinArray] with coordinates relative to origin_um (band/col unset)."""
    if len(pins_um) == 0:
        return []
    xy = pins_um[:, :2]
    dia = pins_um[:, 2] * 2.0

    arrays = []
    for d in sorted(set(np.round(dia / DIAM_ROUND_UM).astype(int))):
        m = np.round(dia / DIAM_ROUND_UM).astype(int) == d
        gxy = xy[m]
        gdia = dia[m]
        # split same-diameter pins into spatially separate blocks
        nn = _nn_spacing(gxy)
        link = (CC_LINK_FRAC * nn) if np.isfinite(nn) else 1e9
        labels, k = _connected_components(gxy, link)
        for lab in range(k):
            sel = labels == lab
            bxy = gxy[sel]
            if len(bxy) < 2:
                continue
            ux = _unique_axis(bxy[:, 0], nn)
            uy = _unique_axis(bxy[:, 1], nn)
            px = float(np.median(np.diff(ux))) if len(ux) > 1 else float(nn)
            py = float(np.median(np.diff(uy))) if len(uy) > 1 else float(nn)
            nx, ny = len(ux), len(uy)
            lat_vecs = np.array([[px, 0.0], [0.0, py]], float)   # axis-aligned (rectangular) default
            # A filled rectangular grid has n_pins ~ nx*ny; a triangular/oblique lattice fills only
            # part of its index box, so ONLY then do we fit a general primitive basis.  This keeps
            # every existing square/rectangular array on the exact unique-line pitch/nx/ny it had.
            if len(bxy) < RECT_FILL_MIN * nx * ny:
                fit = _fit_lattice_basis(bxy)
                if fit is not None and not _axis_aligned_basis(fit["a1"], fit["a2"]):
                    a1, a2 = fit["a1"], fit["a2"]
                    ij = fit["indices"]
                    nx = int(ij[:, 0].max() - ij[:, 0].min() + 1)   # index extents along a1, a2
                    ny = int(ij[:, 1].max() - ij[:, 1].min() + 1)
                    px = float(np.hypot(a1[0], a1[1]))              # |a1|,|a2| (== NN pitch for a
                    py = float(np.hypot(a2[0], a2[1]))             #  triangular lattice) as the scale
                    lat_vecs = np.array([a1, a2], float)
            rel = bxy - np.asarray(origin_um)
            arrays.append(PinArray(
                array_id=-1, band=-1, col=-1,
                diameter_um=float(np.mean(gdia[sel])),
                diameter_std_um=float(np.std(gdia[sel])),
                pitch_x_um=px, pitch_y_um=py,
                nx=nx, ny=ny, n_pins=int(sel.sum()),
                centers_um=rel,
                cx_um=float(rel[:, 0].mean()), cy_um=float(rel[:, 1].mean()),
                x0_um=float(rel[:, 0].min()), y0_um=float(rel[:, 1].min()),
                x1_um=float(rel[:, 0].max()), y1_um=float(rel[:, 1].max()),
                lattice_vectors=lat_vecs,
            ))
    return arrays


def _assign_bands(arrays):
    """Sort arrays into bands (rows of arrays) by y-centroid, then columns by x."""
    if not arrays:
        return arrays
    arrays = sorted(arrays, key=lambda a: a.cy_um)
    heights = np.array([a.height_um for a in arrays])
    typ_h = float(np.median(heights)) if len(heights) else 0.0
    band_gap = max(BAND_GAP_FRAC * typ_h, 1.0)

    band_idx = 0
    prev_cy = arrays[0].cy_um
    for a in arrays:
        if a.cy_um - prev_cy > band_gap:
            band_idx += 1
        object.__setattr__(a, "band", band_idx + 1)  # 1-based
        prev_cy = a.cy_um
    # columns within a band, ordered by x
    out = []
    for b in sorted(set(a.band for a in arrays)):
        row = sorted([a for a in arrays if a.band == b], key=lambda a: a.cx_um)
        for c, a in enumerate(row, start=1):
            a.col = c
            out.append(a)
    # global id ordered band-major then column
    out.sort(key=lambda a: (a.band, a.col))
    for i, a in enumerate(out, start=1):
        a.array_id = i
    return out


# ---------------------------------------------------------------- entry point #
def read_design(path: str | Path) -> DXFDesign:
    """Parse a DXF into a :class:`DXFDesign`.  See module docstring."""
    path = Path(path)
    pins_mm, polys, units = _read_entities(path)
    to_um = MM_TO_UM if units in ("mm",) else {
        "cm": 1e4, "m": 1e6, "in": 25400.0, "um": 1.0, "unitless": MM_TO_UM,
    }.get(units, MM_TO_UM)
    # The cell-size / tile-pitch fields below are stored in the file's NATIVE unit but consumed as
    # millimetres (UnitCell.size_um multiplies by MM_TO_UM; register.py multiplies pitch by 1000).
    # That holds only for a mm (or unitless->mm) DXF; a non-mm DXF would silently mis-scale cell size
    # and pitch (a 1000x-too-large size blows up register's min_sep and drops every cell). Pin
    # coordinates ARE unit-correct via to_um, but fail loudly rather than register on a bad cell size.
    if to_um != MM_TO_UM:
        raise ValueError(f"DXF uses '{units}' units; only millimetre DXFs are supported. Re-export "
                         f"in mm ($INSUNITS=4) -- cell-size / tile-pitch handling assumes mm.")

    if len(pins_mm) == 0:
        raise ValueError(f"{path.name}: no CIRCLE entities (pins) found in DXF")

    pins_um = pins_mm.copy()
    pins_um[:, :3] *= to_um                      # x,y,r all to um

    markers = _find_markers(polys, to_um)
    boundary = _boundary_square(polys, markers, to_um)
    cell_pitch = _cell_pitch(markers)
    raw_bbox_mm = (float(pins_mm[:, 0].min()), float(pins_mm[:, 1].min()),
                   float(pins_mm[:, 0].max()), float(pins_mm[:, 1].max()))

    cells: list[UnitCell] = []

    if markers:
        dx, dy = cell_pitch
        span_x = raw_bbox_mm[2] - raw_bbox_mm[0]
        span_y = raw_bbox_mm[3] - raw_bbox_mm[1]
        if len(markers) > 1:
            # tiled cells: assign each pin to its bottom-left anchor marker
            masks, (wx, wy) = _assign_cells(pins_mm[:, :2], markers, dx, dy, span_x, span_y)
        else:
            masks, (wx, wy) = [np.ones(len(pins_mm), bool)], (span_x, span_y)
        for k, (m, mask) in enumerate(zip(markers, masks), start=1):
            ref_mm = m["ref_mm"]
            # DESIGN origin (pins are reported relative to it). The deprecated square sits ON the
            # origin, so its own corner IS the origin. The L is inset, so a single-cell L takes the
            # origin from the cell-boundary corner and its offset is encoded in marker_polygon_um.
            if m["shape"] == "L" and len(markers) == 1 and boundary is not None:
                ox_mm, oy_mm = boundary[0], boundary[1]
            else:
                ox_mm, oy_mm = ref_mm
            ox_um, oy_um = ox_mm * to_um, oy_mm * to_um
            if len(markers) > 1:
                cell_pins_um = pins_um[mask]
                bbox_mm = (ref_mm[0], ref_mm[1], ref_mm[0] + wx, ref_mm[1] + wy)
            else:
                cell_pins_um = pins_um            # single cell: everything belongs to it
                bbox_mm = boundary if boundary is not None else raw_bbox_mm

            arrays = _assign_bands(_build_arrays(cell_pins_um, (ox_um, oy_um)))
            marker_poly_um = (np.asarray(m["verts_mm"], float) - np.array([ox_mm, oy_mm])) * to_um
            cells.append(UnitCell(
                cell_id=k,
                marker_origin_mm=(ox_mm, oy_mm),
                marker_center_mm=m["center_mm"],
                marker_size_um=m["size_um"],
                bbox_mm=bbox_mm,
                arrays=arrays,
                n_pins=sum(a.n_pins for a in arrays),
                marker_shape=m["shape"],
                marker_polygon_um=marker_poly_um,
            ))
    else:
        # no alignment marker: treat the whole drawing as one anchor-less "cell"
        ox_um, oy_um = raw_bbox_mm[0] * to_um, raw_bbox_mm[1] * to_um
        arrays = _assign_bands(_build_arrays(pins_um, (ox_um, oy_um)))
        cells.append(UnitCell(
            cell_id=1,
            marker_origin_mm=(raw_bbox_mm[0], raw_bbox_mm[1]),
            marker_center_mm=(0.5 * (raw_bbox_mm[0] + raw_bbox_mm[2]),
                              0.5 * (raw_bbox_mm[1] + raw_bbox_mm[3])),
            marker_size_um=float("nan"),
            bbox_mm=(boundary if boundary is not None else raw_bbox_mm),
            arrays=arrays,
            n_pins=sum(a.n_pins for a in arrays),
        ))

    return DXFDesign(
        path=str(path), units=units, is_unit_cell=bool(markers),
        cells=cells, n_markers=len(markers), cell_pitch_mm=cell_pitch,
        boundary_bbox_mm=boundary, raw_bbox_mm=raw_bbox_mm,
    )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
    else:                                              # first DXF present, not a hardcoded name
        p = next((Path(__file__).resolve().parent.parent / "DXF").glob("*.dxf"), None)
        if p is None:
            raise SystemExit("no DXF given and none found in DXF/")
    design = read_design(p)
    print(design.summary())
    print(f"\nmarkers={design.n_markers}  cell_pitch_mm={design.cell_pitch_mm}  "
          f"boundary_mm={design.boundary_bbox_mm}")
    for c in design.cells:
        print(f"\ncell {c.cell_id}: {c.n_arrays} arrays, {c.n_bands} bands, "
              f"{c.n_pins} pins, size {c.size_um[0]:.0f}x{c.size_um[1]:.0f} um")
        for a in c.arrays:
            print(f"  band{a.band} col{a.col}: D={a.diameter_um:6.2f}um "
                  f"(+-{a.diameter_std_um:.2f})  {a.nx}x{a.ny}={a.n_pins}  "
                  f"pitch=({a.pitch_x_um:.1f},{a.pitch_y_um:.1f})um  "
                  f"center=({a.cx_um:.0f},{a.cy_um:.0f})um")
