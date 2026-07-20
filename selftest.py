"""
End-to-end self-test on synthetic data (no real VK4 files required).

Validates the v2 pipeline that cannot be exercised on the empty ``VK4/`` folder yet:

  1. Registration recovers a known cell origin (single cell and a tiled 2-cell scan).
  2. Extraction recovers the known diameter / pitch / depth for every array.
  3. The full plot suite renders from a synthetic measurements table.

Run:  python selftest.py       (writes previews/plots under Results/selftest/)
Exit code is non-zero if any check fails, so it can gate CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dxf_geometry import read_design
from register import register_scan
from extract import ArraySample, extract_array
from synth import synth_scan
import report as ra

HERE = Path(__file__).parent
OUT = HERE / "Results" / "legacy" / "selftest"


class Checker:
    def __init__(self):
        self.fails = []
        self.n = 0

    def check(self, cond, msg):
        self.n += 1
        ok = bool(cond)
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            self.fails.append(msg)
        return ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ck = Checker()

    dxf = next((HERE / "DXF").glob("*.dxf"))
    design = read_design(dxf)
    template = design.cells[0]
    print(f"DXF: {template.n_arrays} arrays / {template.n_pins} pins, "
          f"marker {template.marker_size_um:.0f} µm\n")

    # -------------------------------------------------- 1. single-cell scan #
    print("[1] single-cell registration + extraction")
    xppx = yppx = 2.0
    origin = (137.0, 101.0)
    depth = 30.0
    scan, truth = synth_scan(template, x_um_per_px=xppx, y_um_per_px=yppx,
                             origin_px=origin, depth_um=depth, floor_um=60.0,
                             noise_um=0.4, seed=1)
    pls = register_scan(scan, template, n_cells=1, cell_pitch_mm=design.cell_pitch_mm)
    pl = pls[0]
    print(f"    truth origin={origin}  recovered=({pl.origin_col:.1f},{pl.origin_row:.1f}) "
          f"yflip={pl.y_up:+d} method={pl.method} score={pl.score:.2f}")
    ck.check(np.isfinite(pl.origin_col), "registration produced a finite origin")
    ck.check(pl.y_up == truth["y_up"], "y-flip resolved correctly")
    ck.check(abs(pl.origin_col - origin[0]) <= 6, "origin col within 6 px")
    ck.check(abs(pl.origin_row - origin[1]) <= 6, "origin row within 6 px")

    # extraction of every array
    band_target = ra._band_targets(template)
    rows, results = [], []
    dia_err, mid_err, depth_err, pitch_err = [], [], [], []
    for a in template.arrays:
        s = ArraySample(
            filename=f"synth_c1_b{a.band}c{a.col}_D{a.diameter_um:g}",
            vk4_stem="synth", cell_id=1, array_id=a.array_id, band=a.band, col=a.col,
            passes=20 if a.band <= 2 else 40, speed=400.0 + 100 * a.col,
            nominal_diameter_um=a.diameter_um, target_diameter_um=band_target[a.band],
            nominal_pitch_um=a.pitch_um, nominal_pitch_x_um=a.pitch_x_um,
            nominal_pitch_y_um=a.pitch_y_um, nx=a.nx, ny=a.ny, cx_um=a.cx_um, cy_um=a.cy_um)
        res = extract_array(scan, pl, a, s, make_qc=False)
        reliable = res.passes > 0 and not any(k in res.flags for k in ra.CRITICAL_FLAGS)
        rows.append(ra.result_to_row(res, reliable))
        results.append((s, res, reliable))
        if np.isfinite(res.top_diameter_um):
            dia_err.append(res.top_diameter_um - a.diameter_um)
        if np.isfinite(res.diameter_um):
            mid_err.append(res.diameter_um - a.diameter_um)
        if np.isfinite(res.depth_um):
            depth_err.append(res.depth_um - depth)
        pitch_err.append(res.pitch_um - a.pitch_um)

    dia_err = np.array(dia_err); mid_err = np.array(mid_err); depth_err = np.array(depth_err)
    print(f"    mid-diameter error: mean {mid_err.mean():+.2f} µm, "
          f"max |{np.abs(mid_err).max():.2f}| µm  (n={len(mid_err)})")
    print(f"    top-diameter error: mean {dia_err.mean():+.2f} µm, "
          f"max |{np.abs(dia_err).max():.2f}| µm")
    print(f"    depth error:        mean {depth_err.mean():+.2f} µm, "
          f"max |{np.abs(depth_err).max():.2f}| µm")
    ck.check(len(mid_err) == template.n_arrays, "all arrays produced a diameter")
    ck.check(np.abs(mid_err).max() <= 5.0, "every mid-diameter within 5 µm of drawn")
    ck.check(np.abs(dia_err).max() <= 6.0, "every top-diameter within 6 µm of drawn")
    ck.check(np.abs(depth_err).max() <= 4.0, "every depth within 4 µm of truth")
    ck.check(np.abs(pitch_err).max() <= 1e-6, "pitch equals design (known lattice)")
    reliable_frac = np.mean([r["reliable"] for r in rows])
    ck.check(reliable_frac == 1.0, "all synthetic arrays flagged reliable")

    # -------------------------------------------------- 2. tiled 2-cell scan #
    print("\n[2] tiled 2-cell registration")
    scan2, truth2 = synth_scan(template, x_um_per_px=xppx, y_um_per_px=yppx,
                               origin_px=(90.0, 80.0), n_cells=2, cell_gap_um=250.0,
                               depth_um=25.0, seed=2)
    pls2 = register_scan(scan2, template, n_cells=2, cell_pitch_mm=design.cell_pitch_mm)
    for i, (pl2, (tc, tr)) in enumerate(zip(pls2, truth2["origins"]), start=1):
        okc = np.isfinite(pl2.origin_col) and abs(pl2.origin_col - tc) <= 8
        okr = np.isfinite(pl2.origin_row) and abs(pl2.origin_row - tr) <= 8
        print(f"    cell {i}: truth=({tc:.0f},{tr:.0f}) "
              f"recovered=({pl2.origin_col:.0f},{pl2.origin_row:.0f})")
        ck.check(okc and okr, f"cell {i} origin within 8 px")
    ck.check(len({round(p.origin_col) for p in pls2}) == 2, "two distinct cells located")

    # -------------------------------------------------- 3. plots render #
    print("\n[3] plot suite renders")
    df = pd.DataFrame(rows)
    try:
        ra.make_plots(df, results, OUT)
        ra.print_diameter_calibration(df, OUT)
        figs = ["overview_3x3", "dose_collapse", "per_row", "diameter_fit",
                "depth_vs_dose", "grid_overlays"]
        for name in figs:
            ck.check((OUT / "figures" / f"{name}.png").exists(), f"{name}.png written")
        ck.check((OUT / "diameter_calibration.txt").exists(),
                 "diameter_calibration.txt written")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"plotting raised: {e!r}")

    # -------------------------------------------------- 4. tiled DXF parsing #
    print("\n[4] tiled DXF parsing (1xN row + 2x2 grid; regression for the NaN-pitch bug)")
    try:
        import ezdxf
        src = ezdxf.readfile(dxf)
        circles = [(c.dxf.center.x, c.dxf.center.y, c.dxf.radius)
                   for c in src.modelspace() if c.dxftype() == "CIRCLE"]
        npins = template.n_pins

        def _tiled(offsets, path):
            doc = ezdxf.new(); doc.header["$INSUNITS"] = 4        # mm, like the fab DXF
            m = doc.modelspace()
            for (ox, oy) in offsets:
                m.add_lwpolyline([(ox, oy), (ox + 0.2, oy), (ox + 0.2, oy + 0.2),
                                  (ox, oy + 0.2)], close=True)
                for (cx, cy, r) in circles:
                    m.add_circle((cx + ox, cy + oy), r)
            doc.saveas(path)
            return path

        row = _tiled([(0, 0), (3.0, 0)], OUT / "tiled_1x2.dxf")       # single row -> dy=NaN
        grid = _tiled([(0, 0), (3.0, 0), (0, 3.0), (3.0, 3.0)], OUT / "tiled_2x2.dxf")
        drow = read_design(row)
        dgrid = read_design(grid)
        ck.check(len(drow.cells) == 2 and all(c.n_pins == npins for c in drow.cells),
                 "1x2 row -> 2 cells, each full pin count (not merged)")
        ck.check(len(dgrid.cells) == 4 and all(c.n_arrays == template.n_arrays
                                               for c in dgrid.cells),
                 "2x2 grid -> 4 cells, each full array count")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"tiled DXF parsing raised: {e!r}")

    # -------------------------------------------------- 5. non-square pixels #
    print("\n[5] non-square pixels (xppx != yppx)")
    scan_ns, _ = synth_scan(template, x_um_per_px=2.0, y_um_per_px=1.0,
                            origin_px=(120.0, 90.0), depth_um=28.0, seed=4)
    pl_ns = register_scan(scan_ns, template, n_cells=1, cell_pitch_mm=design.cell_pitch_mm)[0]
    ck.check(np.isfinite(pl_ns.origin_col), "non-square: registration succeeded")
    ns_dia, ns_depth = [], []
    for a in template.arrays:
        s = ArraySample(filename="ns", vk4_stem="ns", cell_id=1, array_id=a.array_id,
                        band=a.band, col=a.col, passes=20, speed=400,
                        nominal_diameter_um=a.diameter_um, target_diameter_um=a.diameter_um,
                        nominal_pitch_um=a.pitch_um, nominal_pitch_x_um=a.pitch_x_um,
                        nominal_pitch_y_um=a.pitch_y_um, nx=a.nx, ny=a.ny,
                        cx_um=a.cx_um, cy_um=a.cy_um)
        r = extract_array(scan_ns, pl_ns, a, s, make_qc=False)
        if np.isfinite(r.diameter_um):
            ns_dia.append(r.diameter_um - a.diameter_um)
        if np.isfinite(r.depth_um):
            ns_depth.append(r.depth_um - 28.0)
    ns_dia = np.array(ns_dia); ns_depth = np.array(ns_depth)
    print(f"    non-square mid-Ø err max |{np.abs(ns_dia).max():.2f}| µm, "
          f"depth err max |{np.abs(ns_depth).max():.2f}| µm")
    ck.check(len(ns_dia) == template.n_arrays and np.abs(ns_dia).max() <= 5.0,
             "non-square: every mid-diameter within 5 µm")
    ck.check(np.abs(ns_depth).max() <= 4.0, "non-square: every depth within 4 µm")

    # -------------------------------------------------- 6. rotation-aware measure #
    print("\n[6] rotation-aware classification (manual override with rotation_deg)")
    rot = 4.0
    scan_r, _ = synth_scan(template, x_um_per_px=2.0, y_um_per_px=2.0,
                           origin_px=(150.0, 120.0), depth_um=26.0, rotation_deg=rot, seed=5)
    from register import CellPlacement
    pl_r = CellPlacement(1, 150.0, 120.0, 2.0, 2.0, y_up=1, rotation_deg=rot,
                         score=1.0, method="manual")
    r_dia, r_depth = [], []
    for a in template.arrays:
        s = ArraySample(filename="rot", vk4_stem="rot", cell_id=1, array_id=a.array_id,
                        band=a.band, col=a.col, passes=20, speed=400,
                        nominal_diameter_um=a.diameter_um, target_diameter_um=a.diameter_um,
                        nominal_pitch_um=a.pitch_um, nominal_pitch_x_um=a.pitch_x_um,
                        nominal_pitch_y_um=a.pitch_y_um, nx=a.nx, ny=a.ny,
                        cx_um=a.cx_um, cy_um=a.cy_um)
        r = extract_array(scan_r, pl_r, a, s, make_qc=False)
        if np.isfinite(r.diameter_um):
            r_dia.append(r.diameter_um - a.diameter_um)
        if np.isfinite(r.depth_um):
            r_depth.append(r.depth_um - 26.0)
    r_dia = np.array(r_dia); r_depth = np.array(r_depth)
    print(f"    rotated({rot}°) mid-Ø err max |{np.abs(r_dia).max():.2f}| µm, "
          f"depth err max |{np.abs(r_depth).max():.2f}| µm")
    ck.check(len(r_depth) == template.n_arrays and np.abs(r_depth).max() <= 4.0,
             "rotation: every depth within 4 µm (classification honors rotation)")
    ck.check(np.abs(r_dia).max() <= 6.0, "rotation: every mid-diameter within 6 µm")

    # -------------------------------------------------- 7. quality gate #
    print("\n[7] overstated n_cells -> spurious peaks rejected, not measured as real")
    pls_over = register_scan(scan, template, n_cells=3, cell_pitch_mm=design.cell_pitch_mm)
    n_ok = sum(np.isfinite(p.origin_col) for p in pls_over)
    print(f"    n_cells=3 on a 1-cell scan -> {n_ok} valid placement(s)")
    ck.check(n_ok == 1, "overstated n_cells: exactly one valid placement survives")

    # -------------------------------------------------- 8. degenerate DXF #
    print("\n[8] degenerate DXF (marker + <2 pins) does not crash registration")
    try:
        import ezdxf
        doc = ezdxf.new(); doc.header["$INSUNITS"] = 4
        mm = doc.modelspace()
        mm.add_lwpolyline([(0, 0), (0.2, 0), (0.2, 0.2), (0, 0.2)], close=True)
        mm.add_circle((1.0, 1.0), 0.05)                 # a single isolated pin
        degen = OUT / "degenerate.dxf"; doc.saveas(degen)
        dd = read_design(degen)
        ck.check(dd.cells[0].n_arrays == 0, "degenerate DXF -> 0 arrays (no crash on read)")
        register_scan(scan, dd.cells[0], n_cells=1)     # must not raise on empty arrays
        ck.check(True, "register_scan on empty-array template did not crash")
    except Exception as e:                              # pragma: no cover
        ck.check(False, f"degenerate DXF path raised: {e!r}")

    df.to_csv(OUT / "synth_measurements.csv", index=False)
    print(f"\nWrote synthetic measurements + plots to {OUT}")
    print(f"\n{'='*60}\n{ck.n - len(ck.fails)}/{ck.n} checks passed")
    if ck.fails:
        print("FAILURES:")
        for m in ck.fails:
            print("  -", m)
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
