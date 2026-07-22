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
from register import register_scan, register_sample
from extract import ArraySample, extract_array
from synth import synth_scan
import report as ra

HERE = Path(__file__).parent
OUT = HERE / "Results" / "legacy" / "selftest"


def _raises(fn, *args):
    """True if calling ``fn(*args)`` raises ValueError (used to assert malformed input is rejected)."""
    try:
        fn(*args)
        return False
    except ValueError:
        return True


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
    dia_err, mid_err, depth_err, pitch_err, meas_pitch_err = [], [], [], [], []
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
        if np.isfinite(res.meas_pitch_um):
            meas_pitch_err.append(res.meas_pitch_um - a.pitch_um)

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
    ck.check(len(meas_pitch_err) >= 1 and np.max(np.abs(meas_pitch_err)) <= 3.0,
             "measured (autocorrelation) pitch within 3 µm of design (QC path exercised)")
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

    # -------------------------------------------------- 9. depth-cal cell filter #
    print("\n[9] depth-calibration per-sample cell filter (parse + apply)")
    try:
        import calibrate_depth as cd
        m, ids = cd.parse_cell_spec("1-5, 8, 12-16")
        ck.check(m == "include" and ids == {1, 2, 3, 4, 5, 8, 12, 13, 14, 15, 16},
                 "parse '1-5, 8, 12-16' -> include {1..5,8,12..16}")
        m, ids = cd.parse_cell_spec("!3, 7-9")
        ck.check(m == "exclude" and ids == {3, 7, 8, 9}, "parse '!3, 7-9' -> exclude {3,7,8,9}")
        ck.check(cd.parse_cell_spec("")[1] is None and cd.parse_cell_spec(None)[1] is None,
                 "empty / None spec -> no filter")
        bad = [s for s in ("0", "5-1", "abc", "1-", "-4", "1,,x", "!")
               if not _raises(cd.parse_cell_spec, s)]
        ck.check(not bad, f"every malformed spec raises ValueError (offenders: {bad})")
        # apply on a synthetic 6-cell frame (2 rows per cell -> 12 rows)
        fdf = pd.DataFrame({"cell_id": np.repeat(np.arange(1, 7), 2),
                            "depth_um": np.arange(12, dtype=float)})
        inc = cd._apply_cell_filter(fdf, "S", "1-2, 5")
        ck.check(set(inc["cell_id"]) == {1, 2, 5} and len(inc) == 6, "include keeps only cells 1,2,5")
        exc = cd._apply_cell_filter(fdf, "S", "!3, 4")
        ck.check(set(exc["cell_id"]) == {1, 2, 5, 6} and len(exc) == 8, "exclude drops cells 3,4")
        ck.check(cd._apply_cell_filter(fdf, "S", "") is fdf, "empty spec -> frame returned unchanged")
        nocid = pd.DataFrame({"depth_um": [1.0, 2.0]})
        ck.check(cd._apply_cell_filter(nocid, "S", "1-2") is None,
                 "missing cell_id column -> filter returns None (fail-closed)")
        # end-to-end: load_pooled honours the filter and drops the un-filterable sample
        pooled = cd.load_pooled([("A", None)], cell_filters={"A": "1-2"})  # csv=None forces read error
        ck.check(len(pooled) == 0, "load_pooled skips an unreadable sample without crashing")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"cell-filter path raised: {e!r}")

    # -------------------------------------------------- 10. HIGH-severity review-fix regressions #
    print("\n[10] regression guards for the 5 HIGH review fixes")
    try:
        import assemble, extract, run_sample, tempfile, shutil
        from register import CellPlacement, _assign_grid_indices

        # H1: _estimate_step returns None (caller fails closed) when a paired axis has no correlation,
        # but the tile size when an axis has no adjacent pairs at all (single row/column).
        class _T:
            def __init__(s, a): s.height_raw = np.ones_like(a, dtype=int); s.intensity = a; s.height_um = a
        _rng = np.random.default_rng(0)
        _tiles = {(0, 0): _T(_rng.standard_normal((80, 80))), (0, 1): _T(_rng.standard_normal((80, 80)))}
        _sc, _sr = assemble._estimate_step(_tiles, [0], [0, 1], (80, 80))
        ck.check(_sc is None and _sr == 80, "H1: no-correlation x-pair -> None (fail closed); no y-pair -> tile size")

        # H2: an inverted/undefined top diameter is suppressed to NaN, not published wider than mid.
        _rc = np.arange(0, 40, 1.0)
        _cone = lambda amp: np.maximum(0.0, amp * (1 - _rc / 20.0))
        _b, _m, _t, _ = extract._diameters_from_profile(_rc, _cone(12.0), 0.0, 10.0)
        ck.check(np.isfinite(_t) and _t < _m < _b, "H2: normal pin -> finite ordered top<mid<base")
        _, _m2, _t2, _ = extract._diameters_from_profile(_rc, _cone(5.1), 0.0, 10.0)
        ck.check(np.isfinite(_m2) and np.isnan(_t2), "H2: shallow amplitude -> inverted top suppressed to NaN")

        # H3: alternating cell dropout is numbered 1,3,5 (design-pitch fallback), not compressed 1,2,3.
        def _cols(orig):
            cs = [CellPlacement(0, c, 100.0, 2.0, 2.0, y_up=1, x_right=1, score=1.0, method="marker") for c in orig]
            _assign_grid_indices(cs, 300.0, 300.0)
            return [c.cell_col for c in cs]
        ck.check(_cols([0.0, 300.0, 600.0]) == [1, 2, 3], "H3: complete row -> cols 1,2,3 (unchanged)")
        ck.check(_cols([0.0, 600.0, 1200.0]) == [1, 3, 5], "H3: alternating dropout -> cols 1,3,5 (not 1,2,3)")

        # H4: clear_output_dir refuses to wipe a directory that holds an input; clears otherwise.
        _tmp = Path(tempfile.mkdtemp())
        try:
            _vk4 = _tmp / "out" / "VK4"; _vk4.mkdir(parents=True)
            _scan = _vk4 / "raw_Y0_X0.vk4"; _scan.write_text("precious", encoding="utf-8")
            _refused = False
            try:
                run_sample.clear_output_dir(_tmp / "out", protect=(_vk4,))
            except SystemExit:
                _refused = True
            ck.check(_refused and _scan.exists(), "H4: out_dir holding the VK4 input -> refused, scan preserved")
            _o2 = _tmp / "out2"; _o2.mkdir(); (_o2 / "stale.png").write_text("old", encoding="utf-8")
            run_sample.clear_output_dir(_o2, protect=(_tmp / "elsewhere",))
            ck.check(not (_o2 / "stale.png").exists(), "H4: inputs elsewhere -> stale output cleared")
        finally:
            shutil.rmtree(_tmp, ignore_errors=True)
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"HIGH-fix regression path raised: {e!r}")

    # -------------------------------------------------- 11. tapered pins + debris floor (#18) #
    print("\n[11] tapered-pin ordering + debris-floor depth recovery")
    try:
        def _measure(scan_, pl_, dtruth):
            deps, pairs = [], []
            for a in template.arrays:
                ss = ArraySample(filename="cov", vk4_stem="cov", cell_id=1, array_id=a.array_id,
                                 band=a.band, col=a.col, passes=20, speed=400,
                                 nominal_diameter_um=a.diameter_um, target_diameter_um=a.diameter_um,
                                 nominal_pitch_um=a.pitch_um, nominal_pitch_x_um=a.pitch_x_um,
                                 nominal_pitch_y_um=a.pitch_y_um, nx=a.nx, ny=a.ny,
                                 cx_um=a.cx_um, cy_um=a.cy_um)
                rr = extract_array(scan_, pl_, a, ss, make_qc=False)
                if np.isfinite(rr.depth_um):
                    deps.append(rr.depth_um - dtruth)
                if np.isfinite(rr.base_diameter_um) and np.isfinite(rr.top_diameter_um):
                    pairs.append((rr.base_diameter_um, rr.top_diameter_um))
            return np.array(deps), pairs
        # tapered pins: top must be measured no wider than base (H2 ordering), depth still recovered
        sc_t, _ = synth_scan(template, x_um_per_px=xppx, y_um_per_px=yppx, origin_px=(130.0, 100.0),
                             depth_um=30.0, taper_frac=0.30, seed=11)
        pl_t = register_scan(sc_t, template, n_cells=1, cell_pitch_mm=design.cell_pitch_mm)[0]
        de_t, pairs_t = _measure(sc_t, pl_t, 30.0)
        ck.check(len(de_t) == template.n_arrays and np.abs(de_t).max() <= 4.0,
                 "taper: every depth within 4 µm of truth")
        ck.check(bool(pairs_t) and all(b + 2.0 >= t for b, t in pairs_t),
                 "taper: top diameter not wider than base for any array (no inverted taper)")
        # debris on the floor: depth still recovered from the clean-floor classification
        sc_d, _ = synth_scan(template, x_um_per_px=xppx, y_um_per_px=yppx, origin_px=(130.0, 100.0),
                             depth_um=30.0, debris_frac=0.35, seed=12)
        pl_d = register_scan(sc_d, template, n_cells=1, cell_pitch_mm=design.cell_pitch_mm)[0]
        de_d, _ = _measure(sc_d, pl_d, 30.0)
        ck.check(len(de_d) >= 1 and np.abs(de_d).max() <= 6.0,
                 "debris: depth recovered within 6 µm despite floor debris")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"taper/debris path raised: {e!r}")

    # -------------------------------------------------- 12. dense-array floor leveling (#10) #
    print("\n[12] geometry-aware floor leveling is unbiased by sidewalls on a dense array")
    try:
        from extract import _level_floor as _lf
        _r10 = np.random.default_rng(0)
        _H = _W = 160
        _yy, _xx = np.mgrid[0:_H, 0:_W]
        _z = 0.030 * _xx - 0.020 * _yy + _r10.normal(0, 0.25, (_H, _W))   # tilted, noisy clean floor
        _cen = []
        for _cy in range(10, _H, 20):                                     # dense pin grid (~94% cover)
            for _cx in range(10, _W, 20):
                _cen.append((_cx, _cy))
                _z[((_xx - _cx) ** 2 + (_yy - _cy) ** 2) <= 81] = 10.0 + 0.030 * _cx - 0.020 * _cy
        _cen = np.array(_cen, float); _val = np.ones((_H, _W), bool)
        _rr = np.sqrt(((_xx.ravel()[:, None] - _cen[:, 0]) ** 2
                       + (_yy.ravel()[:, None] - _cen[:, 1]) ** 2)).min(axis=1).reshape(_H, _W)
        _floor = _rr >= 12.0                                              # true clean-floor pixels
        _old = float(np.std(_lf(_z, _val)[_floor]))                       # bottom-40% (sidewall-biased)
        _new = float(np.std(_lf(_z, _val, _cen, 1.0, 1.0, 9.0)[_floor]))  # geometry-aware
        ck.check(_new < 0.4 and _new < 0.5 * _old,
                 f"#10: geometry leveling flattens the dense floor (old {_old:.2f} -> new {_new:.2f})")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"floor-leveling path raised: {e!r}")

    # -------------------------------------------------- 13. flip resolution (#17) #
    print("\n[13] register recovers a y-flip AND x-mirror (not just the identity case)")
    try:
        _allc = template.all_centers_um()
        _oc = _allc[:, 0].max() / xppx + 80.0            # place the doubly-flipped cell fully on-canvas
        _or = _allc[:, 1].max() / yppx + 80.0
        scan_f, _ = synth_scan(template, x_um_per_px=xppx, y_um_per_px=yppx, origin_px=(_oc, _or),
                               depth_um=28.0, y_up=-1, x_right=-1, seed=13)
        pl_f = register_scan(scan_f, template, n_cells=1, cell_pitch_mm=design.cell_pitch_mm)[0]
        ck.check(pl_f.y_up == -1 and pl_f.x_right == -1, "flip: register resolves y_up=-1 AND x_right=-1")
        ck.check(abs(pl_f.origin_col - _oc) <= 6 and abs(pl_f.origin_row - _or) <= 6,
                 "flip: recovered origin within 6 px of truth")
        _fm, _fd = [], []
        for a in template.arrays:
            _ss = ArraySample(filename="f", vk4_stem="f", cell_id=1, array_id=a.array_id, band=a.band,
                              col=a.col, passes=20, speed=400, nominal_diameter_um=a.diameter_um,
                              target_diameter_um=a.diameter_um, nominal_pitch_um=a.pitch_um,
                              nominal_pitch_x_um=a.pitch_x_um, nominal_pitch_y_um=a.pitch_y_um,
                              nx=a.nx, ny=a.ny, cx_um=a.cx_um, cy_um=a.cy_um)
            _rr = extract_array(scan_f, pl_f, a, _ss, make_qc=False)
            if np.isfinite(_rr.diameter_um): _fm.append(_rr.diameter_um - a.diameter_um)
            if np.isfinite(_rr.depth_um): _fd.append(_rr.depth_um - 28.0)
        ck.check(len(_fm) == template.n_arrays and np.abs(_fm).max() <= 5.0,
                 "flip: every mid-diameter within 5 µm under the double flip")
        ck.check(bool(_fd) and np.abs(_fd).max() <= 4.0, "flip: every depth within 4 µm under the double flip")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"flip path raised: {e!r}")

    # ---------------------------------------- 14. L-fiducial detection + tiled-grid recovery (#20, #16) #
    print("\n[14] L-marker detection + register_sample tiled-grid recovery (production path)")
    try:
        _cands = list((HERE / "DXF").glob("*.dxf"))
        if (HERE / "Results").is_dir():
            _cands += list((HERE / "Results").glob("*/figures/*.dxf"))
        _Ltmpl = None
        for _p in _cands:
            try:
                _t = read_design(_p).cells[0]
                if getattr(_t, "marker_shape", "") == "L" and getattr(_t, "marker_polygon_um", None) is not None:
                    _Ltmpl = _t
                    break
            except Exception:
                continue
        if _Ltmpl is None:
            print("    (no L-marker DXF available -> skipping; the L path is exercised when one is present)")
        else:
            _gap = 300.0
            scan_L, _trL = synth_scan(_Ltmpl, x_um_per_px=2.0, y_um_per_px=2.0, origin_px=(110.0, 90.0),
                                      n_cells=3, cell_gap_um=_gap, depth_um=30.0, floor_um=60.0, seed=21)
            _allcL = _Ltmpl.all_centers_um()
            _pitch = (_allcL[:, 0].max() + _gap, _allcL[:, 1].max() + _gap)
            pls_L = register_sample(scan_L, _Ltmpl, cell_pitch_um=_pitch, mirror_x=False)
            ck.check(len(pls_L) == 3, "L: three tiled cells detected via register_sample")
            ck.check(bool(pls_L) and all(p.method == "marker" for p in pls_L),
                     "L: located via the marker branch (L polygon), no square fallback")
            _errs = [min(max(abs(p.origin_col - a), abs(p.origin_row - b)) for a, b in _trL["origins"])
                     for p in pls_L]
            ck.check(bool(_errs) and max(_errs) <= 6.0, "L: every recovered origin within 6 px of truth")
            # #16: the PRODUCTION register_sample path recovers the design-frame grid indices
            # (row-major: 1=left; a single tiled row -> cols 1,2,3, all row 1) and orientation.
            ck.check(sorted(p.cell_col for p in pls_L) == [1, 2, 3] and all(p.cell_row == 1 for p in pls_L),
                     "#16: register_sample recovers the tiled cell grid (cols 1,2,3, single row)")
            ck.check(all(p.x_right == 1 and p.y_up == 1 for p in pls_L),
                     "#16: register_sample reports the expected orientation (x_right=+1, y_up=+1)")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"L-marker path raised: {e!r}")

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
