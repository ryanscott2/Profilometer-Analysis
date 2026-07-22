"""
End-to-end self-test on synthetic data (no real VK4 files required).

Validates the v2 pipeline that cannot be exercised on the empty ``VK4/`` folder yet:

  1. Registration recovers a known cell origin (single cell and a tiled 2-cell scan).
  2. Extraction recovers the known diameter / pitch / depth for every array.
  3. The full plot suite renders from a synthetic measurements table.
  4. Real markerless fabrication DXFs retain their geometry and resolve only with finite-edge proof.

Run:  python selftest.py       (writes previews/plots under Results/selftest/)
Exit code is non-zero if any check fails, so it can gate CI.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dxf_geometry import read_design
from register import (RegistrationAmbiguityError, _dealias_origin, _overlap_score,
                      _register_by_pattern, rasterize_cell_pins, register_scan,
                      register_sample, scan_feature)
from extract import ArraySample, extract_array
from synth import SynthScan, synth_scan
import report as ra

HERE = Path(__file__).parent
OUT = HERE / "Results" / "legacy" / "selftest"


def _dxf_candidates(name: str, fixture_group: str) -> list[Path]:
    """Return read-only DXF locations in local-first, portable-fallback order."""
    candidates: list[Path] = []

    # An explicit override is useful when the Stanford OneDrive folder is mounted elsewhere.
    if override := os.environ.get("PFLM_DXF_DIR"):
        candidates.append(Path(override) / name)

    # Prefer the fabrication files in the user's synced data tree.  The three standalone
    # markerless layouts currently live in the older OneDrive checkout's fixture directory.
    onedrive = os.environ.get("OneDriveCommercial") or os.environ.get("OneDrive")
    if onedrive:
        data_root = Path(onedrive) / "SU26" / "UV Laser PFLM"
        candidates.extend((
            data_root / "DXF" / name,
            data_root / "PYTHON" / "PFLM_Profilometer_Analysis"
            / "tests" / "fixtures" / fixture_group / name,
        ))

    # These paths make the same test portable to GitHub Actions and other machines.  No source
    # is ever opened for writing; the repository DXF is only a fallback test fixture.
    candidates.extend((
        HERE / "DXF" / name,
        HERE / "tests" / "fixtures" / fixture_group / name,
    ))

    # Keep error messages readable if an override happens to duplicate another location.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _resolve_dxf(name: str, fixture_group: str) -> Path:
    """Select an existing DXF without copying, normalizing, or modifying it."""
    candidates = _dxf_candidates(name, fixture_group)
    for path in candidates:
        if path.is_file():
            return path
    searched = "\n    ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {name}; searched:\n    {searched}")


def _canonical_dxf_sha256(path: Path) -> str:
    """Hash text-DXF content while treating LF and CRLF checkouts identically."""
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


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

    # Local runs use the OneDrive fabrication DXF; CI uses the versioned equivalent.
    dxf = _resolve_dxf("071826_UVPFLM_D300.dxf", "registration")
    design = read_design(dxf)
    template = design.cells[0]
    print(f"DXF source (read-only): {dxf}")
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
        ck.check(_sc is None and _sr == (0, 80),
                 "H1: no-correlation x-pair -> None (fail closed); no y-pair -> tile vector")

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

        # H4: output replacement is owned, root-confined, and transactional.
        _tmp = Path(tempfile.mkdtemp())
        try:
            _root = _tmp / "Results"; _root.mkdir()
            _unsafe = _root / "overlap"
            _vk4 = _unsafe / "VK4"; _vk4.mkdir(parents=True)
            _scan = _vk4 / "raw_Y0_X0.vk4"; _scan.write_text("precious", encoding="utf-8")
            _refused = False
            try:
                run_sample._prepare_output_transaction(
                    _unsafe, protect=(_vk4,), results_root=_root)
            except SystemExit:
                _refused = True
            ck.check(_refused and _scan.exists(), "H4: out_dir holding the VK4 input -> refused, scan preserved")

            _unowned = _root / "unowned"; _unowned.mkdir()
            (_unowned / "precious.txt").write_text("mine", encoding="utf-8")
            _refused = False
            try:
                run_sample._prepare_output_transaction(_unowned, results_root=_root)
            except SystemExit:
                _refused = True
            ck.check(_refused and (_unowned / "precious.txt").exists(),
                     "H4: arbitrary non-empty directory is never recursively replaced")

            _final = _root / "sample"; _final.mkdir()
            run_sample._write_results_sentinel(_final, _final, "complete")
            (_final / "old.txt").write_text("last-good", encoding="utf-8")
            _stage, _dest = run_sample._prepare_output_transaction(_final, results_root=_root)
            ck.check((_final / "old.txt").read_text() == "last-good",
                     "H4: previous completed result remains intact while staging")
            (_stage / "legacy").mkdir(); (_stage / "figures").mkdir()
            (_stage / "legacy" / "measurements.csv").write_text("x\n1\n", encoding="utf-8")
            (_stage / "figures" / "run_manifest.json").write_text("{}", encoding="utf-8")
            run_sample._commit_output_transaction(_stage, _dest)
            ck.check(not (_final / "old.txt").exists()
                     and (_final / "legacy" / "measurements.csv").is_file()
                     and run_sample._sentinel_valid(_final),
                     "H4: only a complete staged run atomically replaces the prior result")
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
        _cands = []
        for _fixture_name in ("071826_UVPFLM_D300.dxf", "072026_UVPFLM_D100D50.dxf"):
            try:
                _cands.append(_resolve_dxf(_fixture_name, "registration"))
            except FileNotFoundError:
                pass
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
                                      n_cells=3, cell_gap_um=_gap, depth_um=30.0, floor_um=60.0,
                                      rotation_deg=5.0, seed=21)
            _allcL = _Ltmpl.all_centers_um()
            _pitch = (_allcL[:, 0].max() + _gap, _allcL[:, 1].max() + _gap)
            pls_L = register_sample(scan_L, _Ltmpl, cell_pitch_um=_pitch, mirror_x=False)
            ck.check(len(pls_L) == 3, "L: three tiled cells detected via register_sample")
            ck.check(bool(pls_L) and all(p.method == "lattice" for p in pls_L),
                     "L: one-row marker anchors pass the shared lattice/de-alias/quality gates")
            _errs = [min(max(abs(p.origin_col - a), abs(p.origin_row - b)) for a, b in _trL["origins"])
                     for p in pls_L]
            ck.check(bool(_errs) and max(_errs) <= 2.0, "L: every +5 deg origin within 2 px of truth")
            ck.check(all(abs(p.rotation_deg - 5.0) <= 0.15 for p in pls_L),
                     "#19: register_sample recovers +5 deg within 0.15 deg")
            # #16: the PRODUCTION register_sample path recovers the design-frame grid indices
            # (row-major: 1=left; a single tiled row -> cols 1,2,3, all row 1) and orientation.
            ck.check(sorted(p.cell_col for p in pls_L) == [1, 2, 3] and all(p.cell_row == 1 for p in pls_L),
                     "#16: register_sample recovers the tiled cell grid (cols 1,2,3, single row)")
            ck.check(all(p.x_right == 1 and p.y_up == 1 for p in pls_L),
                     "#16: register_sample reports the expected orientation (x_right=+1, y_up=+1)")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"L-marker path raised: {e!r}")

    # --------------------------------------------- 15. markerless uniform-array aliasing (#2) #
    print("\n[15] real markerless DXFs: finite-edge alias elimination (#2)")
    try:
        _cases = (
            ("D50_P100_1cm2.dxf", 50.0, 100.0, 100, 10_000,
             "1d9ab5d1596b6941631507cbff841e4c3b2d65663decd43ce7b730fbfc6f5807"),
            ("D100_P150_1cm2.dxf", 100.0, 150.0, 66, 4_356,
             "5a27edf89d15583b904e1eeae53ab6bae05047a85bcc49e20a5f3157d2783d0a"),
            # The filename says D300, but the actual fabrication circle radius is 0.1475 mm.
            ("D300_P350_1cm2.dxf", 295.0, 350.0, 28, 784,
             "ed88e62f24f6d20b427bfb1247c2a2b4ae6762adf58105475e763a6b11626cc3"),
        )
        _loaded = []
        for _name, _dia, _pitch, _grid, _npins, _sha in _cases:
            _path = _resolve_dxf(_name, "markerless")
            print(f"    {_name} source (read-only): {_path}")
            _design = read_design(_path)
            _cell = _design.cells[0]
            _array = _cell.arrays[0]
            ck.check(_canonical_dxf_sha256(_path) == _sha,
                     f"#2 {_name}: fixture content is unchanged (LF/CRLF equivalent)")
            ck.check(not _design.is_unit_cell and _design.n_markers == 0
                     and len(_design.cells) == 1 and _cell.marker_polygon_um is None,
                     f"#2 {_name}: parsed as one markerless cell")
            ck.check(len(_cell.arrays) == 1 and _array.nx == _array.ny == _grid
                     and _array.n_pins == _cell.n_pins == _npins == _grid * _grid,
                     f"#2 {_name}: complete {_grid}x{_grid} single uniform grid")
            ck.check(abs(_array.diameter_um - _dia) <= 1e-6
                     and abs(_array.pitch_x_um - _pitch) <= 1e-6
                     and abs(_array.pitch_y_um - _pitch) <= 1e-6,
                     f"#2 {_name}: drawn D{_dia:g}/P{_pitch:g} geometry retained")
            ck.check(_design.boundary_bbox_mm == (0.0, 0.0, 10.0, 10.0)
                     and _cell.size_um == (10_000.0, 10_000.0),
                     f"#2 {_name}: 1 cm2 boundary retained")
            _loaded.append((_name, _cell, _array, _grid))

        _scale = 25.0
        _origin = (20.0, 20.0)
        _margin = inspect.signature(_dealias_origin).parameters["margin"].default
        ck.check(_margin == 0.02, "#2: current de-alias decision margin is 2 percentage points")
        for _case_i, (_name, _cell, _array, _grid) in enumerate(_loaded):
            _px = _array.pitch_x_um / _scale
            _py = _array.pitch_y_um / _scale
            _rad = 0.5 * _array.diameter_um / _scale
            _sx = (_array.x1_um - _array.x0_um) / _scale
            _sy = (_array.y1_um - _array.y0_um) / _scale
            _shape = (int(np.ceil(2 * _origin[1] + _sy + _py + 2 * _rad + 3)),
                      int(np.ceil(2 * _origin[0] + _sx + _px + 2 * _rad + 3)))
            _truth = rasterize_cell_pins(_cell, _scale, _scale, _shape, _origin)
            _shift_x = rasterize_cell_pins(
                _cell, _scale, _scale, _shape, (_origin[0] + _px, _origin[1]))
            _shift_xy = rasterize_cell_pins(
                _cell, _scale, _scale, _shape, (_origin[0] + _px, _origin[1] + _py))
            _score_x = _overlap_score(_truth, _shift_x)
            _score_xy = _overlap_score(_truth, _shift_xy)
            _expected_x = (_grid - 1) / _grid
            _expected_xy = _expected_x ** 2
            ck.check(abs(_score_x - _expected_x) <= 1e-12 and _score_x > 0.96,
                     f"#2 {_name}: one-pitch wrong origin still scores {_score_x:.4f}")
            ck.check(abs(_score_xy - _expected_xy) <= 1e-12 and _score_xy > 0.92,
                     f"#2 {_name}: diagonal pitch alias still scores {_score_xy:.4f}")

            _aliased = (_origin[0] + _px, _origin[1])
            _recovered, _ov = _dealias_origin(
                _truth, _cell, _aliased, _scale, _scale, y_up=1, x_right=1, rot_deg=0.0)
            _recovered_truth = (abs(_recovered[0] - _origin[0]) <= 1e-6
                                and abs(_recovered[1] - _origin[1]) <= 1e-6)
            if _grid in (100, 66):
                ck.check(1.0 / _grid < _margin,
                         f"#2 {_name}: edge evidence {1 / _grid:.4f} is below the {_margin:.2f} margin")
                ck.check(not _recovered_truth,
                         f"#2 {_name}: overlap de-alias alone is demonstrably insufficient")
            else:
                ck.check(_ov > 0.99 and _recovered_truth,
                         f"#2 {_name}: shorter ideal grid clears the margin and de-aliases")

            _refused = False
            try:
                _register_by_pattern(None, _cell, None, None)
            except RegistrationAmbiguityError as _err:
                _refused = ("finite-edge evidence" in str(_err)
                            and "alignment fiducial" in str(_err))
            ck.check(_refused,
                     f"#2 {_name}: missing scan evidence fails closed with actionable error")

            # The production marker-free path now fits lattice phase modulo pitch, enumerates every
            # finite array-index hypothesis, and uses independent edge nodes to select the absolute
            # origin.  Exercise a non-axis-aligned pose for all three actual fabrication geometries.
            _reg_scale = 10.0
            _reg_origin = (60.0, 60.0)
            _reg_angle = 1.7
            _ms, _ = synth_scan(
                _cell, x_um_per_px=_reg_scale, y_um_per_px=_reg_scale,
                origin_px=_reg_origin, marker=False, rotation_deg=_reg_angle,
                seed=820 + _case_i)
            _, _, _, _mv, _mz = scan_feature(_ms)
            _mp = _register_by_pattern(
                _ms, _cell, _mz, _mv, x_right_options=(1,), y_up_options=(1,))
            ck.check(len(_mp) == 1 and _mp[0].method == "uniform-edge",
                     f"#2 {_name}: finite-edge production path returns one placement")
            if _mp:
                ck.check(max(abs(_mp[0].origin_col - _reg_origin[0]),
                             abs(_mp[0].origin_row - _reg_origin[1])) <= 1.0,
                         f"#2 {_name}: absolute origin recovered within 1 px")
                ck.check(abs(_mp[0].rotation_deg - _reg_angle) <= 0.01,
                         f"#2 {_name}: markerless rotation recovered within 0.01 deg")

            # A single interior image has no absolute pin index, but its pitch phase is measurable.
            # Centre the visible patch in a pitch-equivalent DXF placement and prove that downstream
            # diameter/depth extraction uses only the visible, phase-aligned pins.
            _phase_full, _ = synth_scan(
                _cell, x_um_per_px=_reg_scale, y_um_per_px=_reg_scale,
                origin_px=_reg_origin, marker=False, depth_um=30.0,
                seed=850 + _case_i)
            _phase_pitch = _array.pitch_x_um / _reg_scale
            _phase_start = _grid // 5
            _phase_span = 15
            _pc0 = int(_reg_origin[0] + (_phase_start - 0.4) * _phase_pitch)
            _pc1 = int(_reg_origin[0] + (_phase_start + _phase_span + 0.4) * _phase_pitch)
            _pr0 = int(_reg_origin[1] + (_phase_start - 0.4) * _phase_pitch)
            _pr1 = int(_reg_origin[1] + (_phase_start + _phase_span + 0.4) * _phase_pitch)
            _sub = SynthScan(
                _phase_full.height_raw[_pr0:_pr1, _pc0:_pc1],
                _phase_full.intensity[_pr0:_pr1, _pc0:_pc1],
                _reg_scale, _reg_scale, _phase_full.z_um_per_digit)
            _, _, _, _sv, _sz = scan_feature(_sub)
            _sp = _register_by_pattern(
                _sub, _cell, _sz, _sv, x_right_options=(1,), y_up_options=(1,),
                allow_uniform_phase_only=True)[0]
            _true_local = (_reg_origin[0] - _pc0, _reg_origin[1] - _pr0)

            def _phase_error(got, truth, period):
                return abs((got - truth + 0.5 * period) % period - 0.5 * period)

            ck.check(_sp.method == "uniform-phase" and not _sp.absolute_origin
                     and _sp.ambiguous_axes == "xy",
                     f"#2 {_name}: interior image returns explicitly ambiguous phase-only lock")
            ck.check(max(_phase_error(_sp.origin_col, _true_local[0], _phase_pitch),
                         _phase_error(_sp.origin_row, _true_local[1], _phase_pitch)) <= 0.25,
                     f"#2 {_name}: subsection lattice phase aligned within 0.25 px")
            _ss = ArraySample(
                filename="phase-only", vk4_stem="phase-only", cell_id=1,
                array_id=_array.array_id, band=_array.band, col=_array.col,
                passes=1, speed=100.0, nominal_diameter_um=_array.diameter_um,
                target_diameter_um=_array.diameter_um, nominal_pitch_um=_array.pitch_um,
                nominal_pitch_x_um=_array.pitch_x_um, nominal_pitch_y_um=_array.pitch_y_um,
                nx=_array.nx, ny=_array.ny, cx_um=_array.cx_um, cy_um=_array.cy_um)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _sr = extract_array(_sub, _sp, _array, _ss, make_qc=False)
            ck.check(abs(_sr.diameter_um - _array.diameter_um) <= 5.0
                     and abs(_sr.depth_um - 30.0) <= 4.0
                     and 0 < _sr.n_cells < _array.n_pins
                     and not _sr.absolute_origin and _sr.ambiguous_axes == "xy",
                     f"#2 {_name}: visible subset yields accurate diameter/depth only")

        # A scan need not include the far side of the 1 cm design.  One *near* termination plus a
        # full pitch of valid floor is enough on an axis, but a periodic interior is not.  These
        # three crops distinguish the cases that the old whole-pattern overlap conflated.
        _d300 = next(_cell for _name, _cell, _array, _grid in _loaded
                     if _name.startswith("D300"))
        _crop_scale = 10.0
        _crop_origin = (60.0, 60.0)
        _crop_scan, _ = synth_scan(
            _d300, x_um_per_px=_crop_scale, y_um_per_px=_crop_scale,
            origin_px=_crop_origin, marker=False, seed=899)
        _crop_pitch = int(round(_d300.arrays[0].pitch_x_um / _crop_scale))

        def _crop(r0, r1, c0, c1):
            return SynthScan(
                _crop_scan.height_raw[r0:r1, c0:c1],
                _crop_scan.intensity[r0:r1, c0:c1],
                _crop_scan.x_um_per_px, _crop_scan.y_um_per_px,
                _crop_scan.z_um_per_digit)

        def _try_uniform(cropped):
            _, _, _, cv, cz = scan_feature(cropped)
            return _register_by_pattern(
                cropped, _d300, cz, cv, x_right_options=(1,), y_up_options=(1,))

        _corner = _crop(0, int(_crop_origin[1] + 18 * _crop_pitch),
                        0, int(_crop_origin[0] + 18 * _crop_pitch))
        _corner_p = _try_uniform(_corner)
        ck.check(len(_corner_p) == 1
                 and max(abs(_corner_p[0].origin_col - _crop_origin[0]),
                         abs(_corner_p[0].origin_row - _crop_origin[1])) <= 1.0,
                 "#2 partial corner: two finite terminations resolve both absolute indices")

        _one_edge = _crop(
            int(_crop_origin[1] + 5 * _crop_pitch),
            int(_crop_origin[1] + 20 * _crop_pitch),
            0, int(_crop_origin[0] + 18 * _crop_pitch))
        try:
            _try_uniform(_one_edge)
            _one_msg = ""
        except RegistrationAmbiguityError as _err:
            _one_msg = str(_err)
        ck.check("not identifiable along y" in _one_msg,
                 "#2 one-edge crop: resolved X but fails closed on ambiguous Y index")
        _, _, _, _oev, _oez = scan_feature(_one_edge)
        _one_phase = _register_by_pattern(
            _one_edge, _d300, _oez, _oev, x_right_options=(1,), y_up_options=(1,),
            allow_uniform_phase_only=True)[0]
        ck.check(_one_phase.method == "uniform-phase"
                 and _one_phase.ambiguous_axes == "y"
                 and abs(_one_phase.origin_col - _crop_origin[0]) <= 1.0,
                 "#2 one-edge phase mode preserves resolved X and labels only Y ambiguous")

        _interior = _crop(
            int(_crop_origin[1] + 5 * _crop_pitch),
            int(_crop_origin[1] + 20 * _crop_pitch),
            int(_crop_origin[0] + 5 * _crop_pitch),
            int(_crop_origin[0] + 20 * _crop_pitch))
        try:
            _try_uniform(_interior)
            _interior_msg = ""
        except RegistrationAmbiguityError as _err:
            _interior_msg = str(_err)
        ck.check("not identifiable along x,y" in _interior_msg,
                 "#2 interior crop: pitch-periodic X/Y indices remain explicitly ambiguous")
        _auto_phase = register_sample(_interior, _d300, mirror_x=False)
        ck.check(len(_auto_phase) == 1 and _auto_phase[0].method == "uniform-phase"
                 and not _auto_phase[0].absolute_origin,
                 "#2 register_sample automatically enables subsection phase-only mode")

        # Minimum supported one-frame geometry: 3x2 complete D300 pins (matching the number of
        # testable pins in the real 1024x768 VK4 tile after border-clipped pins are excluded).
        _small_start = 8
        _small = _crop(
            int(_crop_origin[1] + (_small_start - 0.6) * _crop_pitch),
            int(_crop_origin[1] + (_small_start + 1.6) * _crop_pitch),
            int(_crop_origin[0] + (_small_start - 0.6) * _crop_pitch),
            int(_crop_origin[0] + (_small_start + 2.6) * _crop_pitch))
        _small_p = register_sample(_small, _d300, mirror_x=False)
        ck.check(len(_small_p) == 1 and _small_p[0].method == "uniform-phase"
                 and _small_p[0].score >= 0.9,
                 "#2 one 3x2-pin image is sufficient for a high-quality phase-only lock")

        _ns_origin = (60.0, 60.0)
        _ns_scan, _ = synth_scan(
            _d300, x_um_per_px=8.0, y_um_per_px=12.0, origin_px=_ns_origin,
            marker=False, rotation_deg=-2.3, seed=922)
        _ns_p = _try_uniform(_ns_scan)
        ck.check(len(_ns_p) == 1
                 and max(abs(_ns_p[0].origin_col - _ns_origin[0]),
                         abs(_ns_p[0].origin_row - _ns_origin[1])) <= 1.0
                 and abs(_ns_p[0].rotation_deg + 2.3) <= 0.01,
                 "#2 markerless finite-edge path handles non-square pixels and rotation")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"markerless-DXF alias regression path raised: {e!r}")

    # ------------------------------------------------ 16. multi-degree rotation recovery (#19) #
    print("\n[16] real L-fiducial: register_scan + marker-free multi-degree rotation (#19)")
    try:
        _lpath = _resolve_dxf("072026_UVPFLM_D100D50.dxf", "registration")
        print(f"    registration source (read-only): {_lpath}")
        ck.check(_canonical_dxf_sha256(_lpath)
                 == "2c1c655ee9ab507a76386782eee88489e75e9d412663bd1631ea349f40aaf4ae",
                 "#19: real L-marker content is unchanged (LF/CRLF equivalent)")
        _lt = read_design(_lpath).cells[0]
        ck.check(_lt.marker_shape == "L" and _lt.marker_polygon_um is not None,
                 "#19: rotation fixture contains the real asymmetric L fiducial")
        for _angle in (-5.0, 3.0, 5.0):
            _rs, _rt = synth_scan(
                _lt, x_um_per_px=2.0, y_um_per_px=2.0, origin_px=(140.0, 140.0),
                rotation_deg=_angle, depth_um=30.0, seed=300 + int(_angle))
            _rp = register_scan(_rs, _lt, n_cells=1)[0]
            _ro = _rt["origins"][0]
            ck.check(np.isfinite(_rp.origin_col)
                     and max(abs(_rp.origin_col - _ro[0]), abs(_rp.origin_row - _ro[1])) <= 2.0,
                     f"#19: register_scan {_angle:+g} deg origin within 2 px")
            ck.check(abs(_rp.rotation_deg - _angle) <= 0.15,
                     f"#19: register_scan recovers {_angle:+g} deg within 0.15 deg")

        # The two-array pattern is aperiodic enough to register without its marker. This directly
        # exercises the broadened +/-5 deg marker-free fallback rather than the marker estimator.
        _ps, _pt = synth_scan(
            _lt, x_um_per_px=2.0, y_um_per_px=2.0, origin_px=(140.0, 140.0),
            rotation_deg=3.0, marker=False, depth_um=30.0, seed=319)
        _, _, _, _pv, _pz0 = scan_feature(_ps)
        _pp = _register_by_pattern(
            _ps, _lt, _pz0, _pv, x_right_options=(1,), y_up_options=(1,))
        ck.check(len(_pp) == 1, "#19: marker-free two-array pattern finds exactly one cell")
        if _pp:
            _po = _pt["origins"][0]
            ck.check(max(abs(_pp[0].origin_col - _po[0]), abs(_pp[0].origin_row - _po[1])) <= 2.0,
                     "#19: marker-free +3 deg origin within 2 px")
            ck.check(abs(_pp[0].rotation_deg - 3.0) <= 0.15,
                     "#19: marker-free fallback recovers +3 deg within 0.15 deg")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"multi-degree rotation regression path raised: {e!r}")

    # ------------------------------------------ 17. adversarial-review regression matrix #
    print("\n[17] adversarial-review safety, geometry, parsing, and statistics guards")
    try:
        import copy
        import tempfile
        import shutil
        import assemble
        import calibrate_depth as cd
        import laser_params
        import pflm_ui
        from dxf_geometry import validate_equivalent_cells

        # Laser labels must consume the whole cell and encode a positive integer pass count and
        # positive finite speed. A malformed nonblank grid entry must identify its exact location.
        ck.check(laser_params.parse_pxsy("P12_S400")[:2] == (12, 400.0),
                 "labels: canonical P12_S400 parses")
        _bad_labels = ("P1.5_S400", "P0_S400", "P2_S0", "P2_S400junk", "xP2_S400")
        ck.check(all(laser_params.parse_pxsy(x) is None for x in _bad_labels),
                 "labels: partial, fractional-pass, and nonpositive labels are rejected")
        _td = Path(tempfile.mkdtemp())
        try:
            _bad_csv = _td / "cell_params.csv"
            _bad_csv.write_text("P1_S100,P2_S200\nP3_S300,bad\n", encoding="utf-8")
            try:
                laser_params.load_cell_params(_bad_csv)
                _msg = ""
            except ValueError as _err:
                _msg = str(_err)
            ck.check("row 2, column 2" in _msg,
                     "labels: invalid CSV entry reports row and column")
        finally:
            shutil.rmtree(_td, ignore_errors=True)

        # Sanitized UI names are not unique identifiers; reject two names mapping to one folder.
        ck.check(pflm_ui._sample_name_collision("D100 D50", {"D100/D50": {}}) == "D100/D50",
                 "UI: colliding sanitized sample names are detected before overwrite")

        # The production one-template path must reject a heterogeneous multi-cell fabrication DXF.
        _hetero = copy.deepcopy(dgrid)
        _hetero.cells[1].arrays[0].centers_um = _hetero.cells[1].arrays[0].centers_um.copy()
        _hetero.cells[1].arrays[0].centers_um[0, 0] += 5.0
        ck.check(_raises(validate_equivalent_cells, _hetero),
                 "DXF: heterogeneous tiled cells are rejected instead of using cells[0]")
        validate_equivalent_cells(dgrid)
        ck.check(True, "DXF: equivalent tiled cells remain accepted")

        # Estimate both components of each tile-origin vector, including perpendicular stage drift.
        class _Tile:
            def __init__(self, a):
                self.height_raw = np.ones_like(a, dtype=np.uint16)
                self.intensity = a
                self.height_um = a
        _rng17 = np.random.default_rng(42)
        _global = _rng17.normal(size=(300, 300))
        _orig = {(0, 0): (20, 0), (0, 1): (120, 8),
                 (1, 0): (28, 100), (1, 1): (128, 108)}
        _tiles17 = {k: _Tile(_global[r:r + 160, c:c + 160])
                    for k, (c, r) in _orig.items()}
        _cv, _rv = assemble._estimate_step(_tiles17, [0, 1], [0, 1], (160, 160), ds=2)
        ck.check(_cv == (100, 8) and _rv == (8, 100),
                 "assembly: full 2-D column/row step vectors preserve cross-axis drift")

        # Duplicate filename coordinates must fail with both source names before a silent overwrite.
        _td = Path(tempfile.mkdtemp())
        _read0 = assemble.read_vk4
        try:
            (_td / "first_Y0_X0.vk4").write_bytes(b"1")
            (_td / "second_Y0_X0.vk4").write_bytes(b"2")
            assemble.read_vk4 = lambda _p: object()
            try:
                assemble.assemble_tiles(_td, verbose=False)
                _dupmsg = ""
            except ValueError as _err:
                _dupmsg = str(_err)
            ck.check("first_Y0_X0.vk4" in _dupmsg and "second_Y0_X0.vk4" in _dupmsg,
                     "assembly: duplicate Y/X coordinates name both conflicting files")
        finally:
            assemble.read_vk4 = _read0
            shutil.rmtree(_td, ignore_errors=True)

        # Rotated footprint bounds must remain correct with anisotropic pixels.
        _anis, _atruth = synth_scan(
            _lt, x_um_per_px=2.0, y_um_per_px=3.0, origin_px=(160.0, 130.0),
            rotation_deg=4.0, depth_um=30.0, seed=417)
        _ap = register_scan(_anis, _lt, n_cells=1)[0]
        _ao = _atruth["origins"][0]
        ck.check(max(abs(_ap.origin_col - _ao[0]), abs(_ap.origin_row - _ao[1])) <= 2.0,
                 "registration: rotated non-square-pixel origin remains within 2 px")

        # QC must be strict by default; cell-band aggregation removes array pseudo-replication.
        _legacy = pd.DataFrame({"depth_um": [10.0], "passes": [1], "speed": [100.0],
                                "dose_ratio": [0.01]})
        ck.check(_raises(cd.apply_gates, _legacy),
                 "calibration: missing reliable/debris QC fails closed by default")
        _raw = pd.DataFrame({
            "sample": ["A"] * 6, "cell_id": [1, 1, 1, 2, 2, 2], "band": [1] * 6,
            "passes": [10] * 3 + [20] * 3, "speed": [100.0] * 6,
            "dose_ratio": [0.1] * 3 + [0.2] * 3,
            "depth_um": [9.0, 10.0, 11.0, 19.0, 20.0, 21.0],
            "drawn_diameter_um": [100.0] * 6, "nominal_pitch_um": [150.0] * 6,
            "reliable": [True] * 6, "debris_fraction": [0.1] * 6,
        })
        _gated, _ = cd.apply_gates(_raw)
        _units = cd.aggregate_cell_bands(_gated)
        ck.check(len(_units) == 2 and set(_units["array_rows"]) == {3}
                 and set(_units["depth_um"]) == {10.0, 20.0},
                 "calibration: six array rows collapse to two independent cell-band medians")
        _other_pitch = _units.iloc[[0]].copy()
        _other_pitch["sample"] = "B"; _other_pitch["cell_id"] = 1
        _other_pitch["nominal_pitch_um"] = 350.0
        _family_keys = {key for key, _g in cd._band_groups(
            pd.concat([_units, _other_pitch], ignore_index=True), band_meta=None)}
        ck.check(_family_keys == {(1, 150.0), (1, 350.0)},
                 "calibration: same local band number at different pitches remains separate")

        _best, _ = cd.choose_recommended(
            {"ok": True, "aicc": 12.0, "r2": 0.5},
            {"ok": True, "aicc": 8.0, "adj_r2": 0.4},
            {"ok": True, "aicc": 10.0, "adj_r2": 0.6})
        ck.check(_best == "log-dose", "calibration: model selection uses lowest AICc, not in-sample R²")
        _none, _why = cd.choose_recommended(
            {"ok": True, "aicc": 1.0, "r2": 0.05},
            {"ok": True, "aicc": 2.0, "adj_r2": 0.05}, {"ok": False})
        ck.check(_none is None and "suppressed" in _why,
                 "calibration: poor fits suppress inverse recommendations")
        _satfit = {"ok": True, "a": 100.0, "k": 2.0,
                   "beta": np.array([100.0, 2.0]), "cov": np.diag([1.0, 0.01])}
        _, _lo, _hi, _note = cd.invert_target("saturating", _satfit, {}, {}, 50.0)
        ck.check(np.isfinite(_lo) and np.isfinite(_hi) and "CI" in _note and " PI " not in _note,
                 "calibration: parameter-only inverse interval is labeled CI, not prediction interval")

        _dirty = pd.DataFrame({"reliable": [True, True], "flags": ["", "wide-D"],
                               "top_diameter_um": [100.0, 180.0],
                               "diameter_um": [105.0, 190.0],
                               "drawn_diameter_um": [100.0, 100.0],
                               "passes": [10, 10], "speed": [100.0, 100.0]})
        ck.check(len(ra._fit_subset(_dirty)) == 1
                 and "_fit_subset" in inspect.getsource(ra.make_diameter_model),
                 "diameter model: debris-widened wide-D rows use the clean fit subset")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"adversarial-review regression path raised: {e!r}")

    df.to_csv(OUT / "synth_measurements.csv", index=False)
    print(f"\nWrote synthetic measurements + plots to {OUT}")
    print(f"\n{'='*60}\n{ck.n - len(ck.fails)}/{ck.n} checks passed")
    if ck.fails:
        print("FAILURES:")
        for m in ck.fails:
            print("  -", m)
        raise SystemExit(1)
    print("ALL REQUIRED CHECKS PASSED")


if __name__ == "__main__":
    main()
