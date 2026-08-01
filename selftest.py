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

import accel
from dxf_geometry import read_design
from register import (RegistrationAmbiguityError, _dealias_origin, _lattice_is_oblique,
                      _overlap_score, _register_by_pattern, rasterize_cell_pins, register_scan,
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
    accel.set_force_cpu(True)   # the gate must be deterministic: pin the CPU float64 NCC path
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
        figs = ["overview_3x3", "dose_collapse", "per_row", "diameter_fit",
                "depth_vs_dose", "grid_overlays"]
        for name in figs:
            ck.check((OUT / "figures" / f"{name}.png").exists(), f"{name}.png written")
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

        # passes×speed (interaction) is preferred whenever it fits with meaningful signal, even when a
        # dose form has a lower AICc -- depth does not collapse to dose = passes/speed.
        _best, _ = cd.choose_recommended(
            {"ok": True, "aicc": 8.0, "r2": 0.9},
            {"ok": True, "aicc": 6.0, "adj_r2": 0.9},
            {"ok": True, "aicc": 10.0, "adj_r2": 0.6})
        ck.check(_best == "interaction", "calibration: passes×speed model preferred over dose forms")
        # falls back to the lowest-AICc dose form when the interaction model is not estimable
        _fb, _ = cd.choose_recommended(
            {"ok": True, "aicc": 12.0, "r2": 0.5},
            {"ok": True, "aicc": 8.0, "adj_r2": 0.4},
            {"ok": False})
        ck.check(_fb == "log-dose", "calibration: fallback to lowest-AICc dose form when no interaction fit")
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

    # ------------------------------------------ 18. Phase-1 perf refactors are bit-exact (perf) #
    print("\n[18] Phase-1 optimizations reproduce the reference output bit-for-bit")
    try:
        def _rasterize_ref(cell, xppx, yppx, shape, origin, y_up=1, x_right=1, rot_deg=0.0):
            """The pre-optimization nested-loop rasteriser, kept here as the equivalence oracle."""
            m = np.zeros(shape, bool); H, W = shape; oc, orow = origin
            th = np.deg2rad(rot_deg); c, s = np.cos(th), np.sin(th)
            for a in cell.arrays:
                rx = 0.5 * a.diameter_um / xppx; ry = 0.5 * a.diameter_um / yppx
                rrx, rry = int(np.ceil(rx)), int(np.ceil(ry))
                for (x_um, y_um) in a.centers_um:
                    xr, yr = x_right * x_um, y_up * y_um
                    xr, yr = c * xr - s * yr, s * xr + c * yr
                    ci = int(round(oc + xr / xppx)); ri = int(round(orow + yr / yppx))
                    for dr in range(-rry, rry + 1):
                        for dc in range(-rrx, rrx + 1):
                            if (dc / rx) ** 2 + (dr / ry) ** 2 <= 1.0:
                                yy, xx = ri + dr, ci + dc
                                if 0 <= yy < H and 0 <= xx < W:
                                    m[yy, xx] = True
            return m

        # Cover square + non-square pixels, rotation, exact-half origins (tie-to-even rounding),
        # and negative origins that clip many pins off-canvas -- the edge cases where a naive
        # vectorisation (e.g. truncating round, np.roll) would diverge from the loop.
        _shape18 = (160, 200)
        _cfgs18 = [(24.0, 24.0, 0.0, (30.0, 30.0)),
                   (24.0, 20.0, 0.0, (25.0, 40.0)),
                   (24.0, 24.0, 3.7, (30.5, 30.5)),
                   (24.0, 24.0, -2.3, (-12.0, 18.0)),
                   (28.0, 21.0, 4.9, (-40.5, -8.5))]
        _rast_ok = all(
            np.array_equal(
                rasterize_cell_pins(template, _sx, _sy, _shape18, _org, rot_deg=_rot),
                _rasterize_ref(template, _sx, _sy, _shape18, _org, rot_deg=_rot))
            for (_sx, _sy, _rot, _org) in _cfgs18)
        ck.check(_rast_ok,
                 "#P1 rasterize_cell_pins vectorised == loop (square/non-square/rot/half/off-canvas)")

        # share-rr: classification must be byte-identical whether rr is recomputed or shared.
        from extract import _classify_floor_depth, _level_floor, _nearest_pin_um

        def _eqnan(x, y):
            return np.array_equal(np.asarray(x, float), np.asarray(y, float), equal_nan=True)

        _r18 = np.random.default_rng(1)
        _H18 = _W18 = 120
        _yy18, _xx18 = np.mgrid[0:_H18, 0:_W18]
        _z18 = 0.02 * _xx18 - 0.015 * _yy18 + _r18.normal(0, 0.2, (_H18, _W18))
        _cen18 = []
        for _cy in range(12, _H18, 24):
            for _cx in range(12, _W18, 24):
                _cen18.append((_cx, _cy))
                _z18[((_xx18 - _cx) ** 2 + (_yy18 - _cy) ** 2) <= 64] = 8.0
        _cen18 = np.array(_cen18, float); _val18 = np.ones((_H18, _W18), bool)
        _pxu18 = _pyu18 = 1.5
        _shared18 = _nearest_pin_um(_z18.shape, _cen18, _pxu18, _pyu18)
        _clsA = _classify_floor_depth(_z18, _val18, _cen18, _pxu18, _pyu18, 12.0, 24.0 * _pxu18)
        _clsB = _classify_floor_depth(_z18, _val18, _cen18, _pxu18, _pyu18, 12.0, 24.0 * _pxu18,
                                      rr=_shared18)
        ck.check(np.isfinite(_clsA["depth"]),
                 "#P1 share-rr test actually exercises a successful classification (finite depth)")
        ck.check(_eqnan(_clsA["rr"], _clsB["rr"])
                 and np.array_equal(_clsA["pin_mask"], _clsB["pin_mask"])
                 and np.array_equal(_clsA["floor_region"], _clsB["floor_region"])
                 and _eqnan(_clsA["depth"], _clsB["depth"])
                 and _eqnan(_clsA["clean_floor"], _clsB["clean_floor"])
                 and _eqnan(_clsA["pin_top"], _clsB["pin_top"]),
                 "#P1 _classify_floor_depth byte-identical with shared vs recomputed rr")

        # Frozen oracle of the ORIGINAL inline nearest-pin formula so a FUTURE edit to
        # _nearest_pin_um that diverges from the pre-refactor code is caught (the check above is
        # self-referential -- both branches call the new helper).  _classify used bare
        # centers*array; _level_floor used np.asarray(centers,float)*array -- exercise BOTH by
        # feeding an ndarray and a plain Python list.
        from scipy.spatial import cKDTree as _ckd

        def _rr_oracle(shape, centers, pxu, pyu, wrap):
            _H, _W = shape
            _yo, _xo = np.mgrid[0:_H, 0:_W]
            _cen = (np.asarray(centers, float) if wrap else centers) * np.array([pxu, pyu])
            return _ckd(_cen).query(
                np.column_stack([(_xo * pxu).ravel(), (_yo * pyu).ravel()]))[0].reshape(_H, _W)

        _rr_nd = _nearest_pin_um(_z18.shape, _cen18, _pxu18, _pyu18)
        _rr_li = _nearest_pin_um(_z18.shape, _cen18.tolist(), _pxu18, _pyu18)
        ck.check(_eqnan(_rr_nd, _rr_oracle(_z18.shape, _cen18, _pxu18, _pyu18, wrap=False))
                 and _eqnan(_rr_li, _rr_oracle(_z18.shape, _cen18.tolist(), _pxu18, _pyu18, wrap=True)),
                 "#P1 _nearest_pin_um matches the original inline cKDTree formula (ndarray + list)")

        # _level_floor's rr path shapes the leveled z0 (hence depth): byte-check it directly.
        _rpin18 = float(min(0.5 * 12.0, 0.46 * 24.0 * _pxu18))
        _lfA = _level_floor(_z18, _val18, _cen18, _pxu18, _pyu18, _rpin18)
        _lfB = _level_floor(_z18, _val18, _cen18, _pxu18, _pyu18, _rpin18, rr=_shared18)
        ck.check(_eqnan(_lfA, _lfB),
                 "#P1 _level_floor byte-identical with shared vs recomputed rr")

        # The rr= kwarg must actually be CONSUMED (not silently ignored): a deliberately wrong rr
        # must change both classification and leveling.
        _rr_bad = _shared18 + 1000.0
        _clsW = _classify_floor_depth(_z18, _val18, _cen18, _pxu18, _pyu18, 12.0, 24.0 * _pxu18,
                                      rr=_rr_bad)
        _lfW = _level_floor(_z18, _val18, _cen18, _pxu18, _pyu18, _rpin18, rr=_rr_bad)
        ck.check(not np.array_equal(_clsW["floor_region"], _clsA["floor_region"])
                 and not _eqnan(_lfA, _lfW),
                 "#P1 rr= kwarg is consumed by _classify_floor_depth and _level_floor")

        # Exact-half pixel centres (tie-to-even): odd-um centres at 2 um/px land every centre on
        # X.5, so this genuinely exercises np.rint's round-half-to-even against int(round()).
        import types
        _halfcell = types.SimpleNamespace(arrays=[types.SimpleNamespace(
            diameter_um=6.0,
            centers_um=np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 1.0], [1.0, 5.0], [7.0, 7.0]],
                                float))])
        _cpx = _halfcell.arrays[0].centers_um[:, 0] / 2.0
        _has_half = bool(np.any(np.abs(_cpx - np.rint(_cpx)) == 0.5))
        _half_ok = np.array_equal(rasterize_cell_pins(_halfcell, 2.0, 2.0, (24, 24), (0.0, 0.0)),
                                  _rasterize_ref(_halfcell, 2.0, 2.0, (24, 24), (0.0, 0.0)))
        ck.check(_has_half and _half_ok,
                 "#P1 rasterize matches loop on exact-half pixel centres (tie-to-even actually hit)")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"Phase-1 bit-exactness path raised: {e!r}")

    # ---------------------------------------- 19. accel FFT/NCC backend equivalence (Phase 2) #
    print("\n[19] accel backend: CPU byte-identical, decisive->CPU, GPU/pyfftw agree")
    try:
        def _ncc_ref(img, tpl):     # frozen copy of the historical register._pattern_ncc
            from scipy.signal import fftconvolve
            t = tpl.astype(np.float64); t0 = t - t.mean()
            tnorm = float(np.sqrt((t0 * t0).sum())); ones = np.ones_like(t)
            if tnorm < 1e-9:
                Hi, Wi = img.shape; Ht, Wt = t.shape
                return np.zeros((Hi + Ht - 1, Wi + Wt - 1))
            num = fftconvolve(img, t0[::-1, ::-1], mode="full")
            s1 = fftconvolve(img, ones[::-1, ::-1], mode="full")
            s2 = fftconvolve(img * img, ones[::-1, ::-1], mode="full")
            n = float(t.size)
            denom = np.sqrt(np.maximum(s2 - s1 * s1 / n, 0.0)) * tnorm
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(denom > 1e-9, num / denom, 0.0)

        _r19 = np.random.default_rng(7)
        _img19 = 0.1 * _r19.random((220, 260))
        _tpl19 = _r19.random((28, 32))
        _img19[60:88, 90:122] += 3.0 * _tpl19          # plant an unambiguous peak (stable argmax)
        _ref19 = _ncc_ref(_img19, _tpl19)

        def _argmax2d(a):
            return np.unravel_index(int(np.argmax(a)), a.shape)

        ck.check(np.array_equal(accel._ncc_scipy(_img19, _tpl19), _ref19),
                 "#P2 accel CPU (scipy) path is byte-identical to the historical _pattern_ncc")
        with accel.force_cpu():
            ck.check(accel.select_backend(decisive=False) == "scipy"
                     and np.array_equal(accel.pattern_ncc(_img19, _tpl19), _ref19),
                     "#P2 force_cpu pins the deterministic scipy path")
        ck.check(accel.select_backend(decisive=True) == "scipy",
                 "#P2 decisive NCC calls always resolve to CPU float64")

        if accel._have_pyfftw():
            _pf19 = accel._ncc_pyfftw(_img19, _tpl19)
            ck.check(float(np.max(np.abs(_pf19 - _ref19))) < 1e-8
                     and _argmax2d(_pf19) == _argmax2d(_ref19),
                     "#P2 pyfftw NCC agrees with scipy (<1e-8) and gives the same peak")
        else:
            ck.check(True, "#P2 pyfftw not installed (agreement check skipped)")

        if accel._cupy() is not None:
            _cu19 = accel._ncc_cupy(_img19, _tpl19)
            ck.check(float(np.max(np.abs(_cu19 - _ref19))) < 5e-3
                     and _argmax2d(_cu19) == _argmax2d(_ref19),
                     "#P2 cupy GPU NCC agrees with scipy (float32 tol) and same peak")
        else:
            ck.check(True, "#P2 cupy/GPU not available (agreement check skipped)")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"accel backend path raised: {e!r}")

    # ---------------------------------- 20. GPU opt-in cannot change register output (Phase 2b) #
    print("\n[20] opting into the GPU backend does not change register output (skipped w/o a GPU)")
    try:
        if accel._cupy() is None:
            ck.check(True, "#P2b GPU not available -- GPU-insulation check skipped")
        else:
            _lt20 = read_design(
                _resolve_dxf("072026_UVPFLM_D100D50.dxf", "registration")).cells[0]
            _rs20, _rt20 = synth_scan(_lt20, x_um_per_px=2.0, y_um_per_px=2.0,
                                      origin_px=(140.0, 140.0), rotation_deg=3.0,
                                      depth_um=30.0, seed=2020)

            def _reg20(gpu):                         # gpu=True: opt into the GPU backend for this run
                _prevf = accel._FORCE_CPU
                _preve = os.environ.get("PFLM_ACCEL")
                accel.set_force_cpu(not gpu)
                if gpu:
                    os.environ["PFLM_ACCEL"] = "cupy"
                try:
                    return register_scan(_rs20, _lt20, n_cells=1)[0]
                finally:
                    accel.set_force_cpu(_prevf)
                    if _preve is None:
                        os.environ.pop("PFLM_ACCEL", None)
                    else:
                        os.environ["PFLM_ACCEL"] = _preve

            _pc = _reg20(False)                      # deterministic CPU float64
            _pg = _reg20(True)                       # GPU opted in -- must be byte-identical because
            #                                          every register NCC is decisive -> CPU float64
            ck.check(_pc.x_right == _pg.x_right and _pc.y_up == _pg.y_up
                     and _pc.method == _pg.method
                     and _pc.origin_col == _pg.origin_col and _pc.origin_row == _pg.origin_row
                     and _pc.rotation_deg == _pg.rotation_deg,
                     "#P2b register output byte-identical with GPU opted in (decisive NCCs insulate "
                     "registration from float32 GPU)")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"GPU-insulation path raised: {e!r}")

    # ------------------------------------------- 21. CPU parallelism == serial (Phase 2c) #
    print("\n[21] parallel per-array extraction is identical to serial (Phase 2c)")
    try:
        import parallel
        import run_sample

        _pe21 = os.environ.pop("PFLM_JOBS", None)             # default-on: all cores when unset
        try:
            _defon21 = parallel.resolve_jobs(None) == (os.cpu_count() or 1)
        finally:
            if _pe21 is not None:
                os.environ["PFLM_JOBS"] = _pe21
        ck.check(_defon21 and parallel.resolve_jobs("1") == 1 and parallel.resolve_jobs("3") == 3
                 and parallel.resolve_jobs(0) >= 1 and parallel.resolve_jobs("bad") == 1,
                 "#P2c resolve_jobs: default-on (all cores), explicit honored, junk -> serial")

        def _res_eq(r1, r2):
            d1, d2 = r1.__dict__, r2.__dict__
            if d1.keys() != d2.keys():
                return False
            for k in d1:
                v1, v2 = d1[k], d2[k]
                try:
                    if np.array_equal(np.asarray(v1, float), np.asarray(v2, float), equal_nan=True):
                        continue
                except (TypeError, ValueError):
                    pass
                if v1 != v2:
                    return False
            return True

        # Use the 2-array L cell (not the 1-array base template) so jobs=2 truly SPAWNS workers
        # (pmap_shared short-circuits to serial when there is <=1 item).
        _lt21 = read_design(
            _resolve_dxf("072026_UVPFLM_D100D50.dxf", "registration")).cells[0]
        _rs21, _rt21 = synth_scan(_lt21, x_um_per_px=2.0, y_um_per_px=2.0,
                                  origin_px=(140.0, 140.0), rotation_deg=0.0, depth_um=30.0, seed=77)
        _pl21 = register_scan(_rs21, _lt21, n_cells=1)[0]
        _bt21 = ra._band_targets(_lt21)
        _work21 = []
        for a in _lt21.arrays:
            _s21 = ArraySample(
                filename=f"p2c_b{a.band}c{a.col}_D{a.diameter_um:g}", vk4_stem="p2c", cell_id=1,
                array_id=a.array_id, band=a.band, col=a.col, passes=20, speed=400.0,
                nominal_diameter_um=a.diameter_um, target_diameter_um=_bt21[a.band],
                nominal_pitch_um=a.pitch_um, nominal_pitch_x_um=a.pitch_x_um,
                nominal_pitch_y_um=a.pitch_y_um, nx=a.nx, ny=a.ny, cx_um=a.cx_um, cy_um=a.cy_um)
            _work21.append((_pl21, a, _s21, None, False))    # (pl, array, sample, qc_path, make_qc)

        _serial21 = parallel.pmap_shared(run_sample._extract_worker, _work21, _rs21, jobs=1)
        _par21 = parallel.pmap_shared(run_sample._extract_worker, _work21, _rs21, jobs=2)
        ck.check(len(_serial21) == len(_par21) == len(_work21) and len(_work21) >= 2,
                 f"#P2c pmap_shared spawns for >1 item, one result each ({len(_par21)} arrays)")
        ck.check(all(_res_eq(_a, _b) for _a, _b in zip(_serial21, _par21)),
                 "#P2c parallel (jobs=2) per-array extraction is field-identical to serial (jobs=1)")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"CPU parallelism path raised: {e!r}")

    # ------------------------------- 22. geom-edge-margin on a wide grid (Phase 3, #2) #
    print("\n[22] geom-edge-margin resolves a wide grid the old area threshold would reject")
    try:
        from types import SimpleNamespace
        from register import _register_uniform_lattice, _edge_margin_threshold
        from synth import _stamp_disk

        # Unit-lock the min-rule: O(N) edge term fixes large grids; min with the old area rule keeps
        # the bar never stricter than before (no recall regression on narrow arrays); floor at 5.
        _th = _edge_margin_threshold
        ck.check(_th(3630, 30) == 15.0 and 0.01 * 3630 > 30,
                 "#P3 large one-edge: geom margin (15) accepts what the old area rule (~36) rejected")
        ck.check(_th(324, 18) == 5.0,
                 "#P3 narrow array: min(area,geom) holds at the floor 5 (no recall loss vs old)")
        ck.check(_th(0, 0) == 5.0 and _th(10000, 200) == 100.0,
                 "#P3 edge-margin threshold: floor 5; large both-edge grid uses the O(N) edge term")

        # A 150-col x 30-row uniform grid, rendered so the NEAR u-edge (col 0) + its floor are in
        # frame but the far u-edge is cropped out (only 121 columns visible); both v-edges are in
        # frame.  The u-axis then has ONE captured termination (~NY=30 discriminating nodes) over a
        # wide interior (matched ~ 121*30 = 3630).  The retired threshold min_raw = 0.01*matched
        # (~36) exceeds that one-edge evidence, so the OLD code would have marked u ambiguous; the
        # geom-edge-margin (>= 0.5 * the geometrically expected edge nodes) accepts it.
        _P, _Dpx, _NX, _NY, _cols = 6.0, 3.0, 150, 30, 121
        _mL = _mT = 12
        _Wv = _mL + (_cols - 1) * int(_P) + 3            # frame ends just past the last visible column
        _Hv = _mT + (_NY - 1) * int(_P) + _mT
        _z0v = np.zeros((_Hv, _Wv), float)
        _valv = np.ones((_Hv, _Wv), bool)
        for _j in range(_cols):
            for _i in range(_NY):
                _stamp_disk(_z0v, _mL + _j * _P, _mT + _i * _P, 0.5 * _Dpx, 5.0, "set")
        _scanv = SimpleNamespace(x_um_per_px=1.0, y_um_per_px=1.0)
        _tmplv = SimpleNamespace(arrays=[SimpleNamespace(
            pitch_x_um=_P, pitch_y_um=_P, nx=_NX, ny=_NY, diameter_um=_Dpx)])
        _plv = _register_uniform_lattice(_scanv, _tmplv, _z0v, _valv,
                                         x_right_options=(1,), y_up_options=(1,),
                                         angles_deg=(0.0,), allow_phase_only=True)[0]
        ck.check(_plv.method == "uniform-edge" and _plv.absolute_origin
                 and not _plv.ambiguous_axes
                 and abs(_plv.origin_col - _mL) <= 3.0 and abs(_plv.origin_row - _mT) <= 3.0,
                 f"#P3 wide one-edge grid resolves BOTH absolute indices "
                 f"(method={_plv.method}, absolute={_plv.absolute_origin}, "
                 f"ambiguous='{_plv.ambiguous_axes}')")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"geom-edge-margin path raised: {e!r}")

    # ---------------------- 23. multi-disjoint-snapshot stitched heightmap (Center + TopLeft) #
    print("\n[23] multi-disjoint-snapshot: independent crops measured + tiled into one heightmap")
    try:
        import tempfile
        from run_sample import (build_snapshot_montage, snapshots_from_dir, _label_from_name,
                                _visible_content_box)
        from assemble import assemble_tiles

        _msd = _resolve_dxf("D300_P350_1cm2.dxf", "markerless")
        _mcell = read_design(_msd).cells[0]
        _marr = _mcell.arrays[0]
        _msc = 10.0
        _mo = (60.0, 60.0)
        _mp = _marr.pitch_x_um / _msc

        # Two INDEPENDENT captures of the same uniform cell at DIFFERENT absolute floors (60 vs 90
        # um): separate sessions do not share a Z zero, so the montage must floor-level each panel.
        _full_ctr, _ = synth_scan(_mcell, x_um_per_px=_msc, y_um_per_px=_msc, origin_px=_mo,
                                  marker=False, depth_um=30.0, floor_um=60.0, seed=970)
        _full_cor, _ = synth_scan(_mcell, x_um_per_px=_msc, y_um_per_px=_msc, origin_px=_mo,
                                  marker=False, depth_um=30.0, floor_um=90.0, seed=971)

        def _crop(full, r0, r1, c0, c1):
            return SynthScan(full.height_raw[r0:r1, c0:c1], full.intensity[r0:r1, c0:c1],
                             full.x_um_per_px, full.y_um_per_px, full.z_um_per_digit)

        # "Center" = deep interior crop (phase-only, xy) ; "TopLeft" = corner crop with two
        # captured pin terminations (absolute origin resolved) -- exactly the user's two cases.
        _center = _crop(_full_ctr, int(_mo[1] + 5 * _mp), int(_mo[1] + 20 * _mp),
                        int(_mo[0] + 5 * _mp), int(_mo[0] + 20 * _mp))
        _corner = _crop(_full_cor, 0, int(_mo[1] + 18 * _mp), 0, int(_mo[0] + 18 * _mp))
        _cp = register_sample(_center, _mcell, mirror_x=False)
        _tp = register_sample(_corner, _mcell, mirror_x=False)

        # (A) per-snapshot registration: interior stays phase-only/ambiguous; corner resolves both.
        ck.check(len(_cp) == 1 and _cp[0].method == "uniform-phase"
                 and not _cp[0].absolute_origin and _cp[0].ambiguous_axes == "xy",
                 "#23A Center interior crop -> phase-only lock, ambiguous xy (position not absolute)")
        ck.check(len(_tp) == 1 and _tp[0].absolute_origin and not _tp[0].ambiguous_axes
                 and max(abs(_tp[0].origin_col - _mo[0]), abs(_tp[0].origin_row - _mo[1])) <= 2.0,
                 "#23A TopLeft corner crop -> absolute origin resolved on both axes")

        _tiles = [dict(label="Center", scan=_center, placement=_cp[0], snapshot_id=1),
                  dict(label="TopLeft", scan=_corner, placement=_tp[0], snapshot_id=2)]

        # (B) both snapshots measured in ONE dataset: build rows exactly as analyze_multi_snapshot
        # does; assert a snapshot column, one row per array per snapshot, unique identities, and
        # accurate geometry recovered from each partial crop.
        _bt = ra._band_targets(_mcell)
        _rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _t in _tiles:
                for a in _mcell.arrays:
                    _s = ArraySample(
                        filename=f"{_t['label']}_b{a.band}c{a.col}_D{a.diameter_um:g}",
                        vk4_stem=_t["label"], cell_id=_t["snapshot_id"], array_id=a.array_id,
                        band=a.band, col=a.col, passes=35, speed=400.0,
                        nominal_diameter_um=a.diameter_um, target_diameter_um=_bt[a.band],
                        nominal_pitch_um=a.pitch_um, nominal_pitch_x_um=a.pitch_x_um,
                        nominal_pitch_y_um=a.pitch_y_um, nx=a.nx, ny=a.ny,
                        cx_um=a.cx_um, cy_um=a.cy_um)
                    _res = extract_array(_t["scan"], _t["placement"], a, _s, make_qc=False)
                    _row = ra.result_to_row(_res, True)
                    _row["snapshot"], _row["snapshot_id"] = _t["label"], _t["snapshot_id"]
                    _rows.append(_row)
        _df = pd.DataFrame(_rows)
        ck.check("snapshot" in _df.columns and len(_df) == 2 * _mcell.n_arrays
                 and set(_df["snapshot"]) == {"Center", "TopLeft"}
                 and _df["cell_id"].nunique() == 2 and _df["filename"].nunique() == len(_df),
                 "#23B both snapshots measured in one table, tagged + collision-free")
        ck.check(bool((_df["diameter_um"].sub(295.0).abs() <= 5.0).all()
                      and (_df["depth_um"].sub(30.0).abs() <= 4.0).all()
                      and (_df["n_cells"] > 0).all()
                      and (_df["n_cells"] < _marr.nx * _marr.ny).all()),
                 "#23B each partial crop recovers accurate diameter/depth from its visible pins")
        ck.check(bool((~_df.loc[_df["snapshot"] == "Center", "absolute_origin"]).all()
                      and _df.loc[_df["snapshot"] == "TopLeft", "absolute_origin"].all()),
                 "#23B phase-only/absolute status propagates per-snapshot into the CSV rows")

        # (C) the montage: ONE image (height + matching intensity), panels DISJOINT with a gutter,
        # each floor-referenced so the 30 um relief survives despite the 30 um floor offset.
        _m = build_snapshot_montage(_tiles, _mcell)
        _b0, _b1 = _m["boxes"][0], _m["boxes"][1]
        _gutter = _m["canvas"][:, _b0["c1"]:_b1["c0"]]
        _reg0 = _m["canvas"][_b0["r0"]:_b0["r1"], _b0["c0"]:_b0["c1"]]
        _reg1 = _m["canvas"][_b1["r0"]:_b1["r1"], _b1["c0"]:_b1["c1"]]
        ck.check(_m is not None and _m["canvas"].shape[1] >= _reg0.shape[1] + _reg1.shape[1]
                 and _b0["c1"] < _b1["c0"] and bool(np.all(np.isnan(_gutter)))
                 and (_b1["c0"] - _b0["c1"]) == _m["gutter_px"],
                 "#23C montage stitches both tiles side by side, disjoint across an all-NaN gutter")
        ck.check(bool(np.isfinite(_reg0).any() and np.isfinite(_reg1).any()
                      and 25.0 <= np.nanmax(_reg0) <= 35.0 and 25.0 <= np.nanmax(_reg1) <= 35.0),
                 "#23C both panels contribute relief; per-tile floor-levelling neutralises Z offset")
        ck.check(_m["intensity"] is not None and _m["intensity"].shape == _m["canvas"].shape,
                 "#23C intensity montage exists and is tiled in the SAME layout as the height map")
        ck.check([b["label"] for b in _m["boxes"]] == ["Center", "TopLeft"]
                 and not _m["boxes"][0]["placement"].absolute_origin
                 and _m["boxes"][1]["placement"].absolute_origin,
                 "#23C montage panels carry the snapshot names + honest phase-only/absolute flags")

        # (D) fail-closed: the raster assembler must REFUSE disjoint, non-_Y_X snapshots (never
        # silently mosaic them) -- proving the multi-snapshot path bypasses assemble entirely.
        _td = Path(tempfile.mkdtemp())
        try:
            (_td / "072226_PFLMTIM_D300_Center.vk4").write_bytes(b"not a raster tile")
            (_td / "072226_PFLMTIM_D300_TopLeft.vk4").write_bytes(b"not a raster tile")
            _raster_refused = False
            try:
                assemble_tiles(_td, verbose=False)
            except (SystemExit, ValueError):
                _raster_refused = True
            ck.check(_raster_refused,
                     "#23D assemble_tiles refuses disjoint non-_Y_X snapshots (no silent mosaic)")
            _disc = snapshots_from_dir(_td)
            ck.check([lab for _, lab in _disc] == ["Center", "TopLeft"]
                     and _label_from_name("072226_PFLMTIM_D50_TopLeft.vk4") == "TopLeft",
                     "#23D snapshot discovery labels tiles by their trailing filename token")
        finally:
            shutil.rmtree(_td, ignore_errors=True)

        # (E) order-invariance: the montage layout + per-tile content do not depend on input order.
        _m_rev = build_snapshot_montage(list(reversed(_tiles)), _mcell)
        def _sig(m):
            out = {}
            for b in m["boxes"]:
                reg = m["canvas"][b["r0"]:b["r1"], b["c0"]:b["c1"]]
                out[b["label"]] = (reg.shape, round(float(np.nanmax(reg)), 3))
            return out
        ck.check(_sig(_m) == _sig(_m_rev),
                 "#23E montage panels are identical regardless of snapshot input order")

        # (F) 3D centre-5x5 height map: a well-formed centre block + a rendered surface file.
        from run_sample import save_3d_pin_map, _center_block_box
        _b3 = _center_block_box(_corner, _tp[0], _marr)
        ck.check(_b3 is not None and _b3[1] > _b3[0] and _b3[3] > _b3[2]
                 and (_b3[1] - _b3[0]) <= 6 * _marr.pitch_x_um,
                 "#23F centre-5x5 block box is well-formed and ~5 pitches wide")
        _td3 = Path(tempfile.mkdtemp())
        try:
            _ok3 = save_3d_pin_map(_corner, _tp[0], _mcell, _marr, _td3 / "a.png",
                                   param_label="Passes: 35\nSpeed: 400 mm/s")
            ck.check(bool(_ok3) and (_td3 / "a.png").is_file(),
                     "#23F 3D centre-5x5 height map renders to a file")
            # presentation styling: 10 in wide at 300 dpi -> a slide-ready raster, not a thumbnail
            import matplotlib.image as _mpimg
            _w3 = _mpimg.imread(_td3 / "a.png").shape[1]
            ck.check(_w3 >= 2500, f"#23F 3D height map is written at presentation size ({_w3} px wide)")
        finally:
            shutil.rmtree(_td3, ignore_errors=True)
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"multi-disjoint-snapshot path raised: {e!r}")

    # ---------------------- 24. non-square (triangular) lattice: read + register + measure #
    print("\n[24] non-square TRIANGULAR lattice: geometry + markerless registration + extraction")
    try:
        _tri = (
            ("072326 D50 P100 TRIANGULAR.dxf", 50.0, 100.0, 2900,
             "c3abc09a3f8cbf28f5d0f28e637661e0d5a62fbefb1f3c5ffabb33f19e1c42f8"),
            ("072326 D100 P150 TRIANGULAR.dxf", 95.0, 150.0, 1254,
             "4e5886b3cffb5b852ee7425495dd60a37810220faff6d64315e7227c8e615bda"),
            ("072326 D300 P350 TRIANGULAR.dxf", 300.0, 350.0, 208,
             "0e3d0c3dda4a376a693a68b370fab0922fd4b8148f99a6215cd7022ae7baff8c"),
        )
        for _name, _dia, _pitch, _npins, _sha in _tri:
            _path = _resolve_dxf(_name, "triangular")
            print(f"    {_name} source (read-only): {_path}")
            _design = read_design(_path)
            _cell = _design.cells[0]
            _arr = _cell.arrays[0]
            _lv = np.asarray(_arr.lattice_vectors, float)
            ck.check(_canonical_dxf_sha256(_path) == _sha,
                     f"#24 {_name}: fixture content is unchanged (LF/CRLF equivalent)")
            ck.check(not _design.is_unit_cell and _design.n_markers == 0
                     and len(_design.cells) == 1 and _cell.marker_polygon_um is None,
                     f"#24 {_name}: parsed as one markerless cell")
            ck.check(len(_cell.arrays) == 1 and _arr.n_pins == _cell.n_pins == _npins
                     and abs(_arr.diameter_um - _dia) <= 1.0,
                     f"#24 {_name}: single {_npins}-pin array, drawn D{_dia:g} um")
            # NON-square lattice: the primitive basis is oblique and all six nearest neighbours sit
            # at the pitch -- i.e. |a1| == |a2| == min(|a1-a2|, |a1+a2|) == pitch (a triangular
            # lattice), whereas a square grid would give an axis-aligned basis with |a1-a2| = sqrt2*p.
            _steps = [np.hypot(*_lv[0]), np.hypot(*_lv[1]),
                      min(np.hypot(*(_lv[0] - _lv[1])), np.hypot(*(_lv[0] + _lv[1])))]
            ck.check(_lattice_is_oblique(_lv)
                     and max(abs(s - _pitch) for s in _steps) <= 1e-3
                     and abs(_arr.pitch_um - _pitch) <= 1e-6,
                     f"#24 {_name}: oblique (triangular) primitive basis, NN pitch {_pitch:g} um")

            _scale = 10.0
            _org = (60.0, 60.0)
            _ang = 1.5
            _p = _pitch / _scale
            # (A) full-array markerless scan at a stage rotation -> absolute finite-edge lock
            _full, _ = synth_scan(_cell, x_um_per_px=_scale, y_um_per_px=_scale, origin_px=_org,
                                  marker=False, rotation_deg=_ang, depth_um=30.0, floor_um=60.0,
                                  seed=740 + int(_dia))
            _, _, _, _fv, _fz = scan_feature(_full)
            _fp = _register_by_pattern(_full, _cell, _fz, _fv,
                                       x_right_options=(1,), y_up_options=(1,))
            ck.check(len(_fp) == 1 and _fp[0].method == "uniform-edge" and _fp[0].absolute_origin,
                     f"#24 {_name}: full triangular array resolves an absolute finite-edge origin")
            if _fp:
                ck.check(max(abs(_fp[0].origin_col - _org[0]),
                             abs(_fp[0].origin_row - _org[1])) <= 1.5
                         and abs(_fp[0].rotation_deg - _ang) <= 0.05,
                         f"#24 {_name}: absolute origin + {_ang:g} deg rotation recovered")

            # (B) deep-interior FOV crop -> phase-only lock (position not absolute), yet the visible
            # pins still yield accurate diameter/depth via the aligned oblique lattice.
            _tf, _ = synth_scan(_cell, x_um_per_px=_scale, y_um_per_px=_scale, origin_px=_org,
                                marker=False, depth_um=30.0, floor_um=60.0, taper_frac=0.2,
                                seed=760 + int(_dia))
            _allc = _arr.centers_um
            _cx = _org[0] + _allc[:, 0].mean() / _scale
            _cy = _org[1] + _allc[:, 1].mean() / _scale
            _fov = int(2600 / _scale)
            _H, _W = _tf.height_raw.shape
            _c0, _c1 = max(0, int(_cx - _fov / 2)), min(_W, int(_cx + _fov / 2))
            _r0, _r1 = max(0, int(_cy - _fov / 2)), min(_H, int(_cy + _fov / 2))
            _sub = SynthScan(_tf.height_raw[_r0:_r1, _c0:_c1], _tf.intensity[_r0:_r1, _c0:_c1],
                             _scale, _scale, _tf.z_um_per_digit)
            _sp = register_sample(_sub, _cell, mirror_x=False)
            ck.check(len(_sp) == 1 and _sp[0].method == "uniform-phase"
                     and not _sp[0].absolute_origin and _sp[0].ambiguous_axes == "xy",
                     f"#24 {_name}: interior crop -> honest phase-only lock (ambiguous xy)")
            _ss = ArraySample(
                filename="tri", vk4_stem="tri", cell_id=1, array_id=_arr.array_id,
                band=_arr.band, col=_arr.col, passes=1, speed=100.0,
                nominal_diameter_um=_arr.diameter_um, target_diameter_um=_arr.diameter_um,
                nominal_pitch_um=_arr.pitch_um, nominal_pitch_x_um=_arr.pitch_x_um,
                nominal_pitch_y_um=_arr.pitch_y_um, nx=_arr.nx, ny=_arr.ny,
                cx_um=_arr.cx_um, cy_um=_arr.cy_um)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _tr = extract_array(_sub, _sp[0], _arr, _ss, make_qc=False)
            ck.check(abs(_tr.diameter_um - _dia) <= 5.0 and abs(_tr.depth_um - 30.0) <= 4.0
                     and abs(_tr.pitch_um - _pitch) <= 1e-6 and 0 < _tr.n_cells < _arr.n_pins,
                     f"#24 {_name}: triangular crop yields accurate D/depth (mid {_tr.diameter_um:.1f}, "
                     f"depth {_tr.depth_um:.1f}) from the aligned lattice")
            ck.check(_tr.pin_centers_thumb is not None and len(_tr.pin_centers_thumb) > 3,
                     f"#24 {_name}: overlay marks actual triangular pin centres (not a square grid)")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"triangular-lattice path raised: {e!r}")

    # ------------------- 25. wafer-row batch runner: plan, dispatch, roll up #
    print("\n[25] wafer-row runner: name/dose/map parsing, DXF pairing, skips, dispatch, rollup")
    _scratch = None
    try:
        import json as _json
        import shutil as _shutil                    # main()-local elsewhere; bind our own
        import tempfile as _tempfile
        import calibrate_depth
        import laser_params
        import run_sample
        import wafer_map as wm
        import run_row as rr
        import row_report as rrep

        # ---- A. VK4 filename grammar (pure) ----
        ck.check(wm.parse_sample_id("072230_PFLMTIM_D50_11_Center") == (1, 1, "Center"),
                 "#25A compact _{col}{row}_ token -> (col, row, label)")
        ck.check(wm.parse_sample_id("072230_PFLMTIM_D300_64_Center") == (6, 4, "Center"),
                 "#25A first digit is the COLUMN, second the ROW")
        ck.check(wm.parse_sample_id("072230_PFLMTIM_D50_11_Center")[:2] == (1, 1)
                 and wm.parse_sample_id("990101_X_D50_23_Center") == (2, 3, "Center"),
                 "#25A the 6-digit date and the D-token are never read as a CR token")
        ck.check(wm.parse_sample_id("072230_PFLMTIM_D50_11_Top_Left") == (1, 1, "Top_Left"),
                 "#25A a multi-underscore snapshot label survives intact")
        ck.check(wm.parse_sample_id("072230_PFLMTIM_D50_112_Center") is None
                 and wm.parse_sample_id("072230_PFLMTIM_D50_1_Center") is None,
                 "#25A 1 or >=3 digits is never split into col/row")
        ck.check(wm.parse_sample_id("072230_PFLMTIM_D50_11") is None
                 and wm.parse_sample_id("072230_11_D50_21_Center") is None,
                 "#25A no snapshot label / two candidates -> refuse, never guess")
        ck.check(wm.parse_sample_id("072230_PFLMTIM_D50_C10R2_Center") == (10, 2, "Center"),
                 "#25A explicit C{col}R{row} escape hatch survives a >9-column wafer")

        # ---- B. grouping (pure, name-only) ----
        _names = [f"072230_PFLMTIM_D{d}_{c}{r}_{s}.vk4"
                  for c, d in ((1, 50), (2, 100), (3, 300)) for r in (1, 2, 3)
                  for s in ("Center", "TopLeft")]
        _by_col, _other, _unp = wm.group_snapshots(_names, 1)
        ck.check(sorted(_by_col) == [1, 2, 3] and all(len(v) == 2 for v in _by_col.values())
                 and len(_other) == 12 and not _unp,
                 "#25B 18 flat names -> 3 samples of 2 snapshots for row 1; 12 in other rows")
        ck.check(all(lab in ("Center", "TopLeft") for v in _by_col.values() for _n, lab in v),
                 "#25B 'Center' repeating ACROSS samples does not trigger the full-stem fallback")
        _dupn = ["072230_X_D50_11_Center.vk4", "072230_Y_D50_11_Center.vk4"]
        _dcol, _, _ = wm.group_snapshots(_dupn, 1)
        ck.check(len(_dcol[1]) == 2 and len({lab for _n, lab in _dcol[1]}) == 2,
                 "#25B colliding labels WITHIN one sample fall back to full stems, never merge")
        _mix, _, _ = wm.group_snapshots(
            ["072230_X_D50_11_Center.vk4"] + [f"072230_X_D100_21_{s}.vk4"
                                              for s in ("TopLeft", "Center", "TopRight")], 1)
        ck.check(len(_mix[1]) == 1 and [lab for _n, lab in _mix[2]] == ["Center", "TopLeft",
                                                                        "TopRight"],
                 "#25B 1 and 3 snapshots both group cleanly, ordered by SNAPSHOT_ORDER")
        _, _, _bad = wm.group_snapshots(["notes.vk4", "foo_Y1_X1.vk4", "a_XX_Center.vk4"], 1)
        ck.check(len(_bad) == 3 and all(r for _n, r in _bad),
                 "#25B untokened names / raster tiles land in 'unparsed' WITH a reason")

        # ---- C. dose + wafer map ----
        ck.check(wm.parse_dose("S400_P25") == wm.parse_dose("P25_S400") == (25, 400.0, "P25_S400"),
                 "#25C both token orders parse and canonicalise identically")
        ck.check(wm.parse_dose("P26_S800") == (26, 800.0, "P26_S800")
                 and wm.parse_dose("D50_P100_S400_P25") is None,
                 "#25C fullmatch-anchored: an embedded dose is refused, not guessed")
        ck.check(run_sample._parse_ps_label("S400_P25")[0] == 0
                 and run_sample._parse_ps_label("D50_P100_S400_P25") == (100, 400.0),
                 "#25C negative pin: run_sample._parse_ps_label reads S-first as passes=0 and a "
                 "geometry pitch as the pass count (this is WHY the row path canonicalises first)")
        ck.check(laser_params.parse_pxsy("S400_P25") is None,
                 "#25C laser_params.parse_pxsy is P-first only")
        _scratch = Path(_tempfile.mkdtemp(prefix="pflm-row-selftest-"))
        _map_txt = ["row,col,laser,geometry,lattice,skip,note,dxf"]
        for _r, _lat in ((1, "hex"), (2, "square")):
            for _c, _g, _p in ((1, "D50 P100", 25), (2, "D50 P100", 20), (3, "D100 P150", 22),
                               (4, "D100 P150", 16), (5, "D300 P350", 17), (6, "D300 P350", 13)):
                _map_txt.append(f"{_r},{_c},S400_P{_p},{_g},{_lat},,,")
        for _c in range(1, 7):
            _map_txt.append(f'3,{_c},,,,1,"Stagger pattern incorrect, disregard",')
        for _c, _g, _p in ((1, "D300 P350", 26), (2, "D300 P350", 34), (3, "D100 P150", 32),
                           (4, "D100 P150", 44), (5, "D50 P100", 40), (6, "D50 P100", 50)):
            _map_txt.append(f"4,{_c},P{_p}_S800,{_g},square,,,")
        _mp = _scratch / "wafer_map.csv"
        _mp.write_text("\n".join(_map_txt) + "\n", encoding="utf-8")
        _ents, _meta, _probs = wm.read_wafer_map(_mp)
        ck.check(len(_ents) == 24 and sum(e.skip for e in _ents) == 6 and not _probs,
                 f"#25C the 24-line wafer map parses clean (got {len(_ents)} entries, "
                 f"{len(_probs)} problems)")
        ck.check([e.laser for e in _ents if e.row == 1] ==
                 ["P25_S400", "P20_S400", "P22_S400", "P16_S400", "P17_S400", "P13_S400"],
                 "#25C S-first row-1 doses canonicalise to the P-first pipeline form")
        _dup = _scratch / "dup.csv"
        _dup.write_text("row,col,laser,geometry,lattice\n1,3,P1_S1,D50 P100,hex\n"
                        "1,3,P2_S1,D50 P100,hex\n", encoding="utf-8")
        _, _, _dprob = wm.read_wafer_map(_dup)
        ck.check(any("duplicate" in p and "2" in p and "3" in p for p in _dprob),
                 "#25C a duplicate (row,col) is reported naming BOTH source lines")
        _badf = _scratch / "bad.csv"
        _badf.write_text("row,col,laser,geometry,lattice,skip\n1,1,nope,D50 P100,hex,\n"
                         "1,2,P1_S1,D50 P100,neither,\n1,3,P1_S1,nope,hex,\n"
                         "1,4,P1_S1,D50 P100,hex,maybe\n", encoding="utf-8")
        _, _, _bprob = wm.read_wafer_map(_badf)
        ck.check(len(_bprob) == 4 and all(any(k in p for k in ("laser", "lattice", "geometry",
                                                               "skip")) for p in _bprob),
                 f"#25C bad dose/lattice/geometry/skip each block, all reported together "
                 f"(got {len(_bprob)})")

        # ---- D. DXF resolution by CONTENT ----
        _dxfdir = _scratch / "DXF"
        _dxfdir.mkdir()
        for _n in ("D50_P100_1cm2.dxf", "D100_P150_1cm2.dxf", "D300_P350_1cm2.dxf"):
            _shutil.copy2(_resolve_dxf(_n, "markerless"), _dxfdir / _n)
        for _n in ("072326 D50 P100 TRIANGULAR.dxf", "072326 D100 P150 TRIANGULAR.dxf",
                   "072326 D300 P350 TRIANGULAR.dxf"):
            _shutil.copy2(_resolve_dxf(_n, "triangular"), _dxfdir / _n)
        for _n in ("071826_UVPFLM_D300.dxf", "072026_UVPFLM_D100D50.dxf"):
            _shutil.copy2(_resolve_dxf(_n, "registration"), _dxfdir / _n)
        _facts, _fprob = rr.index_dxf_dir(_dxfdir)
        _sq = {f.name for f in _facts if f.is_wafer_candidate and f.lattice == "square"}
        _tri = {f.name for f in _facts if f.is_wafer_candidate and f.lattice == "triangular"}
        ck.check(len(_facts) == 8 and not _fprob and len(_sq) == 3 and len(_tri) == 3,
                 f"#25D 8 drawings parsed; 3 markerless square + 3 triangular are candidates "
                 f"(got {len(_sq)}sq/{len(_tri)}tri)")
        ck.check(all(rr.lattice_kind(rr.read_design_cached(_dxfdir / _n).cells[0].arrays[0])
                     == "triangular"
                     for _n in ("072326 D50 P100 TRIANGULAR.dxf",
                                "072326 D100 P150 TRIANGULAR.dxf",
                                "072326 D300 P350 TRIANGULAR.dxf")),
                 "#25D lattice_kind says 'triangular' for all three despite 60/60/120 deg bases")
        _cands350 = [f for f in _facts if abs(f.pitch_um - 350) < 0.5 and f.lattice == "square"]
        _wafer350 = [f for f in _cands350 if f.is_wafer_candidate]
        ck.check(len(_cands350) == 2 and len(_wafer350) == 1,
                 f"#25D the markerless predicate resolves the (350, square) collision "
                 f"({len(_cands350)} match pitch+lattice, {len(_wafer350)} survives)")
        _need1 = {(e.geometry, e.lattice) for e in _ents if e.row == 1 and not e.skip}
        _map1, _mprob1, _ = rr.resolve_dxfs(_facts, _need1)
        ck.check(not _mprob1 and len(_map1) == 3
                 and all("TRIANGULAR" in Path(v).name for v in _map1.values()),
                 "#25D row 1 (hex) resolves to the 3 TRIANGULAR drawings over 6 samples")
        _need4 = {(e.geometry, e.lattice) for e in _ents if e.row == 4 and not e.skip}
        _map4, _mprob4, _ = rr.resolve_dxfs(_facts, _need4)
        ck.check(not _mprob4 and "P350" in Path(_map4[("D300 P350", "square")]).name.upper()
                 and "P100" in Path(_map4[("D50 P100", "square")]).name.upper(),
                 "#25D row 4 (square, REVERSED pairing) resolves per-line, not by column order")
        _mx, _mxp, _ = rr.resolve_dxfs(_facts, {("D200 P250", "square")})
        ck.check(not _mx and len(_mxp) == 1 and "D200 P250" in _mxp[0] and "square" in _mxp[0],
                 "#25D an unmatched geometry is one blocking problem naming geometry + lattice")
        _ov, _ovp, _ovw = rr.resolve_dxfs(
            _facts, {("D50 P100", "square")},
            explicit={("D50 P100", "square"): str(_dxfdir / "072326 D50 P100 TRIANGULAR.dxf")})
        ck.check(not _ov and _ovp and "override rejected" in _ovp[0],
                 "#25D a dxf= override with the wrong lattice is VERIFIED and rejected")
        _nd, _ndp, _ndw = rr.resolve_dxfs(_facts, _need4)
        ck.check(not _ndp and not any("295" in w or "Ø" in w for w in _ndw),
                 "#25D a drawing deliberately undersized against its nominal label "
                 "(D300_P350 draws 295 µm) is neither an error nor warning noise")
        _tri_lie = _scratch / "DXF2"
        _tri_lie.mkdir()
        _shutil.copy2(_dxfdir / "D50_P100_1cm2.dxf", _tri_lie / "072326 D50 P100 TRIANGULAR.dxf")
        _lf, _ = rr.index_dxf_dir(_tri_lie)
        ck.check(any("TRIANGULAR" in w and "square" in w
                     for w in rr.resolve_dxfs(_lf, {("D50 P100", "square")})[2]),
                 "#25D a filename claiming TRIANGULAR over a square drawing IS warned about")

        # ---- E. plan composition ----
        _rnames = [f"072230_PFLMTIM_D{d}_{c}{r}_{s}.vk4"
                   for c, d in ((1, 50), (2, 50), (3, 100), (4, 100), (5, 300), (6, 300))
                   for r in (1, 2, 3) for s in ("Center", "TopLeft")]
        _p1 = wm.plan_row(_ents, _rnames, _map1, 1, date_tag="072426")
        ck.check(not _p1.blocking and len(_p1.ready) == 6
                 and len({s.dxf for s in _p1.ready}) == 3,
                 "#25E row 1 plans 6 ready samples over 3 distinct DXFs")
        _p3 = wm.plan_row(_ents, _rnames, {}, 3, date_tag="072426")
        ck.check(all(s.status == "skipped" for s in _p3.samples) and len(_p3.samples) == 6
                 and not _p3.blocking,
                 "#25E row 3 is six skipped samples with notes, and does not block")
        _p1miss = wm.plan_row(_ents, [n for n in _rnames if "_41_" not in n], _map1, 1)
        ck.check(sum(s.status == "no-vk4" for s in _p1miss.samples) == 1
                 and len(_p1miss.ready) == 5 and not _p1miss.blocking,
                 "#25E a missing column is 'no-vk4'; the other five still run")
        _swapped = {(e.geometry, e.lattice): _map4.get((e.geometry, e.lattice), "x")
                    for e in _ents if e.row == 4 and not e.skip}
        _p4bad = wm.plan_row(
            [wm.MapEntry(row=4, col=1, line=1, laser="P26_S800", passes=26, speed=800.0,
                         geometry="D50 P100", nominal_d_um=50.0, nominal_p_um=100.0,
                         lattice="square")],
            ["072230_PFLMTIM_D300_14_Center.vk4"], {("D50 P100", "square"): "x"}, 4)
        ck.check(_p4bad.blocking and _p4bad.samples[0].status == "name-mismatch"
                 and "REVERSES" in _p4bad.samples[0].reason,
                 "#25E a transposed map is caught by the VK4 D-token cross-check, before any run")
        _p4ok = wm.plan_row(
            [wm.MapEntry(row=4, col=1, line=1, laser="P26_S800", passes=26, speed=800.0,
                         geometry="D50 P100", nominal_d_um=50.0, nominal_p_um=100.0,
                         lattice="square")],
            ["072230_PFLMTIM_D300_14_Center.vk4"], {("D50 P100", "square"): "x"}, 4,
            strict_names=False)
        ck.check(not _p4ok.blocking and _p4ok.samples[0].status == "ready" and _p4ok.warnings,
                 "#25E strict_names=False downgrades the mismatch to a warning")
        _outs = [s.out_name for s in _p1.samples]
        ck.check(len(set(_outs)) == len(_outs)
                 and not any(ch in n for n in _outs for ch in '[]?*<>:"|')
                 and wm.row_out_name("072426", 1) == "072426 Row 1",
                 "#25E sample folder names are unique and filesystem-safe; row name is as specified")

        # ---- F. the driver on disk, with analyze stubbed ----
        _root = _scratch / "Results"
        _root.mkdir()
        _rowdir = _root / wm.row_out_name("072426", 1)
        _seen_kwargs = {}

        def _stub(snapshots, out_dir, dxf_path, *, passes, speed, cell_label,
                  make_qc=False, jobs=None, results_root=None):
            _col = int(Path(out_dir).name.split()[0][1:])
            _seen_kwargs[_col] = (passes, speed, cell_label)
            if _col == 3:
                raise SystemExit("No snapshots could be registered against the DXF.")
            _st, _fin = run_sample._prepare_output_transaction(
                out_dir, protect=(dxf_path,), results_root=results_root)
            _arr = rr.read_design_cached(dxf_path).cells[0].arrays[0]
            _rows = []
            for _sid, (_pn, _lab) in enumerate(snapshots, start=1):
                _r = {c: float("nan") for c in rrep.canonical_columns()}
                _r.update(dict(filename=f"{_lab}", vk4_stem=_lab, cell_id=_sid, array_id=1,
                               band=1, col=1, passes=passes, speed=speed,
                               nominal_pitch_um=_arr.pitch_um, depth_um=10.0 + passes,
                               dose_ratio=passes * 400.0 / speed,   # gates require it finite
                               diameter_um=_arr.diameter_um + 1.0,
                               top_diameter_um=_arr.diameter_um - 2.0,
                               base_diameter_um=_arr.diameter_um + 4.0, taper_um=6.0,
                               disc_mid_um=1.0, disc_top_um=-2.0, disc_base_um=4.0,
                               drawn_diameter_um=_arr.diameter_um,
                               nominal_diameter_um=_arr.diameter_um, debris_fraction=0.02,
                               reg_score=0.8, flags="", reliable=True))
                _r.update({"snapshot": _lab, "snapshot_id": _sid, "cell_label": cell_label})
                _rows.append(_r)
            _d = pd.DataFrame(_rows)
            (_st / "legacy").mkdir(parents=True, exist_ok=True)
            (_st / "figures").mkdir(parents=True, exist_ok=True)
            _d.to_csv(_st / "legacy" / "measurements.csv", index=False)
            (_st / "figures" / "run_manifest.json").write_text(
                _json.dumps({"created": rr._now()}), encoding="utf-8")
            run_sample._commit_output_transaction(_st, _fin)
            return _d, [], []

        for _n in _rnames:
            (_scratch / _n).touch()
        _paths = {_n: _scratch / _n for _n in _rnames}
        _recs, _panels = rr.run_row(_p1, _rowdir, paths=_paths, results_root=_root,
                                    analyze=_stub, capture_panels=False, meta={"map": str(_mp)})
        _subs = sorted(p.name for p in _rowdir.iterdir()
                       if p.is_dir() and not p.name.startswith("."))
        ck.check(len(_subs) == 5 and all((_rowdir / s / "legacy" / "measurements.csv").is_file()
                                         for s in _subs),
                 f"#25F 5 of 6 samples committed as grandchildren of the Results root "
                 f"(got {len(_subs)})")
        ck.check(_seen_kwargs.get(1) == (25, 400.0, "P25_S400"),
                 "#25F the driver passes passes/speed as explicit kwargs and the canonical dose")
        _r3 = next(r for r in _recs if r.planned.entry.col == 3)
        ck.check(_r3.status == "failed" and "SystemExit" in _r3.reason
                 and sum(1 for r in _recs if r.produced_data) == 5,
                 "#25F a SystemExit on sample 3 does not abort the row (SystemExit is not Exception)")
        ck.check(not [p for p in _rowdir.iterdir() if "staging" in p.name],
                 "#25F no .staging-* residue is left behind")
        _written = rr.write_rollup(_recs, _p1, _rowdir,
                                   {"map": str(_mp), "dxf_dir": str(_dxfdir), "vk4_dirs": []},
                                   _panels)
        ck.check(not run_sample._looks_like_legacy_output(_rowdir)
                 and not (_rowdir / "legacy").exists() and not (_rowdir / "figures").exists(),
                 "#25F the row container never gains legacy/ or figures/")
        try:
            run_sample._validate_output_target(_rowdir, results_root=_root)
            _guarded = False
        except SystemExit as _e:
            _guarded = "CONTAINS other PFLM results" in str(_e)
        ck.check(_guarded,
                 "#25F the container is REFUSED as a transaction target (a commit there would "
                 "delete every sample inside it)")
        try:
            run_sample._validate_output_target(_rowdir / _subs[0], results_root=_root)
            _nested_ok = True
        except SystemExit:
            _nested_ok = False
        ck.check(_nested_ok, "#25F regression pin: a normal single-sample dataset still validates")
        _fake_orphan = _rowdir / f".{_subs[0]}.staging-zzz"
        _shutil.copytree(_rowdir / _subs[0], _fake_orphan)
        ck.check(len([n for n, _c in calibrate_depth.discover_samples(_root, _root / "etch depth")
                      if n.startswith("072426 Row 1/")]) == 5,
                 "#25F depth calibration finds the 5 nested samples and skips the dot-staging orphan")
        _shutil.rmtree(_fake_orphan, ignore_errors=True)
        ck.check(_json.loads((_rowdir / ".pflm-row.json").read_text(encoding="utf-8"))["row"] == 1,
                 "#25F the row container marks itself before the first sample runs")

        # ---- G. rollup + figures ----
        _roll = rrep.build_rollup(_recs, row=1, date_tag="072426")
        _expect = rrep.IDENT_COLS + rrep.canonical_columns() + rrep.MODE_EXTRAS
        ck.check(list(_roll.columns)[:len(_expect)] == _expect,
                 "#25G rollup column order is IDENT + canonical + mode extras, explicitly reindexed")
        ck.check(len(_roll) == 11
                 and not _roll[_roll["snapshot_id"] > 0].duplicated(
                     subset=["sample", "snapshot", "array_id"]).any()
                 and list(_roll["wafer_col"]) == sorted(_roll["wafer_col"]),
                 "#25G 10 measured + 1 placeholder row; identities unique; sorted by wafer column")
        _ph = _roll[_roll["wafer_col"] == 3]
        ck.check(len(_ph) == 1 and not bool(_ph["reliable"].iloc[0])
                 and not np.isfinite(_ph["depth_um"].iloc[0])
                 and _ph["status"].iloc[0] == "failed",
                 "#25G a failed sample is ONE placeholder row, reliable=False, NaN measurements")
        _kept, _gate_rep = calibrate_depth.apply_gates(_roll.copy())
        ck.check(len(_kept) == 10 and 3 not in set(_kept["wafer_col"]),
                 "#25G the placeholder is dropped by the depth-calibration gates without error")
        _empty = rrep.build_rollup([], row=1, date_tag="072426")
        ck.check(_empty.empty and list(_empty.columns) == _expect,
                 "#25G an empty rollup still carries the full schema (fail closed, write no CSV)")
        _units = rrep.build_units(_roll)
        ck.check(len(_units) == 6 and _units["lattice"].eq("hex").all()
                 and np.isfinite(_units.loc[_units["wafer_col"] == 3, "phi_design"]).all(),
                 "#25G one unit row per sample; the failed column keeps a design Phi from the map")
        _figs = rrep.make_row_figures(_roll, _units, _rowdir / "row_figures", panels=None,
                                      plan=_p1, records=_recs)
        for _want in ("row_depth_vs_passes.png", "row_diameter_fidelity.png",
                      "row_porosity.png", "row_summary_table.png"):
            ck.check(any(p.name == _want for p in _figs),
                     f"#25G {_want} is written even with one column entirely failed")
        ck.check(not any(p.name == "row_montage.png" for p in _figs),
                 "#25G the montage is SKIPPED (not faked) when no panels were captured")
        # Drive the montage for real: captured panels for only SOME columns, so the missing one
        # must become an all-NaN placeholder panel (np.nanpercentile over an all-NaN slice warns,
        # hence simplefilter('error') -- this whole path is otherwise never exercised).
        _rngm = np.random.default_rng(3)
        _mpanels = [(int(c), f"c{int(c)}", "D50 P100", "P25_S400",
                     (_rngm.random((40, 60)) * 20).astype(np.float32))
                    for c in _units["wafer_col"] if int(c) not in (3, 6)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _mfig = rrep._fig_montage(_units, _mpanels, _rowdir / "row_figures" / "row_montage.png")
        ck.check(_mfig is not None and _mfig.is_file(),
                 "#25G the montage renders with captured panels, NaN-padding the missing columns")


        ck.check("5/6 samples registered" in rrep.render_row_summary(_roll, _units, _recs, _p1),
                 "#25G row_summary.txt states the true registered count")

        # ---- H. regressions pinned from the adversarial review ----
        try:
            rr.validate_row_container(_rowdir / _subs[0])    # an existing single-sample dataset
            _blocked = False
        except SystemExit as _e:
            _blocked = "SINGLE-sample result" in str(_e)
        ck.check(_blocked,
                 "#25H run_row REFUSES an existing single-sample dataset as a row container "
                 "(before any sample commits inside it)")
        _wf = _scratch / "warnword.csv"
        _wf.write_text("row,col,laser,geometry,lattice\n1,1,WARNING,D50 P100,square\n",
                       encoding="utf-8")
        _, _, _wp = wm.read_wafer_map(_wf)
        ck.check(_wp and not any(x.startswith("WARNING") for x in _wp),
                 "#25H a map cell whose TEXT is 'WARNING' still blocks — classification is by "
                 "leading marker, not substring")
        _e1 = wm.MapEntry(row=1, col=1, line=1, laser="P1_S1", passes=1, speed=1.0,
                          geometry="D50 P100", nominal_d_um=50.0, nominal_p_um=100.0,
                          lattice="square")
        ck.check(wm.plan_row([_e1], ["a_D50_11_Center.vk4", "a_D300_11_TopLeft.vk4"],
                             {("D50 P100", "square"): "x"}, 1).samples[0].status == "name-mismatch",
                 "#25H a MIXED VK4 D-token set is a mismatch — every snapshot must agree")
        _orph = wm.plan_row([_e1], ["a_D50_11_Center.vk4", "a_D100_21_Center.vk4"],
                            {("D50 P100", "square"): "x"}, 1)
        ck.check(_orph.blocking and any("no line in the wafer map" in p for p in _orph.problems),
                 "#25H a wafer column with VK4 files but no map line blocks instead of vanishing")
        _qn = _scratch / "quoted.csv"
        _qn.write_text('# date: 072426\nrow,col,laser,geometry,lattice,skip,note\n'
                       '1,1,S400_P25,D50 P100,hex,,"first\n# not a comment"\n'
                       '1,2,S400_P20,D50 P100,hex,,ok\n', encoding="utf-8")
        _qe, _qm, _qp = wm.read_wafer_map(_qn)
        ck.check(len(_qe) == 2 and not _qp and [x.line for x in _qe] == [4, 5]
                 and "not a comment" in _qe[0].note,
                 "#25H a quoted note containing a newline (whose continuation starts with '#') "
                 "loses no record and keeps true line numbers")
        _wide = rr.DxfFact(path="w.dxf", name="w.dxf", n_cells=1, n_arrays=1, is_unit_cell=False,
                           marker_shape="", lattice="square", pitch_um=350.3, diameter_um=295.0,
                           n_pins=100)
        ck.check(rr.resolve_dxfs([_wide], {("D300 P350", "square")})[0]
                 .get(("D300 P350", "square")) == "w.dxf",
                 "#25H PITCH_TOL_UM is honoured: a drawing pitched 350.3 µm matches a declared 350")

        # ---- I. staggered (centered-rectangular) lattice, as drawn on the 072426 wafer ----
        import types as _types

        def _arr(a1, a2):
            return _types.SimpleNamespace(lattice_vectors=np.array([a1, a2], float),
                                          pitch_x_um=float(np.linalg.norm(a1)),
                                          pitch_y_um=float(np.linalg.norm(a2)),
                                          pitch_um=0.5 * (float(np.linalg.norm(a1))
                                                          + float(np.linalg.norm(a2))))
        _stag = _arr((0.0, 150.0), (-150.0, 75.0))          # rows at 150 um offset by half a period
        ck.check(rr.lattice_kind(_stag) == "staggered"
                 and rr.lattice_kind(_arr((0.0, 100.0), (-86.6025, 50.0))) == "triangular"
                 and rr.lattice_kind(_arr((100.0, 0.0), (0.0, 100.0))) == "square",
                 "#25I lattice_kind separates staggered from triangular and square (hex is the "
                 "special case of a centered-rectangular lattice, so order matters)")
        ck.check(abs(rr.lattice_pitch_um(_stag) - 150.0) < 1e-6
                 and abs(_stag.pitch_um - 158.855) < 0.01,
                 "#25I lattice_pitch_um gives the 150 µm PERIOD where PinArray.pitch_um would give "
                 "the meaningless 158.9 µm mean of the two basis lengths")
        ck.check(wm.parse_lattice("Stagger") == wm.parse_lattice("staggered") == "stagger"
                 and wm.parse_lattice("Hex") == "hex" and wm.parse_lattice("neither") is None,
                 "#25I the wafer map accepts stagger as a third lattice, distinct from hex")
        _sf = rr.DxfFact(path="s50.dxf", name="s50.dxf", n_cells=1, n_arrays=1, is_unit_cell=False,
                         marker_shape="", lattice="staggered", pitch_um=150.0, diameter_um=50.0,
                         n_pins=1122)
        _sf100 = rr.DxfFact(path="s100.dxf", name="s100.dxf", n_cells=1, n_arrays=1,
                            is_unit_cell=False, marker_shape="", lattice="staggered",
                            pitch_um=150.0, diameter_um=100.0, n_pins=1089)
        _sm, _sp, _sw = rr.resolve_dxfs([_sf, _sf100], {("D50 P150", "stagger"),
                                                        ("D100 P150", "stagger")})
        ck.check(not _sp and _sm[("D50 P150", "stagger")] == "s50.dxf"
                 and _sm[("D100 P150", "stagger")] == "s100.dxf" and len(_sw) == 2,
                 "#25I two staggered drawings on the SAME period are separated by drawn diameter, "
                 "with a warning, instead of being called ambiguous")
    except Exception as e:                                   # pragma: no cover
        ck.check(False, f"wafer-row runner path raised: {e!r}")
    finally:
        if _scratch is not None:
            __import__("shutil").rmtree(_scratch, ignore_errors=True)

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
