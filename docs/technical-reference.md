# Technical reference

This document preserves the detailed operating notes, data contracts, coordinate
conventions, and failure-mode guidance for the PFLM Profilometer Analysis pipeline.
For the project overview and quick start, see the [main README](../README.md).

## DXF-driven pin-fin metrology

Version 2 of the UV-laser pin-fin profilometry analysis. v1 is not in this
repository; it is archived beside it, at `UV Laser PFLM/archive/pinfin_analysis`.
Where v1 hard-coded the array geometry and one array per VK4, **v2 reads the geometry
straight from the fabrication DXF** and measures every array of every unit cell in a scan.

Pipeline:

1. **DXF → geometry** (`dxf_geometry.py`) — parse the drawing (units mm, via `ezdxf`) into a
   unit-cell template: every `CIRCLE` is a pin (drawn Ø = 2·radius); a small `LWPOLYLINE` near the
   bottom-left is the **alignment marker**. Two marker styles are recognised: the legacy ~200 µm
   **square** (deprecated; its corner is the cell origin) and a new asymmetric **L** fiducial
   (50 µm-wide arms) inset from the origin (its offset is derived from the cell boundary, so pins
   stay in true design coordinates). Pins are grouped into arrays (contiguous blocks of one
   diameter on a regular grid) with per-array diameter, pitch, `nx×ny`, and coordinates **relative
   to the cell origin**. The asymmetric L is the preferred marker; legacy square-marker samples
   remain supported through their established small-angle registration path.
2. **Register** (`register.py`) — a VK4 scan may contain several tiled copies of the unit
   cell. Each is located by detecting its 200 µm marker (matched-filter on the scan's edge
   map, polarity-agnostic), then snapping to the pin lattice. Tiled cells are ordered
   left→right, bottom→top. A manual offset override is available if auto-detection struggles.
3. **Measure** (`extract.py`) — for each array, the known pin lattice is used to **classify
   every pixel as pin / clean-floor / debris**; the floor and pin-top *heights* (hence depth)
   come from those classified populations (robust to debris). The diameter is read from the
   mean pin, built by **stacking the patches centred on each known pin** (no fold wrap/edge
   artifacts), at three heights (base / mid / top) to capture taper.
4. **Report** (`run_sample.py` + `report.py`) — `results/<sample>/legacy/measurements.csv` plus the
   figure set (clean figures under `figures/`, the v1 set under `legacy/figures/`).

## Workflow — full tiled sample (`run_sample.py`)

The whole chip is scanned as a continuous `..._Y{n}_X{m}.vk4` tile raster spanning many unit
cells. The driver stitches the tiles into one scan (`assemble.py`), finds every unit cell by its
200 µm alignment marker, and measures all of them. Cell layout notes for this dataset:

- The Keyence images **X-mirrored** vs the DXF (the design's bottom-left marker appears at each
  cell's bottom-**right**). This is handled as a per-cell coordinate transform — the array is
  never flipped; the presentation figures are flipped back to design orientation.
- Registration anchors on the **unique marker** plus one global rotation (not the periodic pin
  lattice, whose overlap aliases a pitch away), and uses a pin/floor **height-contrast gate**
  to reject flat un-ablated wafer.
- Asymmetric L markers and marker-free patterns search stage rotations from −5° to +5°.
  Aperiodic multi-array layouts use whole-pattern correlation. Complete markerless uniform arrays
  use a dedicated finite-edge solver: it fits lattice phase modulo pitch, enumerates every integer
  array-index hypothesis, and accepts an absolute origin only when physical pin terminations resolve
  **both** axes. Evidence is counted by lattice nodes with a spatial block bootstrap, not by pixels.
  For a single interior image, `register_sample` returns an explicitly labelled `uniform-phase`
  placement: rotation and pin centres are aligned for geometry measurement, while
  `absolute_origin=false` and `ambiguous_axes` prevent its arbitrary pitch-equivalent index from
  masquerading as an absolute origin. The strict low-level path can still fail closed.
- Cells are indexed by **design (row, col) with (1,1) = DXF top-left** (marker-anchored). Laser
  parameters are supplied per cell in `csv/cell_params.csv` as a plain **grid in that design
  orientation**: line `r` = design row `r` (top first), column `c` = design col `c` (left first),
  each entry a `P{passes}_S{speed}` label (no header, no index columns, no template file). The
  Keyence scan is X-mirrored vs the design, so author the grid **as the DXF is drawn, not as the
  raw scan looks**. `run_sample.py` prints a `design(r,c) → marker x/y, rot, reg` table each run so
  you can confirm the mapping (a `rot` near ±180° flags a re-oriented wafer).

```bash
# 1. dxf/*.dxf present; drop the tile raster in vk4/ (…_Y1_X1.vk4 …)
python python/run_sample.py                    # -> results/direct/{figures,legacy,...}
# then fill csv/cell_params.csv (a P{passes}_S{speed} grid, DXF orientation) and re-run for the
# laser-parameter plots; or: python python/run_sample.py <vk4_dir> <out_dir> [<dxf>] [<cell_csv>]
```

## Workflow — disjoint snapshots of one uniform cell (`run_sample.py --snapshots`)

Sometimes a scan is not a continuous raster but a few **independent snapshots** of one large
uniform unit cell — e.g. a `..._Center.vk4` interior crop and a `..._TopLeft.vk4` corner crop of
the same design — captured at unknown, unrelated stage positions. They have no overlap to stitch
by, and their absolute position in the cell is neither known nor needed. This mode registers
**each snapshot independently** against the one DXF cell (an interior crop resolves *phase-only*; a
corner that captures pin terminations can resolve an absolute origin), measures the pins visible in
each, and renders a **side-by-side tiled montage** under the usual figure names:

- `figures/sample_heightmap.png` — each snapshot's floor-referenced, design-oriented height crop
  tiled side by side and labelled by snapshot name. Per-tile floor levelling lets separate captures
  with different absolute Z share one honest colour scale. The panels are independent captures, not
  a spatial mosaic (their relative position is unknown) — the mm x/y axes are therefore a *scale*
  (true within a panel), not a position on the sample.
- `figures/intensity_map.png` — the same snapshots' intensity, tiled in the identical layout.
- `figures/3D height map/<snapshot>/array{id}_D{d}_P{p}.png` — a true-aspect 3D height surface of
  the **centre 5×5 pins** of each array, per snapshot, drawn presentation-style (low camera, large
  type, 300 dpi) so it can go straight on a slide. Diameter/pitch (and every geometry) come from
  the DXF, never the filename.
- `legacy/measurements.csv` — one row per array per snapshot, tagged with a `snapshot` column plus
  the per-snapshot `reg_method` / `absolute_origin` / `ambiguous_axes`.

All snapshots in one dataset share a single laser dose. Provide them as a folder of `*.vk4` files
**not** named `_Y{n}_X{m}` (that suffix selects the tiled-raster mode); each file's trailing
`_`-token is its label (`Center`, `TopLeft`, …). The tiled-raster `assemble_tiles` path is bypassed
entirely — it would refuse (or, on a periodic pin lattice, alias) disjoint non-overlapping crops.

```bash
python python/run_sample.py --snapshots <vk4_dir> <out_dir> <dxf> P{passes}_S{speed}
```

**From the UI:** select the dataset's folder as usual — the VK4 label shows
`snapshot montage: Center, TopLeft` when it detects disjoint snapshots, and Run routes to this mode
automatically (the shared dose is the first `P{passes}_S{speed}` in the params box).

## Workflow — a whole wafer row at once (`run_row.py`)

A wafer carries a GRID of samples: each is one uniform-cell dataset (one pin geometry, one laser
dose, one or more disjoint snapshots) — exactly the `--snapshots` unit above. `run_row.py` runs an
entire wafer ROW in one command and rolls the results up.

**VK4 naming.** Each file carries a compact `_{col}{row}_` token before its snapshot label — the
FIRST digit is the wafer column, the SECOND is the wafer row:

```
072230_PFLMTIM_D50_11_Center.vk4      -> col 1, row 1, snapshot 'Center'
072230_PFLMTIM_D50_13_TopLeft.vk4     -> col 1, row 3, snapshot 'TopLeft'
```

An explicit `C{col}R{row}` token also works, for a wafer wider than nine columns. A stem with no
unambiguous token is never guessed at — it is listed as unparsed and blocks the run until it is
renamed or `--allow-unparsed` is given.

**`csv/wafer_map.csv`** declares, per (row, col), the dose, the pin geometry and the lattice:

```csv
# date: 072426
row,col,laser,geometry,lattice,skip,note,dxf
1,1,S400_P25,D50 P100,hex,,,
1,3,S400_P22,D100 P150,hex,,,
3,1,,,,1,"Stagger pattern incorrect, disregard",
4,1,P26_S800,D300 P350,square,,,
```

`laser` accepts either token order (`S400_P25` or `P26_S800`) and canonicalises to the P-first form
the rest of the pipeline uses. Geometry is declared **per line and never inferred**: on this wafer
the column→geometry pairing REVERSES between rows 1–2 and row 4. `skip` drops a row from the run
while keeping the design point on record. `dxf` optionally names a drawing explicitly (still
content-verified). Every problem in the file is reported at once, with line numbers.

**DXF resolution is by CONTENT, not filename.** Each drawing is parsed and keyed on
`(pitch, lattice)`, where the lattice comes from the primitive basis (90° between equal vectors =
square, 60° = triangular/hex — `pitch_x_um`/`pitch_y_um` are equal for both and cannot tell them
apart). Only single markerless uniform arrays are candidates, which resolves the collision between
a wafer DXF and a legacy tiled design at the same pitch. Zero or ≥2 candidates is a blocking error
naming every file considered; nothing is guessed. Drawn diameter is a ±15 % sanity tolerance, never
a key — drawings are deliberately undersized against their nominal label (`D300_P350` draws 295 µm).

Three cross-checks guard the one genuinely silent failure — a transposed map, where a
right-pitch/wrong-diameter DXF registers cleanly and poisons every `drawn_diameter_um`: the VK4
filename's `D` token must agree with the map (blocking; `--allow-name-mismatch` to override), the
DXF filename's pitch/`TRIANGULAR` tokens must agree with its own content (warning), and after the
run `median(diameter_um)/drawn_diameter_um` must land in [0.7, 1.6] (status `suspect`).

```bash
python python/run_row.py --row 1 --dry-run
python python/run_row.py --row 1 --vk4 <dir> [<dir> ...] --dxf-dir <dir>
```

`--vk4` takes several folders, and a folder with no top-level `*.vk4` is searched recursively — so
pointing at the PARENT of per-geometry folders works, which matters because one wafer row is spread
across them. The full plan is printed before anything runs; `--dry-run` stops there. Exit codes:
`0` all ready samples produced data, `1` partial row, `2` none succeeded, `3` plan blocked (nothing
written). One sample failing never aborts the row — `SystemExit` is caught alongside `Exception`,
because that is how registration failure is reported.

Output layout — note the extra nesting level:

```
results/072426 Row 1/                  <- a plain CONTAINER, never a transaction target
    .pflm-row.json
    c1 D50 P100 P25_S400/              <- a normal, fully transactional per-sample dataset
    ...
    row_measurements.csv               <- every sample's rows + wafer_col/geometry/lattice/laser
    row_units.csv                      <- one row per sample (medians + design/achieved Φ)
    row_summary.txt  row_manifest.json
    row_figures/                       <- row_depth_vs_passes, row_diameter_fidelity,
                                          row_porosity, row_summary_table, row_montage
```

**The container must never contain `figures/` or a `legacy/measurements.csv`.**
`run_sample._looks_like_legacy_output` ANDs exactly those two conditions, and a transaction
committed on the container would rename the whole subtree into `%TEMP%` and delete it — destroying
every per-sample result while printing success. Hence the `row_*` names, plain atomic writes for
the rollup, and `run_sample._contains_owned_datasets`, which refuses the container as an output
target from the other side. A sample that failed to register keeps exactly one placeholder row in
`row_measurements.csv` (all-NaN, `reliable=False`): a missing design point must never be a missing
row. `row_measurements.csv` is built to be `pd.concat`ed across rows for a later hex-vs-square
comparison.

With two doses per geometry and a constant speed, no model is fitted — the figures connect the
measured points and label the segment slope. Pool several rows with `calibrate_depth.py` for a fit;
it descends one level into a row container, so the samples appear as
`072426 Row 1/c1 D50 P100 P25_S400`.

**From the UI:** select the wafer folder — the VK4 label shows `wafer map: 9 samples, rows 1,2,3`,
a `wafer row` chip and a row picker appear in the toolbar, and Run drives `run_row.py` in one
subprocess with the usual console/Stop plumbing.

Inspect just the DXF geometry:

```bash
python python/dxf_geometry.py                 # prints arrays / bands / pitches / diameters
```

Validate the whole pipeline on synthetic data (no VK4 files needed):

```bash
python python/selftest.py                     # synthetic pipeline + real-DXF alias regressions
python python/synth.py                        # writes results/synth_preview.png
```

When the Stanford OneDrive environment is available, `selftest.py` opens its original DXFs directly
from that data tree (read-only). Set `PFLM_DXF_DIR` to prefer another flat DXF directory. GitHub
Actions and other machines automatically fall back to the committed test fixtures. Integrity checks
ignore LF-versus-CRLF line endings but still detect any geometry/content change, and `.gitattributes`
prevents Git from rewriting DXF bytes during checkout.

## Folders

| folder | contents |
|--------|----------|
| `dxf/` | the fabrication DXF (one unit cell, or a larger tiled design) |
| `vk4/` | Keyence profilometer scans (`*.vk4`) |
| `csv/` | `cell_params.csv` (per-cell laser settings), `radial_sets.csv` (radial overlays), `wafer_map.csv` (per-sample dose/geometry/lattice for `run_row.py`) |
| `results/` | analysis outputs. Every run uses a dedicated `results/<dataset name>/` child; direct runs default to `results/direct/`. A wafer row adds one nesting level: `results/<date> Row <n>/<sample>/` |

## Laser parameters CSV (`cell_params.csv`)

Per-cell laser settings are not in the scan or the DXF — you supply them in
`csv/cell_params.csv` as a plain **grid in DXF/design orientation**: line `r` (top = row 1) is
design row `r`, column `c` (left = col 1) is design col `c`, each entry a `P{passes}_S{speed}`
label. No header, no index columns.

```csv
P30_S400,P35_S400,P40_S400
P45_S400,P50_S400,P55_S400
P60_S400,P65_S400,P70_S400
```

The Keyence scan is X-mirrored vs the design, so author the grid **as the DXF is drawn, not as
the raw scan looks**. Blank cells are skipped; a blank line still advances the row index (so an
intentional gap keeps later rows on their true numbers). Radial-average overlay sets live in
`csv/radial_sets.csv` — one comma-separated `P{passes}_S{speed}` set per line; empty = overlay
every parameter present.

## Registration

`register_sample` locates every unit cell automatically: it detects each cell's alignment marker
(an absolute, off-pin-lattice anchor), estimates the tile lattice from the strongest cells, and
probes every lattice node — so a dense periodic array registers cleanly and spurious marker hits
are rejected by construction. L-marker detection uses a coarse-to-fine −5° to +5° angle search;
the deprecated symmetric square retains its proven legacy detector. Marker-free aperiodic layouts
use whole-pattern correlation; a complete single uniform grid instead uses finite-array termination
evidence. It resolves a partial scan only when at least one pin-termination edge plus roughly one
pitch of valid floor is visible in **each** lattice direction. Otherwise the default high-level path
uses a centred, pitch-equivalent `uniform-phase` placement so diameter/depth can be measured from one
subsection; its absolute pin index remains explicitly unresolved. Pass
`allow_uniform_phase_only=False` to `register_sample` wherever absolute identity is mandatory.
Confirm the result from the `design(r,c) → marker x/y, rot, reg`
table printed each run (a `rot` near ±180° flags a re-oriented wafer) and the per-cell reports in
`results/<sample>/figures/cells/`. Cell (row,col) indices are absolute (a dropped interior
row/column leaves a hole, not a shift); a `cell_params` entry with no registered cell is warned as
a possible edge-dropout that could shift the parameter mapping. (`register_scan` remains the
low-level single-cell primitive used by `selftest.py`.)

The markerless finite-edge acceptance margin scales with the geometrically expected termination
evidence (~O(N), `register._edge_margin_threshold`), not the interior match area (~O(N²)), so a large
uniform grid stays identifiable when a real edge is captured while a no-edge crop still fails closed.

**Deferred registration hardening — pick up when real VK4 scans are in.** Two refinements are
intentionally deferred until fabrication scans exist, so they can be validated and tuned against
genuine imperfections rather than synthetic assumptions:

- **Honest per-axis "no-edge" vs "weak-edge" reporting** (`_register_uniform_lattice`). An unresolved
  axis is currently reported the same whether *no* termination edge was in frame (inherently
  unresolvable → `uniform-phase` is the correct answer) or an edge *was* captured but its evidence was
  too weak. Splitting the two would make the refusal message and the `ambiguous_axes` metadata state
  *why* each axis is unresolved. Needs real partial scans to confirm the "edge present" test is not
  fooled by interior noise.
- **Tile-seam phase-drift detect-and-refuse** (`_fit_uniform_lattice` + `assemble` stitching). A
  consistent sub-pitch tile-seam offset on a stitched mosaic can bias the fitted lattice phase and
  quietly shift predicted node positions. The plan is to *detect* seam-correlated phase residual and
  **refuse** (fail-closed) — never silently "correct" the phase (that would violate the "cleanest real
  results" / fail-closed policy). Sizing the threshold requires real stitched VK4s.

(A third idea, restricting the finite-index candidate enumeration to the observed span, was measured
to prune only ~6–9% of candidates on these wide grids — and the enumeration is not the bottleneck —
so it was dropped as not worth the added complexity.)

## Outputs (`results/<dataset name>/`)

The UI writes each run under `results/<sample name>/` — named after the sample selected in the UI
(a sample must be selected to run) — so different datasets keep separate result sets. A direct run
defaults to `results/direct/`; an explicit output must still be a dedicated child of this
repository's `results/` root. The paths below are relative to that per-run output root.

> **Each run is transactional.** New artifacts are written to an owned hidden staging sibling and
> validated before the completed directory atomically replaces the prior result. A failed run
> leaves the last completed result intact. Replacement is restricted to a dedicated child of
> `results/`; arbitrary, unowned, root, or input-overlapping directories are refused. An internal
> `.pflm-results.json` sentinel records ownership so recursive cleanup cannot target user data.

Clean outputs live under `figures/`; the v1 plot set, `measurements.csv` and per-array QC live under
`legacy/`.

| file | contents |
|------|----------|
| `legacy/measurements.csv` | one row per array per cell: base·mid·top Ø (`base_extrapolated` flag when the base crossing was buried), depth, **design** pitch (`pitch_*`) + scan-**measured** pitch (`meas_pitch_*`), drawn Ø, laser params, registration quality, `absolute_origin` / `ambiguous_axes`, reliability flags |
| `figures/intensity_map.png`, `figures/sample_heightmap.png` | intensity/cell map + full-sample height map (design orientation; height map carries physical mm x/y axes) |
| `figures/3D height map/[cell_x*_y*/]array*.png` | true-aspect 3D height surface of the centre 5×5 pins of each array (per unit cell for a tiled sample), presentation-styled at 300 dpi, with a passes/speed info box (top-left) |
| `figures/cells/cell_x*_y*.png` | per-cell report: height/intensity with pin overlay + measured-vs-drawn table |
| `figures/param_summary.png`, `figures/param_depth_scatter.png` | depth & mid-Ø oversizing vs laser passes/speed (reliable arrays; ◇ D300 □ D100 △ D50) |
| `figures/radial_overlays/<set>/a*.png` | mean-pin radial profiles overlaid across a laser-parameter set |
| `figures/` (provenance) | DXF copy, `vk4_source.zip`, `cell_params.csv`, `radial_sets.csv`, `run_manifest.json` (git commit + inputs) |
| `legacy/figures/{overview_3x3,per_row,diameter_fit,depth_vs_dose,dose_collapse,grid_overlays}.png` | the v1 plot set |
| `legacy/qc/<name>.png` | per-array QC (only with `make_qc`): height + known pins, mean pin with fitted rings, pin/floor/debris classification, radial profile |

## Depth calibration across samples (`calibrate_depth.py`)

A **post-hoc, additive** tool that pools the completed runs to answer "for this pin geometry, what
depth does a given (passes, speed) produce — and inversely, what (passes, speed) hits a target
depth (e.g. 55 µm)?" It is the reframed inverse of the per-run diameter model
(`report.make_diameter_model`, which fits Ø ~ drawn+passes+speed): here the **response is etch
depth** and the **predictors are the laser dose**.

Pooling across samples is defensible because `depth_um` = pin-top − clean-floor is a **local**
differential — per-tile Z offsets and stage tilt cancel — so unlike absolute height it is
comparable across samples/wafers/dates. Nothing in the per-run pipeline changes; it only **reads**
each sample's `legacy/measurements.csv`.

```
python python/calibrate_depth.py [--include A B] [--exclude C] [--targets 45,55,65] \
                          [--results Results] [--out "results/etch depth"] \
                          [--bands band_defs.csv] [--cell-filters cells.json] \
                          [--max-debris X] [--drop-shallow] [--allow-legacy-qc]
```

- **Discovers** the samples (folders under `results/` with a `legacy/measurements.csv`), injects a
  `sample` column, pools them; `--include`/`--exclude` select the set (default = all).
- **Per-sample cell selection** — `--cell-filters` names a JSON file mapping a sample folder name to
  a `cell_id` spec, applied **before** pooling so a single bad cell can be dropped without discarding
  the whole sample. A spec lists the cells to **keep** (`"1-5, 8, 12-16"`); a leading `!` **excludes**
  them (`"!3, 7"`); blank/omitted = all cells. `cell_id` is the 1-based unit-cell index (row-major:
  row 1 = top, col 1 = left). The applied filters are echoed to the console and recorded in
  `depth_calibration.txt` for provenance. A sample whose CSV lacks a `cell_id` column is skipped
  (fail-closed) rather than pooled unfiltered.
- **Gates** for a trustworthy depth read (`reliable`, finite depth) and prints the retained/total
  counts and *why* rows dropped. The debris cut is **off by default** — `debris_fraction` is a poor
  proxy for a bad depth read, so you vet which samples are good; pass `--max-debris X` to re-enable
  it. Missing `reliable` fails closed. `--allow-legacy-qc` is an explicit unsafe compatibility
  override for manually reviewed older CSVs. `shallow` (<3 µm) points are kept by default — they
  anchor the low-dose rise — but `--drop-shallow` removes them.
- **Bands** — `--bands` names a file whose every row is one band, `min_Ø, max_Ø, pitch` (µm): the
  first two numbers are the drawn-diameter range the band covers, the third its centre-to-centre
  pitch. A pooled row joins a band only if **both** its drawn Ø is in range (ends inclusive) **and**
  its pitch matches the band's declared pitch (strict, within ±1 µm); rows matching no band —
  Ø out of range, mismatched pitch, or missing pitch — are dropped and reported. Omitting `--bands`
  falls back to the measurements' own `band` **plus nominal pitch**, so a local `band 1` from a
  D50/P100 design cannot alias a `band 1` from a D300/P350 design.
- **Collapses technical replicates** to one median observation per sample/cell/band before fitting;
  arrays sharing a cell exposure are not counted as independent trials. The disjoint tiles of a
  **snapshot** dataset (e.g. Center + TopLeft) are replicate views of one uniform cell, so they
  collapse to a single observation per band with their **depth averaged** — not two independent
  trials.
- **Fits per band** (never pooling across pitch/diameter families): a saturating NLS
  `depth = a·(1−e^{−k·dose})`, a log-dose OLS (+ drawn-Ø covariate), and a passes×speed interaction
  OLS — all reported with R²/adj-R²/AICc/95% CI/p. Etch depth does **not** collapse to
  `dose = passes/speed` (at a fixed ratio, depth spans tens of µm depending on the actual passes and
  speed), so the **passes×speed interaction** model is recommended whenever it fits with meaningful
  signal (adj-R² ≥ 0.10); the dose-only forms are a fallback for single-factor sweeps. When nothing
  is informative, inverse recommendations are suppressed.
- **Pools CIs across samples** with a `sample` random-intercept MixedLM (falls back to a sample
  fixed factor if it won't converge), and prints a `sample × dose` coverage table so sample↔dose
  confounding is visible.
- **Writes** to `results/etch depth/`: `depth_calibration.txt` (fits, pooled model,
  coverage, per-target inversion with an inverse-mean confidence interval, and the extrapolation box),
  `depth_vs_passes_speed_3d.png` (per-band 3-D depth = f(passes, speed): measured points + the
  recommended-model surface + a target-depth plane), `depth_parity.png`, and `depth_heatmap.png`
  (passes×speed predicted-depth heatmap with the target-depth contour — read off which (P, S) hits
  the target). `depth_heatmap.png` (name kept for compatibility) is a **scatter of the measured
  cells only** — one marker per cell-band median at its (scan speed, passes), coloured by measured
  depth, with a **red dashed ring** on any cell whose depth landed within ±`TARGET_WINDOW_UM`
  (5 µm) of a requested target, i.e. the 50–60 µm window for the default 55 µm target. There is
  deliberately no model surface and no fitted contour: a colour field over the whole rectangle is a
  picture of depths nobody measured, and most of it would be extrapolation. Everything on that
  figure is a measurement; the fitted models and their caveats stay in `depth_calibration.txt`.

**From the UI:** the *Depth calibration* panel (right column) lists the discovered samples
(multi-select; none selected = all) in a two-column table — the sample name and an inline **cells**
column. Double-click a row's *cells* to include/exclude specific `cell_id`s for that sample
(`1-5, 8, 12-16`; prefix `!` to exclude; blank = all); the panel also has a **band-definitions** box
(one `min_Ø, max_Ø, pitch` per line; number of rows = number of bands; blank = use the CSV `band`
column) and a target depth (default `55`, comma-separated allowed). It runs the tool on the
selection — output streams to the console and the results land in `results/etch depth/` (browsable
at left).

## Modules

- `dxf_geometry.py` — DXF → unit cells → pin arrays.
- `vk4.py` — Keyence VK4 reader (unchanged from v1).
- `assemble.py` — stitch a `_Y{n}_X{m}` tile raster into one scan.
- `register.py` — marker detection, DXF↔scan transform (`CellPlacement`, incl. X-mirror +
  rotation), `register_scan` (per-file) and `register_sample` (full tiled sample grid).
- `extract.py` — per-array measurement (classification-based heights, pin-stacked diameter).
- `laser_params.py` — the `cell_params.csv` grid reader (by-cell (row, col)).
- `report.py` — shared measurement-row builder + the legacy plot suite.
- `run_sample.py` — full-sample driver (assemble → register grid → measure → plots); also the
  `--snapshots` multi-snapshot mode (register each disjoint crop of one uniform cell independently →
  measure → tiled montage under the usual figure names).
- `wafer_map.py` — wafer-row vocabulary: the VK4 `_{col}{row}_` filename grammar, the
  `wafer_map.csv` reader, and `plan_row` (pure; stdlib only, so `pflm_ui.py` can import it without
  pulling in numpy/pandas/matplotlib). Also owns the shared `safe_name` folder sanitiser.
- `run_row.py` — wafer-row driver: content-based DXF resolution, preflight plan, one
  `analyze_multi_snapshot` call per wafer column, error containment, rollup.
- `row_report.py` — the row rollup: combined `row_measurements.csv`/`row_units.csv` schema and the
  five cross-sample comparison figures.
- `calibrate_depth.py` — cross-sample etch-depth calibration (see below); reuses `report._ols_fit`.
- `figstyle.py` — house figure typography: named type-size roles (`TITLE`, `LABEL`, `TICK`,
  `LEGEND`, `ANNOT`, …) and the `PLOT_RC` rc-context, all derived from one `BUMP` constant. Every
  figure writer imports it, so a presentation-size change is a one-line edit rather than ~130
  scattered literals.
- `pflm_ui.py` — Tkinter sample-tester GUI (sample library, run/stop, figure preview,
  depth-calibration panel).
- `synth.py`, `selftest.py` — synthetic scan generator and end-to-end validation.

## Notes / assumptions

- DXF units are millimetres (`$INSUNITS = 4`); the alignment marker is a `LWPOLYLINE` near a cell's
  bottom-left — either a 160–240 µm **square** (deprecated) or an **L** fiducial (recognised by its
  6-vertex outline / low fill fraction). A DXF with no marker is treated as a single
  anchor-less design.
- Arrays are found by grouping same-diameter pins and splitting spatially, so neighbouring
  arrays never merge even when the inter-array gap equals the pitch (as in this design).
- **Depth/floor/pin-top come from the DXF-driven pin/floor/debris classification**, not from
  the folded profile; diameters come from the stacked mean pin referenced to those heights.
- Small stage rotation is assumed ~0 (a `rotation_deg` override exists; the measurement is
  rotation-aware when it is set). Registration is validated on synthetic scans
  (`selftest.py`); tune against the first real multi-cell VK4.
- Hardened + regression-tested (see `selftest.py`) for: tiled cell grids including a single
  row/column (1×N), non-square pixels (x≠y µm/px), y-flipped rasters (cells still numbered
  bottom→top to match the CSV), a spurious-marker quality gate when `n_cells` is overstated,
  and degenerate DXFs. Wildcard-only CSVs warn rather than silently dropping tiled cells.
- The provided `071626_UVLaserPFLM_4x4_singlecell.dxf` is one cell: 14 arrays (5×5 each,
  350 pins), 4 bands — Ø 50–67.5 µm @100 µm pitch (bands 1–2) and 100–125 µm @150 µm pitch
  (bands 3–4).
```
