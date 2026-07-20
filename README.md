# PFLM Profilometer Analysis (v2) — DXF-driven pin-fin metrology

Version 2 of the UV-laser pin-fin profilometry analysis (v1 = `../pinfin_analysis`).
Where v1 hard-coded the array geometry and one array per VK4, **v2 reads the geometry
straight from the fabrication DXF** and measures every array of every unit cell in a scan.

Pipeline:

1. **DXF → geometry** (`dxf_geometry.py`) — parse the drawing (units mm, via `ezdxf`) into a
   unit-cell template: every `CIRCLE` is a pin (drawn Ø = 2·radius); the ~200 µm square
   `LWPOLYLINE` at the bottom-left is the **alignment marker** = the origin. Pins are grouped
   into arrays (contiguous blocks of one diameter on a regular grid) with per-array diameter,
   pitch, `nx×ny`, and pin coordinates **relative to the marker**.
2. **Register** (`register.py`) — a VK4 scan may contain several tiled copies of the unit
   cell. Each is located by detecting its 200 µm marker (matched-filter on the scan's edge
   map, polarity-agnostic), then snapping to the pin lattice. Tiled cells are ordered
   left→right, bottom→top. A manual offset override is available if auto-detection struggles.
3. **Measure** (`extract.py`) — for each array, the known pin lattice is used to **classify
   every pixel as pin / clean-floor / debris**; the floor and pin-top *heights* (hence depth)
   come from those classified populations (robust to debris). The diameter is read from the
   mean pin, built by **stacking the patches centred on each known pin** (no fold wrap/edge
   artifacts), at three heights (base / mid / top) to capture taper.
4. **Report** (`run_analysis.py`) — `Results/measurements.csv` plus the same plot set as v1.

## Two workflows

**A. Full tiled sample (`run_sample.py`) — the main workflow for this data.** The whole chip
is scanned as a continuous `..._Y{n}_X{m}.vk4` tile raster spanning many unit cells. The
driver stitches the tiles into one scan (`assemble.py`), finds every unit cell by its 200 µm
alignment marker, and measures all of them. Cell layout notes for this dataset:

- The Keyence images **X-mirrored** vs the DXF (the design's bottom-left marker appears at each
  cell's bottom-**right**). This is handled as a per-cell coordinate transform — the array is
  never flipped; the presentation figures are flipped back to design orientation.
- Registration anchors on the **unique marker** plus one global rotation (not the periodic pin
  lattice, whose overlap aliases a pitch away), and uses a pin/floor **height-contrast gate**
  to reject flat un-ablated wafer.
- Cells are indexed by **design (row, col) with (1,1) = DXF top-left** (marker-anchored). Laser
  parameters are supplied per cell in `CSV/cell_params.csv` as a plain **grid in that design
  orientation**: line `r` = design row `r` (top first), column `c` = design col `c` (left first),
  each entry a `P{passes}_S{speed}` label (no header, no index columns, no template file). The
  Keyence scan is X-mirrored vs the design, so author the grid **as the DXF is drawn, not as the
  raw scan looks**. `run_sample.py` prints a `design(r,c) → marker x/y, rot, reg` table each run so
  you can confirm the mapping (a `rot` near ±180° flags a re-oriented wafer).

```bash
# 1. DXF/*.dxf present; drop the tile raster in VK4/ (…_Y1_X1.vk4 …)
python run_sample.py                    # -> Results/measurements.csv, cell_overview.png, plots
# then fill CSV/cell_params.csv (a P{passes}_S{speed} grid, DXF orientation) and re-run for the
# laser-parameter plots; or: python run_sample.py <vk4_dir> <out_dir> [<dxf>] [<cell_csv>]
```

**B. Individual scans (`run_analysis.py`).** One VK4 per capture (one or a few unit cells),
laser params by file in `CSV/laser_params.csv`.

```bash
python run_analysis.py
# or:  python run_analysis.py <vk4_dir> <out_dir> [<dxf_path>] [<csv_path>]
```

Inspect just the DXF geometry:

```bash
python dxf_geometry.py                 # prints arrays / bands / pitches / diameters
```

Validate the whole pipeline on synthetic data (no VK4 files needed):

```bash
python selftest.py                     # registration + extraction + plots, with assertions
python synth.py                        # writes Results/synth_preview.png
```

## Folders

| folder | contents |
|--------|----------|
| `DXF/` | the fabrication DXF (one unit cell, or a larger tiled design) |
| `VK4/` | Keyence profilometer scans (`*.vk4`) |
| `CSV/` | `laser_params.csv` — the per-cell laser settings you supply |
| `Results/` | `measurements.csv`, figures, per-file height maps, per-array QC |

## Laser parameters CSV

A scan can hold several unit cells machined with different settings, which are not in the
scan or the DXF — you provide them in `CSV/laser_params.csv`, one row per (file, cell):

```csv
vk4_file,cell,passes,speed,label
scan_A.vk4,1,20,400,bottom-left
scan_A.vk4,2,20,500,bottom-right
scan_A.vk4,3,40,800,top-left
scan_A.vk4,4,40,1000,top-right
single_cell_scan.vk4,,20,400,whole scan is one cell
```

* `cell` = 1-based tiled-grid order (left→right, bottom→top), as enumerated by registration.
  The number of cells the analysis looks for in a file = the max `cell` you list for it.
* `vk4_file` blank/`*` = default for any file; `cell` blank/`*` = every cell of that file.
* After each run a `laser_params_TEMPLATE.csv` of the cells actually detected is written for
  you to fill in. Rows beginning with `#` are treated as comments. On an interactive
  terminal, cells with no CSV entry are prompted for.

## Registration and the manual override

Auto-detection uses the alignment marker plus the pin lattice; the per-file height map
(`Results/heightmaps/<file>.png`) overlays the detected marker (white square) and pin
centres (red `+`) so you can confirm it. If a cell mis-registers, pass a manual override to
`register.register_scan(scan, template, n_cells, overrides={cell_id: {...}})` with
`origin_col`, `origin_row` (scan pixel of the marker's bottom-left corner) and optionally
`y_up` (+1/−1) and `rotation_deg`.

## Outputs (`Results/`)

| file | contents |
|------|----------|
| `measurements.csv` | one row per array per cell: measured pitch / base·mid·top Ø / depth, drawn Ø, laser params, registration quality, reliability flags |
| `heightmaps/<file>.png` | full scan height map with the registration overlay (one per VK4) |
| `qc/<name>.png` | per-array QC: height + known pins, mean pin with fitted rings, pin/floor/debris classification, radial profile |
| `figures/overview_3x3.png` | Ø / pitch / depth vs passes, speed, drawn Ø |
| `figures/per_row.png` | per-band depth & diameter vs speed |
| `figures/diameter_fit.png` | per-band measured-vs-drawn Ø with linear fit |
| `figures/depth_vs_dose.png` | depth vs the passes/speed dose proxy |
| `figures/dose_collapse.png` | depth & Ø-discrepancy vs dose |
| `figures/grid_overlays.png` | montage of the known pin grid on every array |
| `diameter_calibration.txt` | "draw X to get Y" calibration from the fits |

## Modules

- `dxf_geometry.py` — DXF → unit cells → pin arrays.
- `vk4.py` — Keyence VK4 reader (unchanged from v1).
- `assemble.py` — stitch a `_Y{n}_X{m}` tile raster into one scan.
- `register.py` — marker detection, DXF↔scan transform (`CellPlacement`, incl. X-mirror +
  rotation), `register_scan` (per-file) and `register_sample` (full tiled sample grid).
- `extract.py` — per-array measurement (classification-based heights, pin-stacked diameter).
- `laser_params.py` — CSV readers/template writers (by file, and by cell (row,col)).
- `run_sample.py` — full-sample driver (assemble → register grid → measure → plots).
- `run_analysis.py` — per-file batch driver + the shared plot set.
- `synth.py`, `selftest.py` — synthetic scan generator and end-to-end validation (35 checks).

## Notes / assumptions

- DXF units are millimetres (`$INSUNITS = 4`); a square 160–240 µm `LWPOLYLINE` at a cell's
  bottom-left is taken as its alignment marker. A DXF with no marker is treated as a single
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
