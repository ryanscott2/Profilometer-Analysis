"""
Per-cell laser parameters, supplied by the user as a CSV.

A single VK4 scan may contain several tiled unit cells, each machined with different laser
settings. Those settings are not in the scan or the DXF, so the user provides them in
``CSV/laser_params.csv``. One row per (VK4 file, cell):

    vk4_file,cell,passes,speed,label
    scan_A.vk4,1,20,400,bottom-left
    scan_A.vk4,2,20,500,bottom-right
    scan_A.vk4,3,40,800,top-left
    ...

* ``vk4_file`` — the scan filename (basename; extension optional). ``*`` or blank = a
  default applied to any file/cell without an explicit row.
* ``cell``     — 1-based cell index in tiled-grid order (left->right, bottom->top), as
  enumerated by the registration. Blank/``*`` = applies to every cell of that file.
* ``passes``   — integer laser passes.
* ``speed``    — scan speed (mm/s).
* ``label``    — optional human note (wafer position, etc.).

Lookup precedence for a given (file, cell): exact (file,cell) > (file,*) > (*,cell) > (*,*).
``write_template`` emits a ready-to-edit CSV listing every (file, cell) actually detected.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


CSV_NAME = "laser_params.csv"
TEMPLATE_NAME = "laser_params_TEMPLATE.csv"
FIELDS = ["vk4_file", "cell", "passes", "speed", "label"]


@dataclass
class CellParams:
    passes: int
    speed: float
    label: str = ""

    @property
    def valid(self) -> bool:
        return self.passes and self.passes > 0 and self.speed and self.speed > 0


def _norm_file(name: str) -> str:
    name = (name or "").strip()
    if name in ("", "*"):
        return "*"
    return Path(name).name.lower()


def _norm_cell(v) -> str:
    v = str(v or "").strip()
    return "*" if v in ("", "*") else v


class LaserParamTable:
    """Loaded CSV with (file,cell)-precedence lookup."""

    def __init__(self, rows: dict):
        self._rows = rows            # {(file_key, cell_key): CellParams}

    @classmethod
    def load(cls, csv_path) -> "LaserParamTable":
        rows = {}
        p = Path(csv_path)
        if not p.exists():
            return cls(rows)
        with p.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            lower = {(k or "").strip().lower(): k for k in (reader.fieldnames or [])}
            for raw in reader:
                if not raw:
                    continue
                get = lambda key: raw.get(lower.get(key, key), "")
                if str(get("vk4_file")).strip().startswith("#"):
                    continue                      # comment row
                fk = _norm_file(get("vk4_file"))
                ck = _norm_cell(get("cell"))
                try:
                    passes = int(float(str(get("passes")).strip()))
                except (ValueError, TypeError):
                    passes = 0
                try:
                    speed = float(str(get("speed")).strip())
                except (ValueError, TypeError):
                    speed = float("nan")
                rows[(fk, ck)] = CellParams(passes, speed, str(get("label")).strip())
        return cls(rows)

    def explicit_cells(self, vk4_file: str):
        """Return (sorted explicit cell indices for this file, wildcard_present).

        Only rows whose file key matches THIS file exactly drive the cell count. Global
        wildcard-file rows (``*,cell,...``) apply their *parameters* to every file (see
        ``lookup``) but must not inflate/deflate a specific file's inferred cell count."""
        fk = _norm_file(vk4_file)
        cells, wild = [], False
        for (f, c) in self._rows:
            if f != fk:                       # wildcard-file rows are params-only, not counts
                continue
            if c == "*":
                wild = True
            else:
                try:
                    cells.append(int(c))
                except ValueError:
                    pass
        return sorted(set(cells)), wild

    def n_cells_for(self, vk4_file: str, default: int = 1) -> int:
        """Best guess at how many cells this scan holds, from the CSV."""
        cells, wild = self.explicit_cells(vk4_file)
        if cells:
            return max(cells)
        return default

    def lookup(self, vk4_file: str, cell: int) -> CellParams | None:
        fk = _norm_file(vk4_file)
        ck = str(cell)
        for key in [(fk, ck), (fk, "*"), ("*", ck), ("*", "*")]:
            if key in self._rows:
                return self._rows[key]
        return None

    def __len__(self):
        return len(self._rows)


def write_template(csv_dir, detected: dict, overwrite: bool = False) -> Path:
    """Write a template CSV listing every detected (vk4_file, cell).

    detected : {vk4_filename: n_cells}
    """
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    out = csv_dir / TEMPLATE_NAME
    if out.exists() and not overwrite:
        out = csv_dir / (TEMPLATE_NAME.replace(".csv", "_new.csv"))
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for fname, ncells in sorted(detected.items()):
            for c in range(1, max(1, ncells) + 1):
                w.writerow([fname, c, "", "", ""])
    return out


def prompt_for_params(vk4_file: str, cell: int, n_cells: int) -> CellParams:
    """Interactive fallback when the CSV has no entry (used only on a live terminal)."""
    print(f"\n[laser params] {vk4_file}  cell {cell}/{n_cells} — no CSV entry found.")
    try:
        passes = int(input("    passes (int): ").strip())
        speed = float(input("    speed  (mm/s): ").strip())
        label = input("    label (optional): ").strip()
    except (EOFError, ValueError, KeyboardInterrupt):
        print("    -> no valid input; using passes=0 speed=nan (row will be flagged)")
        return CellParams(0, float("nan"), "")
    return CellParams(passes, speed, label)


# --------------------------------------------------------------------------- #
# Per-unit-cell laser parameters for the tiled full-sample workflow (run_sample.py).
# The CSV is a plain GRID in DESIGN/DXF orientation: line r = design row r (top = row 1), column
# c = design col c (left = col 1), every entry a 'P{passes}_S{speed}' label. This is the frame
# registration numbers cells in ((1,1) = the DXF top-left, anchored on the alignment marker).
# NOTE the Keyence scan is X-mirrored vs the design, so this grid is authored as the DXF is
# DRAWN, not as the raw profilometer image looks (there design col 1 appears on the right).
# No header, no index columns, no other layout.
CELL_CSV_NAME = "cell_params.csv"


_PXSY_RE = re.compile(r"P\s*([\d.]+)\s*_?\s*S\s*([\d.]+)", re.IGNORECASE)


def parse_pxsy(text):
    """Parse a 'P{passes}_S{speed}' label -> (passes:int, speed:float, label:str) or None."""
    m = _PXSY_RE.search(str(text))
    if not m:
        return None
    return int(float(m.group(1))), float(m.group(2)), str(text).strip()


def load_cell_params(csv_path) -> dict:
    """Load a (row, col) -> CellParams map from the cell-parameter GRID.

    Grid position is the DESIGN/DXF cell position: line ``r`` (1-based, top first) is design row r
    and column ``c`` (1-based, left first) is design col c, so entry (r, c) holds that cell's
    ``P{passes}_S{speed}`` laser parameters. This matches how registration numbers cells
    ((1,1) = the DXF top-left). The raw Keyence image is X-mirrored vs the design, so the grid is
    authored as the DXF is drawn, NOT as the scan looks. There is no header, no index columns and
    no other accepted layout. Blank cells are skipped; a blank line still advances the row index so
    an intentional gap keeps later cells on their true row numbers."""
    out = {}
    p = Path(csv_path)
    if not p.exists():
        return out
    with p.open(newline="", encoding="utf-8-sig") as fh:
        for r_idx, row in enumerate(csv.reader(fh), start=1):
            for c_idx, val in enumerate(row, start=1):
                parsed = parse_pxsy(val) if str(val).strip() else None
                if parsed:
                    out[(r_idx, c_idx)] = CellParams(parsed[0], parsed[1], parsed[2])
    return out


if __name__ == "__main__":       # pragma: no cover
    import sys
    t = LaserParamTable.load(sys.argv[1] if len(sys.argv) > 1 else CSV_NAME)
    print(f"loaded {len(t)} rows")
    for f, c in [("scan_A.vk4", 1), ("scan_A.vk4", 3), ("other.vk4", 1)]:
        print(f, c, "->", t.lookup(f, c))
