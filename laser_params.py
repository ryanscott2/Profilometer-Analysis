"""
Per-unit-cell laser parameters for the tiled full-sample workflow (run_sample.py).

An assembled scan holds several tiled unit cells, each machined with different laser settings
that are not in the scan or the DXF, so the user supplies them in ``CSV/cell_params.csv`` -- a
plain GRID in DESIGN/DXF orientation: line ``r`` (top = row 1) is design row r, column ``c``
(left = col 1) is design col c, and every entry is a ``P{passes}_S{speed}`` label. That is the
frame registration numbers cells in ((1,1) = the DXF top-left, anchored on the alignment marker).
NOTE the Keyence scan is X-mirrored vs the design, so the grid is authored as the DXF is DRAWN,
not as the raw profilometer image looks. No header, no index columns, no other layout.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

CELL_CSV_NAME = "cell_params.csv"


@dataclass
class CellParams:
    passes: int
    speed: float
    label: str = ""

    @property
    def valid(self) -> bool:
        return self.passes and self.passes > 0 and self.speed and self.speed > 0


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
