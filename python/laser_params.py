"""
Per-unit-cell laser parameters for the tiled full-sample workflow (run_sample.py).

An assembled scan holds several tiled unit cells, each machined with different laser settings
that are not in the scan or the DXF, so the user supplies them in ``csv/cell_params.csv`` -- a
plain GRID in DESIGN/DXF orientation: line ``r`` (top = row 1) is design row r, column ``c``
(left = col 1) is design col c, and every entry is a ``P{passes}_S{speed}`` label. That is the
frame registration numbers cells in ((1,1) = the DXF top-left, anchored on the alignment marker).
NOTE the Keyence scan is X-mirrored vs the design, so the grid is authored as the DXF is DRAWN,
not as the raw profilometer image looks. No header, no index columns, no other layout.
"""
from __future__ import annotations

import csv
import math
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


_POSITIVE_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PXSY_RE = re.compile(
    rf"P\s*(\d+)\s*_?\s*S\s*({_POSITIVE_NUMBER})", re.IGNORECASE)


def parse_pxsy(text):
    """Parse a 'P{passes}_S{speed}' label -> (passes:int, speed:float, label:str) or None."""
    label = str(text).strip()
    m = _PXSY_RE.fullmatch(label)
    if not m:
        return None
    passes, speed = int(m.group(1)), float(m.group(2))
    if passes <= 0 or not math.isfinite(speed) or speed <= 0:
        return None
    return passes, speed, label


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
                if not str(val).strip():
                    continue
                parsed = parse_pxsy(val)
                if not parsed:
                    raise ValueError(
                        f"{p}: invalid laser label at row {r_idx}, column {c_idx}: {val!r}; "
                        "expected P{positive integer passes}_S{positive finite speed}")
                out[(r_idx, c_idx)] = CellParams(parsed[0], parsed[1], parsed[2])
    return out
