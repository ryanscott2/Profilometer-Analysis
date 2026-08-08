"""Porosity of the three pin families in a SQUARE vs a HEXAGONAL (triangular) lattice.

Everything is derived from geometry -- nothing is measured or hard-coded:

    square      one pin per p x p cell            -> solid = pi*d^2 / (4*p^2)
    hexagonal   one pin per (sqrt(3)/2)*p^2 cell  -> solid = pi*d^2 / (2*sqrt(3)*p^2)

with ``p`` the nearest-neighbour pitch (identical in both lattices, so the two columns are a
like-for-like comparison) and ``d`` the drawn pin diameter. Porosity is the open (coolant) area
fraction, phi = 1 - solid. The hexagonal cell is sqrt(3)/2 ~ 0.866 of the square one, so the same
pitch packs 2/sqrt(3) ~ 1.155x the pins and phi drops by a few points.

Rendered in the house cell-report table style (grey bold header, 12 pt, auto column widths).

    python porosity_table.py [--out porosity_square_vs_hex.png]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as fs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repo root: modules live in python/, data sits beside it

# (label, drawn diameter um, nearest-neighbour pitch um) -- the three fabricated pin families.
FAMILIES = [("D50 P100", 50.0, 100.0),
            ("D100 P150", 100.0, 150.0),
            ("D300 P350", 300.0, 350.0)]

COL_LABELS = ["array (D/P)", "Φ square", "Φ hex", "ΔΦ (pts)", "pins/mm² sq", "pins/mm² hex"]


def porosity(d_um: float, p_um: float) -> tuple[float, float]:
    """Open-area fraction (square, hexagonal) for diameter ``d_um`` at pitch ``p_um``."""
    pin = math.pi * d_um ** 2 / 4.0
    return 1.0 - pin / (p_um ** 2), 1.0 - pin / (math.sqrt(3) / 2.0 * p_um ** 2)


def pin_density(p_um: float) -> tuple[float, float]:
    """Pins per mm² (square, hexagonal) at pitch ``p_um``."""
    cell_mm2 = (p_um / 1000.0) ** 2
    return 1.0 / cell_mm2, 1.0 / (math.sqrt(3) / 2.0 * cell_mm2)


def build_rows():
    rows = []
    for label, d, p in FAMILIES:
        sq, hexa = porosity(d, p)
        n_sq, n_hex = pin_density(p)
        rows.append([label, f"{100 * sq:.1f}%", f"{100 * hexa:.1f}%",
                     f"−{100 * (sq - hexa):.1f}", f"{n_sq:.0f}", f"{n_hex:.0f}"])
    return rows


def render(path: Path) -> Path:
    rows = build_rows()
    fig = plt.figure(figsize=(11, 1.15 * (len(rows) + 1) + 1.4))
    ax = fig.add_subplot(111); ax.axis("off")
    ax.text(0.0, 1.0, "Porosity — square vs hexagonal lattice at the same pitch",
            transform=ax.transAxes, va="top", fontsize=fs.HEADLINE, weight="bold")
    ax.text(0.0, 0.86, "Φ = open (coolant) area fraction from drawn geometry; the hexagonal cell is "
            "0.866× the square one, so the same pitch packs 1.155× the pins.",
            transform=ax.transAxes, va="top", fontsize=fs.NOTE, style="italic", color="0.3")
    _th = min(0.82, 0.16 * (len(rows) + 1))
    tbl = ax.table(cellText=rows, colLabels=COL_LABELS, cellLoc="center",
                   bbox=[0.0, 0.74 - _th, 1.0, _th])
    tbl.auto_set_font_size(False); tbl.set_fontsize(fs.TABLE)
    tbl.auto_set_column_width(col=list(range(len(COL_LABELS))))
    for c in range(len(COL_LABELS)):
        tbl[0, c].set_facecolor("#dddddd"); tbl[0, c].set_text_props(weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "porosity_square_vs_hex.png"))
    a = ap.parse_args(argv)
    for label, d, p in FAMILIES:                          # same numbers to stdout, for the record
        sq, hexa = porosity(d, p)                         # ASCII only: consoles here are cp1252
        print(f"{label:<10} D{d:g} @ {p:g} um   phi square {100 * sq:5.1f}%   "
              f"phi hex {100 * hexa:5.1f}%   delta {100 * (sq - hexa):4.1f} pts")
    print(f"Wrote {render(Path(a.out))}")


if __name__ == "__main__":
    main()
