"""
Single source of truth for figure typography.

Every figure writer in the project (``run_sample``, ``report``, ``row_report``, ``calibrate_depth``,
``extract``, ``synth``, ``porosity_table``) reads its type sizes from the named roles below, so a
presentation-size change is one edit here instead of ~130 scattered literals -- which is what the
two earlier typography passes had to do by hand.

The base numbers are the sizes the 2026-07-27 pass settled on (axes title 16; axis labels, tick
labels and legend 14); ``BUMP`` is added to all of them. Raise BUMP to make the whole output set
bigger, and nothing can be missed. Note that the depth-calibration figures used to sit 1-2 pt below
the rest (labels 13, ticks 12); they now share this scale, so they grow slightly more than BUMP.
"""
from __future__ import annotations

BUMP = 3                     # points added to every base size below

TITLE = 16 + BUMP            # 19  axes title
TITLE_SM = 13 + BUMP         # 16  subplot title on a narrow panel (a long band label would overrun)
SUPTITLE = 16 + BUMP         # 19  figure suptitle
HEADLINE = 17 + BUMP         # 20  bold text-block heading on the cell/row report pages
LABEL = 14 + BUMP            # 17  axis labels, colorbar labels
TICK = 14 + BUMP             # 17  tick labels (axes and colorbars)
LEGEND = 14 + BUMP           # 17  legends with room to spare
LEGEND_SM = 11 + BUMP        # 14  dense legends: one entry per overlaid curve / per band
ANNOT = 11 + BUMP            # 14  in-axes point callouts (P80_S100, cell medians, ...)
ANNOT_SM = 10 + BUMP         # 13  the densest in-axes text (per-pin / per-array labels)
OVERLAY = 15 + BUMP          # 18  bold text drawn over an image (cell ids, snapshot panel names)
TABLE = 12 + BUMP            # 15  matplotlib table cells
NOTE = 12 + BUMP             # 15  in-axes notes (design-target caption, unit callouts)

# rc context for figures that would otherwise inherit matplotlib's defaults. Applied with
# ``plt.rc_context(PLOT_RC)`` around a whole figure suite so no default-sized element is missed.
PLOT_RC = {"font.size": LABEL, "axes.titlesize": TITLE, "axes.labelsize": LABEL,
           "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": LEGEND,
           "figure.titlesize": SUPTITLE}
