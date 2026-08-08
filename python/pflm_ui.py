"""
PFLM sample-tester UI — a simple Tkinter front-end for ``run_sample.py``.

Keep a library of samples (each = a DXF + a VK4 tile folder + a cell_params grid), switch
between them, edit the laser-parameter grid, run/stop the tiled analysis while watching the
console live, browse the Results folder + preview figures, and export a zip of ``figures/``.
Each run writes under ``results/<sample name>/`` (a sample must be selected to run), so different
datasets never overwrite one another.

Drag-and-drop uses ``tkinterdnd2`` if installed (``pip install tkinterdnd2``); otherwise use the
Browse buttons. Image preview uses Pillow if installed, else Tk's built-in PNG support.

Run:  python pflm_ui.py
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# The wafer-row vocabulary (filename grammar, wafer map, run plan) lives in wafer_map.py, which is
# deliberately stdlib-only so importing it here keeps the UI's startup free of numpy/pandas/
# matplotlib. _safe_name is shared from there so the UI and run_row.py cannot drift apart on what a
# results folder is called.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wafer_map import (safe_name as _safe_name, parse_sample_id, group_snapshots,  # noqa: E402
                       read_wafer_map, rows_present, row_out_name, date_tag_from_names,
                       DEFAULT_MAP_NAME)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repo root: modules live in python/, data sits beside it
SAMPLES_JSON = ROOT / ".ui_samples.json"
WORKSPACE = ROOT / ".ui_workspace"
DEF_OUT = ROOT / "results"
IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".bmp")
TEXT_EXT = (".txt", ".csv", ".log", ".json")

# Depth-calibration tool (calibrate_depth.py). Keep these in sync with that module's OUT_NAME /
# MEAS_REL: it pools the per-sample legacy CSVs and writes the cross-sample analysis here.
CAL_SCRIPT = "calibrate_depth.py"
ROW_SCRIPT = "run_row.py"        # wafer-row batch driver (run_row.py --row N)
ROW_FIGURES_DIR = "row_figures"  # keep in sync with run_row.ROW_FIGURES_DIR
CAL_OUT_NAME = "etch depth"
MEAS_REL = Path("legacy") / "measurements.csv"

# Prefilled band definitions (one band per line: min_Ø, max_Ø, pitch in µm). Matches the provided
# 4x4 single-cell DXF: Ø 50–67.5 µm @100 µm pitch and Ø 100–125 µm @150 µm pitch. Editable; blank
# (or comments only) tells calibrate_depth.py to fall back to the measurements' own 'band' column.
DEFAULT_BAND_DEFS = ("50, 67.5, 100\n"
                     "100, 125, 150\n")

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND = True
except Exception:                                        # pragma: no cover - optional dep
    _DND = False

try:
    from PIL import Image, ImageTk
    _PIL = True
except Exception:                                        # pragma: no cover - optional dep
    _PIL = False


def _sample_name_collision(name, existing_names):
    """Return the existing sample whose Windows output folder collides with ``name``."""
    target = _safe_name(name).casefold()
    return next((other for other in existing_names
                 if other != name and _safe_name(other).casefold() == target), None)


def _parse_drop(data: str):
    """Parse a Tk DnD drop payload into paths (handles {braced paths with spaces})."""
    out, buf, brace = [], "", False
    for ch in data:
        if ch == "{":
            brace, buf = True, ""
        elif ch == "}":
            brace = False
            out.append(buf); buf = ""
        elif ch == " " and not brace:
            if buf:
                out.append(buf); buf = ""
        else:
            buf += ch
    if buf:
        out.append(buf)
    return [p for p in out if p]


# Multi-snapshot mode: a dataset folder can hold either a ``_Y{n}_X{m}`` tile raster (the tiled
# full-sample workflow) OR a set of DISJOINT snapshots -- independent crops of ONE uniform cell
# (e.g. a "Center" and a "TopLeft"). These mirror ``run_sample._RASTER_RE`` / ``snapshots_from_dir``
# so the UI can show which mode a folder will run in and route the run command accordingly.
_RASTER_TILE_RE = re.compile(r"_Y\d+_X\d+$", re.IGNORECASE)


def _snapshot_label(stem):
    """Snapshot label from a filename stem: the trailing ``_``-token (e.g. 'Center', 'TopLeft')."""
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def _wafer_cell_id(stem):
    """``(col, row)`` from a VK4 stem's ``_{col}{row}_`` token, or None. Thin wrapper over
    ``wafer_map.parse_sample_id`` so the UI and ``run_row.py`` agree on the grammar exactly."""
    got = parse_sample_id(stem)
    return None if got is None else (got[0], got[1])


def _classify_vk4_folder(vk4_dir):
    """``(mode, labels)`` for a VK4 folder.

    ``'raster'``   -- holds a ``_Y{n}_X{m}`` tile grid (the assembled full-sample workflow).
    ``'row'``      -- a FLAT MULTI-SAMPLE wafer folder: the plain ``*.vk4`` carry ``_{col}{row}_``
                      tokens for MORE THAN ONE distinct (col, row), i.e. several different samples
                      live here (-> run_row.py). Also detected one level down, so pointing at the
                      PARENT of several per-geometry folders works.
    ``'snapshots'``-- disjoint crops of ONE uniform cell (-> run_sample --snapshots).

    The ``row`` test deliberately runs BEFORE the label-collision fallback below: on a flat wafer
    folder every sample has a ``Center``, so that fallback would otherwise fire and route every
    sample of every row into a single ``analyze_multi_snapshot`` call, which averages them as if
    they were replicate views of one cell -- silently, with no error.

    ``len(cells) > 1``, not ``>= 1``: a single-sample folder that merely carries a CR token is
    today's ``snapshots`` mode and must keep behaving identically."""
    vks = sorted(Path(vk4_dir).glob("*.vk4"))
    if vks and not any(_RASTER_TILE_RE.search(p.stem) for p in vks):
        cells = {c for c in (_wafer_cell_id(p.stem) for p in vks) if c is not None}
        if len(cells) > 1:
            return "row", [f"c{c}r{r}" for c, r in sorted(cells)]
        labels = [_snapshot_label(p.stem) for p in vks]
        if len(set(labels)) != len(labels):                 # collision -> disambiguate, never merge
            labels = [p.stem for p in vks]
        return "snapshots", labels
    if not vks:                                  # maybe a PARENT of per-geometry sample folders
        deep = sorted(Path(vk4_dir).rglob("*.vk4"))
        cells = {c for c in (_wafer_cell_id(p.stem) for p in deep) if c is not None}
        if len(cells) > 1:
            return "row", [f"c{c}r{r}" for c, r in sorted(cells)]
    return "raster", []


def _wafer_rows_in_dir(vk4_dir):
    """``(rows_present, n_samples, snapshot_labels, n_unparsed)`` for a wafer-row folder."""
    p = Path(vk4_dir)
    vks = sorted(p.glob("*.vk4")) or sorted(p.rglob("*.vk4"))
    cells, labels, unparsed = set(), set(), 0
    for f in vks:
        got = parse_sample_id(f.stem)
        if got is None:
            unparsed += 1
            continue
        cells.add((got[0], got[1]))
        labels.add(got[2])
    return sorted({r for _c, r in cells}), len(cells), sorted(labels), unparsed


def _first_ps_label(text):
    """First ``P{passes}_S{speed}`` dose token in the params text (the shared dose for a
    multi-snapshot dataset), or ``''`` if none."""
    m = re.search(r"P\d+_S\d+(?:\.\d+)?", text or "")
    return m.group(0) if m else ""


class App:
    def __init__(self, root):
        self.root = root
        root.title("PFLM sample tester")
        root.geometry("1920x1080")
        self._drawer_open = False            # left Inputs drawer starts collapsed (big preview)
        self.proc = None
        self.cal_proc = None                # depth-calibration subprocess (separate from a run)
        self.cal_cell_specs = {}            # sample folder name -> inline cell-id filter spec ("" = all cells)
        self.q = queue.Queue()
        self.samples = self._load_samples()
        self.cur = {"dxf": "", "vk4_dir": "", "wafer_map": ""}
        self._row_cfg = {}                  # wafer-row mode: resolved map path + date tag
        self._preview_img = None            # keep a ref so Tk doesn't GC it
        self._preview_path = None
        self._tree_paths = {}
        self._build()
        self._refresh_sample_combo()
        self._load_working_tree_defaults()
        self._log(f"drag-and-drop: {'on' if _DND else 'off (pip install tkinterdnd2)'}   "
                  f"image preview: {'Pillow' if _PIL else 'Tk PNG only'}\n")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._drain_console()
        self._refresh_results()

    # ------------------------------------------------------------ persistence #
    def _load_samples(self):
        if SAMPLES_JSON.exists():
            try:
                return json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_samples(self):
        try:
            SAMPLES_JSON.write_text(json.dumps(self.samples, indent=2), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _load_working_tree_defaults(self):
        """Pre-fill from the repo's dxf/ vk4/ csv/ so the currently-loaded sample runs at once."""
        dxfs = sorted((ROOT / "dxf").glob("*.dxf")) if (ROOT / "dxf").is_dir() else []
        if dxfs:
            self._set_dxf(dxfs[0])
        if (ROOT / "vk4").is_dir() and any((ROOT / "vk4").glob("*.vk4")):
            self._set_vk4_dir(ROOT / "vk4")
        csv = ROOT / "csv" / "cell_params.csv"
        if csv.exists():
            self.csv_text.delete("1.0", "end")
            self.csv_text.insert("1.0", csv.read_text(encoding="utf-8-sig"))
        rad = ROOT / "csv" / "radial_sets.csv"
        if rad.exists():
            self.radial_text.delete("1.0", "end")
            self.radial_text.insert("1.0", rad.read_text(encoding="utf-8-sig"))
        # band definitions for the depth calibration: from csv/band_defs.csv if present, else default
        bands = ROOT / "csv" / "band_defs.csv"
        self.cal_bands_text.delete("1.0", "end")
        self.cal_bands_text.insert("1.0", bands.read_text(encoding="utf-8-sig")
                                   if bands.exists() else DEFAULT_BAND_DEFS)

    # ------------------------------------------------------------------ theme #
    def _apply_theme(self):
        """Soft-gray dark theme (Claude-Code style). ttk widgets via a 'clam' Style; the tk widgets
        (Text/Listbox/Canvas) are coloured directly where they are created."""
        P = self.pal = {
            "bg": "#1b1b1b", "toolbar": "#232323", "panel": "#212121", "drawer": "#232323",
            "border": "#2e2e2e", "input": "#191919", "inputborder": "#383838",
            "text": "#cfccca", "muted": "#8f8c88", "help": "#6f6f6f",
            "btn": "#2a2a2a", "btnborder": "#3a3a3a", "btnactive": "#333333",
            "accent": "#4f7ea8", "accent_text": "#0c2334", "accent_active": "#5b8cb6",
            "sel_bg": "#26323c", "sel_fg": "#cfe0ea", "scrim": "#0f0f0f",
            "preview_bg": "#151515", "console_bg": "#141414", "console_fg": "#8fae8f",
            "status_fg": "#7a9a7a",
        }
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=P["bg"], foreground=P["text"], fieldbackground=P["input"],
                     bordercolor=P["border"], lightcolor=P["border"], darkcolor=P["border"],
                     insertcolor=P["text"], font=("Segoe UI", 9))
        st.configure("TFrame", background=P["bg"])
        st.configure("Card.TFrame", background=P["panel"])
        st.configure("Drawer.TFrame", background=P["drawer"])
        st.configure("Toolbar.TFrame", background=P["toolbar"])
        st.configure("TLabel", background=P["bg"], foreground=P["text"])
        st.configure("Card.TLabel", background=P["panel"], foreground=P["text"])
        st.configure("Help.TLabel", background=P["panel"], foreground=P["help"], font=("Segoe UI", 8))
        st.configure("Drawer.TLabel", background=P["drawer"], foreground=P["muted"])
        st.configure("Brand.TLabel", background=P["toolbar"], foreground=P["muted"],
                     font=("Segoe UI", 11))
        st.configure("Status.TLabel", background=P["input"], foreground=P["status_fg"],
                     padding=(10, 4))
        st.configure("Card.TLabelframe", background=P["panel"], bordercolor=P["border"],
                     relief="solid", borderwidth=1)
        st.configure("Card.TLabelframe.Label", background=P["panel"], foreground=P["muted"],
                     font=("Segoe UI", 9))
        for name in ("TButton", "Tool.TButton"):
            st.configure(name, background=P["btn"], foreground=P["text"], bordercolor=P["btnborder"],
                         relief="flat", padding=(10, 6), focuscolor=P["btn"])
            st.map(name, background=[("active", P["btnactive"]), ("disabled", P["panel"])],
                   foreground=[("disabled", P["help"])])
        st.configure("Accent.TButton", background=P["accent"], foreground=P["accent_text"],
                     bordercolor=P["accent"], relief="flat", padding=(16, 6),
                     font=("Segoe UI", 9, "bold"), focuscolor=P["accent"])
        st.map("Accent.TButton", background=[("active", P["accent_active"])])
        st.configure("TCombobox", fieldbackground=P["input"], background=P["btn"],
                     foreground=P["text"], arrowcolor=P["muted"], bordercolor=P["inputborder"],
                     padding=(6, 4))
        st.map("TCombobox", fieldbackground=[("readonly", P["input"])],
               foreground=[("readonly", P["text"])], background=[("readonly", P["btn"])])
        st.configure("TEntry", fieldbackground=P["input"], foreground=P["text"],
                     bordercolor=P["inputborder"], padding=(4, 3))
        st.configure("TCheckbutton", background=P["panel"], foreground=P["text"])
        st.map("TCheckbutton", background=[("active", P["panel"])])
        st.configure("Treeview", background=P["input"], fieldbackground=P["input"],
                     foreground=P["text"], bordercolor=P["border"], borderwidth=0)
        st.map("Treeview", background=[("selected", P["sel_bg"])],
               foreground=[("selected", P["sel_fg"])])
        st.configure("Treeview.Heading", background=P["panel"], foreground=P["muted"],
                     relief="flat", bordercolor=P["border"])
        st.map("Treeview.Heading", background=[("active", P["btn"])])
        for sb in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
            st.configure(sb, background=P["btn"], troughcolor=P["bg"], bordercolor=P["border"],
                         arrowcolor=P["muted"])
            st.map(sb, background=[("active", P["btnactive"])])
        self.root.option_add("*TCombobox*Listbox.background", P["input"])
        self.root.option_add("*TCombobox*Listbox.foreground", P["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", P["sel_bg"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", P["sel_fg"])

    def _mono_text(self, parent, **kw):
        """A dark-themed tk.Text with the shared input colouring (helper for the many editors)."""
        P = self.pal
        opts = dict(background=P["input"], foreground=P["text"], insertbackground=P["text"],
                    relief="flat", borderwidth=0, highlightthickness=1,
                    highlightbackground=P["inputborder"], highlightcolor=P["inputborder"])
        opts.update(kw)
        return tk.Text(parent, **opts)

    # ------------------------------------------------------------------ layout #
    def _build(self):
        root = self.root
        self._apply_theme()
        P = self.pal
        root.configure(bg=P["bg"])
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        # ---------- TOP TOOLBAR: sample switch + primary actions (always reachable) ----------
        tbar = ttk.Frame(root, style="Toolbar.TFrame", padding=(10, 7))
        tbar.grid(row=0, column=0, sticky="ew")
        self.drawer_btn = ttk.Button(tbar, text="☰  Inputs", width=11, style="Tool.TButton",
                                     command=self._toggle_drawer)
        self.drawer_btn.pack(side="left")
        ttk.Label(tbar, text="PFLM", style="Brand.TLabel").pack(side="left", padx=(12, 14))
        self.sample_var = tk.StringVar()
        self.sample_combo = ttk.Combobox(tbar, textvariable=self.sample_var, state="readonly",
                                         width=32)
        self.sample_combo.pack(side="left")
        self.sample_combo.bind("<<ComboboxSelected>>",
                               lambda e: self._load_sample(self.sample_var.get()))
        # Wafer-row mode: a mode chip plus a row picker, both hidden unless the selected VK4 folder
        # classifies as 'row'. Together with Run these are the whole two-click row workflow.
        self.mode_chip = ttk.Label(tbar, text="", style="Status.TLabel")
        self.row_var = tk.StringVar()
        self.row_combo = ttk.Combobox(tbar, textvariable=self.row_var, state="readonly", width=9)
        self.run_btn = ttk.Button(tbar, text="▶  Run", style="Accent.TButton",
                                  command=self._toggle_run)
        self.run_btn.pack(side="right")
        ttk.Button(tbar, text="Export .zip", style="Tool.TButton",
                   command=self._export_zip).pack(side="right", padx=(0, 8))
        self.status = ttk.Label(tbar, text="idle", style="Status.TLabel", anchor="w")
        self.status.pack(side="right", padx=(0, 10))

        # ---------- BODY: workspace (preview + console)  |  right rail ----------
        body = ttk.Frame(root, style="TFrame", padding=(10, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=400)

        mid = ttk.Frame(body, style="TFrame")
        mid.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        mid.columnconfigure(0, weight=1); mid.rowconfigure(0, weight=3); mid.rowconfigure(1, weight=1)
        self._mid = mid

        prevf = ttk.LabelFrame(mid, text=" Preview ", style="Card.TLabelframe", padding=6)
        prevf.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        prevf.bind("<Configure>", self._fit_preview_169)
        # host kept at 16:9 (letterboxed, centred in prevf); canvas + text overlay it
        self.prev_host = tk.Frame(prevf, background=P["preview_bg"])
        self.prev_host.place(relx=0.5, rely=0.5, anchor="center", width=320, height=180)
        self.preview = tk.Canvas(self.prev_host, background=P["preview_bg"], highlightthickness=0)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Configure>", lambda e: self._reshow_image())
        self.text_frame = tk.Frame(self.prev_host, background=P["preview_bg"])
        self.text_frame.columnconfigure(0, weight=1); self.text_frame.rowconfigure(0, weight=1)
        self.preview_text = self._mono_text(self.text_frame, wrap="none", font=("Consolas", 10),
                                            background=P["console_bg"], highlightthickness=0)
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        tvs = ttk.Scrollbar(self.text_frame, orient="vertical", command=self.preview_text.yview)
        tvs.grid(row=0, column=1, sticky="ns")
        ths = ttk.Scrollbar(self.text_frame, orient="horizontal", command=self.preview_text.xview)
        ths.grid(row=1, column=0, sticky="ew")
        self.preview_text.config(yscrollcommand=tvs.set, xscrollcommand=ths.set)

        consf = ttk.LabelFrame(mid, text=" Console ", style="Card.TLabelframe", padding=6)
        consf.grid(row=1, column=0, sticky="nsew")
        consf.columnconfigure(0, weight=1); consf.rowconfigure(0, weight=1)
        self.console = self._mono_text(consf, wrap="none", background=P["console_bg"],
                                       foreground=P["console_fg"], font=("Consolas", 9), height=8,
                                       highlightthickness=0)
        self.console.grid(row=0, column=0, sticky="nsew")
        kv = ttk.Scrollbar(consf, orient="vertical", command=self.console.yview)
        kv.grid(row=0, column=1, sticky="ns")
        kh = ttk.Scrollbar(consf, orient="horizontal", command=self.console.xview)
        kh.grid(row=1, column=0, sticky="ew")
        self.console.config(yscrollcommand=kv.set, xscrollcommand=kh.set)

        # ---------- RIGHT RAIL: results browser + radial sets + depth calibration ----------
        right = ttk.Frame(body, style="TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1); right.rowconfigure(0, weight=1)

        resf = ttk.LabelFrame(right, text=" Results folder ", style="Card.TLabelframe", padding=6)
        resf.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        resf.columnconfigure(0, weight=1); resf.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(resf, show="tree")
        self.tree.grid(row=0, column=0, sticky="nsew")
        tb = ttk.Scrollbar(resf, orient="vertical", command=self.tree.yview)
        tb.grid(row=0, column=1, sticky="ns")
        thb = ttk.Scrollbar(resf, orient="horizontal", command=self.tree.xview)
        thb.grid(row=1, column=0, sticky="ew")
        self.tree.config(yscrollcommand=tb.set, xscrollcommand=thb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        ttk.Button(resf, text="Refresh", style="Tool.TButton",
                   command=self._refresh_results).grid(row=2, column=0, sticky="ew", pady=(6, 0))

        radf = ttk.LabelFrame(right, text=" Radial-average sets ", style="Card.TLabelframe", padding=6)
        radf.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        radf.columnconfigure(0, weight=1); radf.rowconfigure(0, weight=1)
        self.radial_text = self._mono_text(radf, height=5, width=32, wrap="none",
                                           font=("Consolas", 9), undo=True)
        self.radial_text.grid(row=0, column=0, sticky="nsew")
        rvs = ttk.Scrollbar(radf, orient="vertical", command=self.radial_text.yview)
        rvs.grid(row=0, column=1, sticky="ns"); self.radial_text.config(yscrollcommand=rvs.set)
        ttk.Label(radf, text="one overlay set per line · blank = all parameters",
                  style="Help.TLabel", wraplength=350).grid(row=1, column=0, columnspan=2, sticky="w",
                                                            pady=(4, 0))

        # ---------- depth calibration (pool completed samples, post-hoc) ----------
        # Shells out to calibrate_depth.py on the samples selected here; output streams to the
        # console and the report/figures land under results/_depth_calibration (browsable above).
        calf = ttk.LabelFrame(right, text=" Depth calibration ", style="Card.TLabelframe", padding=6)
        calf.grid(row=2, column=0, sticky="ew")
        calf.columnconfigure(0, weight=1)
        clf = ttk.Frame(calf, style="Card.TFrame"); clf.grid(row=0, column=0, columnspan=2, sticky="nsew")
        clf.columnconfigure(0, weight=1); clf.rowconfigure(0, weight=1)
        # Two-column table: sample name + an inline per-sample cell filter. Native extended
        # selection (Ctrl/Shift-click) still picks which samples to pool (as the old listbox did);
        # double-clicking a row's "cells" column edits that sample's cell_id selection in place.
        self.cal_list = ttk.Treeview(clf, height=5, columns=("cells",), selectmode="extended",
                                     show="tree headings")
        self.cal_list.heading("#0", text="sample")
        self.cal_list.heading("cells", text="cells")
        self.cal_list.column("#0", width=200, minwidth=90, anchor="w", stretch=True)
        self.cal_list.column("cells", width=96, minwidth=60, anchor="w", stretch=False)
        self.cal_list.grid(row=0, column=0, sticky="nsew")
        cls = ttk.Scrollbar(clf, orient="vertical", command=self.cal_list.yview)
        cls.grid(row=0, column=1, sticky="ns"); self.cal_list.config(yscrollcommand=cls.set)
        self.cal_list.bind("<Double-1>", self._edit_cal_cells)
        self._cal_cell_editor = None            # active inline Entry over the 'cells' column, if any
        ttk.Label(calf, text="Ctrl/Shift-click to subset · double-click ‘cells’ to filter",
                  style="Help.TLabel", wraplength=360).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 2))
        ttk.Label(calf, text="one band per line: min Ø, max Ø, pitch (µm)",
                  style="Help.TLabel", wraplength=360).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 2))
        bdf = ttk.Frame(calf, style="Card.TFrame"); bdf.grid(row=3, column=0, columnspan=2, sticky="ew")
        bdf.columnconfigure(0, weight=1)
        self.cal_bands_text = self._mono_text(bdf, height=3, width=30, wrap="none",
                                              font=("Consolas", 9), undo=True)
        self.cal_bands_text.grid(row=0, column=0, sticky="nsew")
        bds = ttk.Scrollbar(bdf, orient="vertical", command=self.cal_bands_text.yview)
        bds.grid(row=0, column=1, sticky="ns"); self.cal_bands_text.config(yscrollcommand=bds.set)
        tgtf = ttk.Frame(calf, style="Card.TFrame"); tgtf.grid(row=4, column=0, columnspan=2,
                                                               sticky="ew", pady=(6, 2))
        ttk.Label(tgtf, text="target depth (µm):", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.cal_target = tk.StringVar(value="55")
        ttk.Entry(tgtf, textvariable=self.cal_target, width=16).grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(tgtf, text="(comma-sep OK)", style="Help.TLabel").grid(row=0, column=2, sticky="w", padx=(6, 0))
        self.cal_allow_legacy_qc = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tgtf,
            text="allow legacy files missing QC fields (unsafe)",
            variable=self.cal_allow_legacy_qc,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.cal_btn = ttk.Button(calf, text="Calibrate depth", style="Tool.TButton",
                                  command=self._calibrate_depth)
        self.cal_btn.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self._build_drawer(mid)                 # left Inputs drawer (overlays the preview/console)

    # -------------------------------------------------------- left Inputs drawer #
    def _build_drawer(self, mid):
        """Build the collapsible Inputs drawer + its soft-dim scrim as overlay children of ``mid``
        (the preview/console column), so opening it never covers the right rail. Placed via
        ``_place_drawer`` / hidden via ``_toggle_drawer``; starts collapsed."""
        P = self.pal
        self._scrim = tk.Frame(mid, background=P["scrim"])
        self._scrim.bind("<Button-1>", lambda e: self._toggle_drawer())   # click-away closes it
        dr = self._drawer = tk.Frame(mid, background=P["drawer"], highlightthickness=1,
                                     highlightbackground=P["btnborder"])
        dr.columnconfigure(0, weight=1)

        hdr = ttk.Frame(dr, style="Drawer.TFrame"); hdr.grid(row=0, column=0, sticky="ew",
                                                             padx=10, pady=(9, 4))
        hdr.columnconfigure(0, weight=1)
        ttk.Label(hdr, text="Inputs", style="Drawer.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(hdr, text="◀", width=3, style="Tool.TButton",
                   command=self._toggle_drawer).grid(row=0, column=1, sticky="e")

        lib = ttk.LabelFrame(dr, text=" Samples ", style="Card.TLabelframe", padding=6)
        lib.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 6))
        lib.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(lib, text="Save as…", style="Tool.TButton",
                   command=self._save_as).grid(row=0, column=0, sticky="ew", padx=1)
        ttk.Button(lib, text="Update", style="Tool.TButton",
                   command=self._update_sample).grid(row=0, column=1, sticky="ew", padx=1)
        ttk.Button(lib, text="Delete", style="Tool.TButton",
                   command=self._delete_sample).grid(row=0, column=2, sticky="ew", padx=1)

        dxff = ttk.LabelFrame(dr, text=" DXF file  ·  drag/drop ", style="Card.TLabelframe", padding=6)
        dxff.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6)); dxff.columnconfigure(0, weight=1)
        self.dxf_lbl = tk.Label(dxff, text="(none)", anchor="w", background=P["input"],
                                foreground=P["text"], relief="flat", padx=6, pady=4,
                                highlightthickness=1, highlightbackground=P["inputborder"])
        self.dxf_lbl.grid(row=0, column=0, sticky="ew")
        ttk.Button(dxff, text="Browse…", style="Tool.TButton",
                   command=self._browse_dxf).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._make_drop(self.dxf_lbl, self._on_drop_dxf)

        vk4f = ttk.LabelFrame(dr, text=" VK4 tiles  ·  drop folder or files ",
                              style="Card.TLabelframe", padding=6)
        vk4f.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 6)); dr.rowconfigure(3, weight=1)
        vk4f.columnconfigure(0, weight=1); vk4f.rowconfigure(1, weight=1)
        self.vk4_dir_lbl = tk.Label(vk4f, text="(no folder)", anchor="w", background=P["input"],
                                    foreground=P["text"], relief="flat", padx=6, pady=4,
                                    highlightthickness=1, highlightbackground=P["inputborder"])
        self.vk4_dir_lbl.grid(row=0, column=0, sticky="ew")
        lbf = ttk.Frame(vk4f, style="Card.TFrame"); lbf.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        lbf.columnconfigure(0, weight=1); lbf.rowconfigure(0, weight=1)
        self.vk4_list = tk.Listbox(lbf, height=6, activestyle="none", background=P["input"],
                                   foreground=P["text"], relief="flat", borderwidth=0,
                                   highlightthickness=1, highlightbackground=P["inputborder"],
                                   selectbackground=P["sel_bg"], selectforeground=P["sel_fg"])
        self.vk4_list.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lbf, orient="vertical", command=self.vk4_list.yview)
        sb.grid(row=0, column=1, sticky="ns"); self.vk4_list.config(yscrollcommand=sb.set)
        ttk.Button(vk4f, text="Browse folder…", style="Tool.TButton",
                   command=self._browse_vk4).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self._make_drop(self.vk4_list, self._on_drop_vk4)

        # The wafer map is a large, external, shared, durable artifact that run_row.py reads from
        # disk, so this is a path + Browse (like the DXF block) and NOT a text box -- a text box
        # would fork it into one divergent copy per sample inside .ui_samples.json.
        wmf = ttk.LabelFrame(dr, text=" Wafer map (row mode)  ·  drag/drop ",
                             style="Card.TLabelframe", padding=6)
        wmf.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 6)); wmf.columnconfigure(0, weight=1)
        self.wmap_lbl = tk.Label(wmf, text="(none)", anchor="w", background=P["input"],
                                 foreground=P["text"], relief="flat", padx=6, pady=4,
                                 highlightthickness=1, highlightbackground=P["inputborder"])
        self.wmap_lbl.grid(row=0, column=0, sticky="ew")
        ttk.Button(wmf, text="Browse…", style="Tool.TButton",
                   command=self._browse_wafer_map).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._make_drop(self.wmap_lbl, self._on_drop_wafer_map)

        lpf = ttk.LabelFrame(dr, text=" Laser parameters (cell_params grid) ",
                             style="Card.TLabelframe", padding=6)
        lpf.grid(row=5, column=0, sticky="nsew", padx=10, pady=(0, 10)); dr.rowconfigure(5, weight=1)
        lpf.columnconfigure(0, weight=1); lpf.rowconfigure(0, weight=1)
        self.csv_text = self._mono_text(lpf, height=6, width=34, wrap="none", font=("Consolas", 10),
                                        undo=True)
        self.csv_text.grid(row=0, column=0, sticky="nsew")
        cs = ttk.Scrollbar(lpf, orient="vertical", command=self.csv_text.yview)
        cs.grid(row=0, column=1, sticky="ns"); self.csv_text.config(yscrollcommand=cs.set)
        ttk.Label(lpf, text="grid = design orientation (DXF top-left = line 1, col 1); "
                            "each cell 'P{passes}_S{speed}'",
                  style="Help.TLabel", wraplength=290).grid(row=1, column=0, sticky="w", pady=(4, 0))

    def _place_drawer(self):
        self._scrim.place(x=0, y=0, relwidth=1.0, relheight=1.0)
        self._drawer.place(x=0, y=0, relheight=1.0, width=340)
        self._scrim.lift(); self._drawer.lift()

    def _toggle_drawer(self):
        if self._drawer_open:
            self._drawer.place_forget(); self._scrim.place_forget()
            self._drawer_open = False
            self.drawer_btn.config(text="☰  Inputs")
        else:
            self._place_drawer()
            self._drawer_open = True
            self.drawer_btn.config(text="◀  Inputs")

    def _make_drop(self, widget, handler):
        if _DND:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", lambda e: handler(_parse_drop(e.data)))

    # -------------------------------------------------------------- DXF / VK4 #
    def _set_dxf(self, path):
        self.cur["dxf"] = str(path)
        self.dxf_lbl.config(text=Path(path).name)

    def _browse_dxf(self):
        p = filedialog.askopenfilename(title="Select DXF",
                                       filetypes=[("DXF", "*.dxf"), ("All files", "*.*")])
        if p:
            self._set_dxf(p)

    def _on_drop_dxf(self, paths):
        for p in paths:
            if p.lower().endswith(".dxf"):
                self._set_dxf(p)
                return
        if paths:
            messagebox.showwarning("DXF", "Dropped item is not a .dxf file.")

    def _set_wafer_map(self, path):
        self.cur["wafer_map"] = str(path)
        self.wmap_lbl.config(text=Path(path).name if path else "(none)")

    def _browse_wafer_map(self):
        p = filedialog.askopenfilename(title="Select wafer map CSV",
                                       filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p:
            self._set_wafer_map(p)
            self._set_vk4_dir(self.cur["vk4_dir"]) if self.cur["vk4_dir"] else None

    def _on_drop_wafer_map(self, paths):
        for p in paths:
            if p.lower().endswith(".csv"):
                self._set_wafer_map(p)
                return
        if paths:
            messagebox.showwarning("Wafer map", "Dropped item is not a .csv file.")

    def _find_wafer_map(self, vk4_dir):
        """The wafer map for this folder: an explicit pick wins, else search beside the VK4 folder,
        its parent, then csv/ -- the same order run_row.py uses, so the UI shows what the run
        will actually read."""
        if self.cur.get("wafer_map") and Path(self.cur["wafer_map"]).is_file():
            return Path(self.cur["wafer_map"])
        d = Path(vk4_dir)
        for cand in (d / DEFAULT_MAP_NAME, d.parent / DEFAULT_MAP_NAME,
                     ROOT / "csv" / DEFAULT_MAP_NAME):
            if cand.is_file():
                return cand
        return None

    def _set_vk4_dir(self, d):
        d = Path(d)
        self.cur["vk4_dir"] = str(d)
        self.vk4_dir_lbl.config(text=str(d))
        self.vk4_list.delete(0, tk.END)
        vks = sorted(d.glob("*.vk4")) or sorted(d.rglob("*.vk4"))
        for f in vks:
            self.vk4_list.insert(tk.END, f.name)
        mode, labels = _classify_vk4_folder(d)
        self._row_cfg = {}
        if mode == "row":
            rows, n_samples, snaps, unparsed = _wafer_rows_in_dir(d)
            txt = (f"{d}   ({len(vks)} .vk4 — wafer map: {n_samples} samples, "
                   f"rows {','.join(str(r) for r in rows)}; snapshots: {', '.join(snaps)})")
            if unparsed:
                txt += f"   ⚠ {unparsed} file(s) with no col/row token"
            self.vk4_dir_lbl.config(text=txt)
            self._show_row_controls(rows, d)
        elif mode == "snapshots":                            # disjoint crops -> tiled montage
            self.vk4_dir_lbl.config(
                text=f"{d}   ({len(vks)} .vk4 — snapshot montage: {', '.join(labels)})")
            self._show_row_controls(None, d)
        else:
            self.vk4_dir_lbl.config(text=f"{d}   ({len(vks)} .vk4)")
            self._show_row_controls(None, d)

    def _show_row_controls(self, rows, vk4_dir):
        """Reveal the mode chip + row picker only for a wafer-row folder, and preselect the lowest
        row that is in both the map and the files and has no Results folder yet -- so the common
        case really is 'open the folder, press Run'."""
        if not rows:
            self.mode_chip.pack_forget(); self.row_combo.pack_forget()
            return
        map_path = self._find_wafer_map(vk4_dir)
        mapped = []
        if map_path is not None:
            entries, meta, _probs = read_wafer_map(map_path)
            mapped = [r for r in rows_present(entries) if r in rows]
            self._row_cfg = {"map": str(map_path), "date": meta.get("date", "")}
            self._set_wafer_map(map_path)
        choices = [f"Row {r}" for r in (mapped or rows)]
        self.row_combo["values"] = choices
        if choices:
            names = [f.name for f in Path(vk4_dir).rglob("*.vk4")]
            tag = self._row_cfg.get("date") or date_tag_from_names(names) or ""
            fresh = next((c for c in choices
                          if not (DEF_OUT / _safe_name(
                              row_out_name(tag, int(c.split()[1])))).exists()), choices[0])
            self.row_var.set(fresh if self.row_var.get() not in choices else self.row_var.get())
        self.mode_chip.config(text="wafer row")
        self.mode_chip.pack(side="left", padx=(10, 4))
        self.row_combo.pack(side="left")

    def _browse_vk4(self):
        d = filedialog.askdirectory(title="Select VK4 folder")
        if d:
            self._set_vk4_dir(d)

    def _on_drop_vk4(self, paths):
        if not paths:
            return
        p0 = Path(paths[0])
        self._set_vk4_dir(p0 if p0.is_dir() else p0.parent)

    # ------------------------------------------------------------ sample library #
    def _refresh_sample_combo(self):
        self.sample_combo["values"] = sorted(self.samples)

    def _csv(self):
        return self.csv_text.get("1.0", "end-1c")

    def _radial(self):
        return self.radial_text.get("1.0", "end-1c")

    def _load_sample(self, name):
        s = self.samples.get(name)
        if not s:
            return
        self._set_dxf(s.get("dxf", ""))
        self._set_wafer_map(s.get("wafer_map", ""))          # old samples: absent -> cleared
        if s.get("vk4_dir") and Path(s["vk4_dir"]).is_dir():
            self._set_vk4_dir(s["vk4_dir"])
        self.csv_text.delete("1.0", "end")
        self.csv_text.insert("1.0", s.get("csv_text", ""))
        self.radial_text.delete("1.0", "end")
        self.radial_text.insert("1.0", s.get("radial_text", ""))     # old samples: absent -> empty
        self.status.config(text=f"loaded '{name}'")

    def _snapshot(self):
        return {"dxf": self.cur["dxf"], "vk4_dir": self.cur["vk4_dir"],
                "wafer_map": self.cur.get("wafer_map", ""),
                "csv_text": self._csv(), "radial_text": self._radial()}

    def _save_as(self):
        name = _ask_name(self.root, self.sample_var.get())
        if not name:
            return
        collision = _sample_name_collision(name, self.samples)
        if collision:
            return messagebox.showerror(
                "Sample name collision",
                f"'{name}' and existing sample '{collision}' map to the same Windows Results "
                f"folder ('{_safe_name(name)}'). Choose a distinct name; the run was not saved.")
        self.samples[name] = self._snapshot()
        self._save_samples(); self._refresh_sample_combo(); self.sample_var.set(name)
        self.status.config(text=f"saved '{name}'")

    def _update_sample(self):
        name = self.sample_var.get()
        if not name:
            return self._save_as()
        self.samples[name] = self._snapshot()
        self._save_samples(); self.status.config(text=f"updated '{name}'")

    def _delete_sample(self):
        name = self.sample_var.get()
        if name in self.samples and messagebox.askyesno("Delete", f"Delete sample '{name}'?"):
            del self.samples[name]
            self._save_samples(); self._refresh_sample_combo(); self.sample_var.set("")
            self.status.config(text=f"deleted '{name}'")

    # -------------------------------------------------------------------- run #
    def _dataset_name(self):
        """Folder-safe name for this run's Results subfolder — the sample selected in the UI
        (the Samples dropdown), sanitized for the filesystem. A run requires a selected sample
        (see _toggle_run), so this is never empty at run time."""
        return _safe_name(self.sample_var.get())

    def _dataset_out_dir(self):
        """Per-dataset output root — results/<dataset name>/ — holding this run's figures/ + legacy/."""
        return DEF_OUT / self._dataset_name()

    def _toggle_run(self):
        if self.proc and self.proc.poll() is None:
            self._stop()
            return
        if self.cal_proc and self.cal_proc.poll() is None:
            return messagebox.showerror("Run", "Depth calibration is in progress — wait for it to "
                                               "finish before starting a sample run.")
        vk4_now = self.cur.get("vk4_dir", "")
        if vk4_now and Path(vk4_now).is_dir() and _classify_vk4_folder(vk4_now)[0] == "row":
            return self._start_row_run()
        if not self.sample_var.get().strip():
            return messagebox.showerror(
                "Run", "Select a sample in the Samples dropdown first — its name is used for the "
                "Results subfolder.  Use “Save as…” to name the current setup as a sample.")
        collision = _sample_name_collision(self.sample_var.get(), self.samples)
        if collision:
            return messagebox.showerror(
                "Run", f"This sample collides with '{collision}' at Results\\{self._dataset_name()}. "
                "Rename one sample before running so neither result can overwrite the other.")
        if _safe_name(self.sample_var.get()).casefold() == CAL_OUT_NAME.casefold():
            return messagebox.showerror(
                "Run", f"'{CAL_OUT_NAME}' is reserved for the depth-calibration output folder. "
                f"Rename this sample (e.g. add a date/label) so its Results\\ folder does not collide "
                f"with Results\\{CAL_OUT_NAME} — a collision hides it from calibration and its run "
                f"would overwrite the calibration report.")
        dxf, vk4 = self.cur["dxf"], self.cur["vk4_dir"]
        if not (dxf and Path(dxf).exists()):
            return messagebox.showerror("Run", "Select a DXF file first.")
        if not (vk4 and Path(vk4).is_dir() and any(Path(vk4).glob("*.vk4"))):
            return messagebox.showerror("Run", "Select a VK4 folder that contains .vk4 tiles.")
        WORKSPACE.mkdir(exist_ok=True)
        csv_path = WORKSPACE / "cell_params.csv"
        csv_path.write_text(self._csv(), encoding="utf-8")
        # radial_sets.csv must sit next to cell_params.csv (run_sample reads it from there)
        (WORKSPACE / "radial_sets.csv").write_text(self._radial(), encoding="utf-8")
        # each run writes under results/<dataset name>/ so datasets don't overwrite each other
        # (run_sample clears only this subfolder, leaving other datasets' results intact)
        out_dir = self._dataset_out_dir()
        mode, _labels = _classify_vk4_folder(vk4)
        if mode == "snapshots":
            # Disjoint snapshots of ONE uniform cell -> multi-snapshot montage. Labels come from the
            # filenames; all snapshots share one dose, parsed from the params box (first P#_S#).
            dose = _first_ps_label(self._csv())
            cmd = [sys.executable, "-u", str(HERE / "run_sample.py"),
                   "--snapshots", str(vk4), str(out_dir), str(dxf), dose]
        else:
            cmd = [sys.executable, "-u", str(HERE / "run_sample.py"),
                   str(vk4), str(out_dir), str(dxf), str(csv_path)]
        self.console.delete("1.0", "end")
        self._log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except Exception as e:
            self.proc = None
            return messagebox.showerror("Run", str(e))
        self.run_btn.config(text="■  Stop"); self.status.config(text="running…")
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    # ------------------------------------------------------------- wafer-row run #
    def _row_number(self):
        try:
            return int(self.row_var.get().split()[1])
        except (IndexError, ValueError):
            return 0

    def _row_name(self):
        tag = self._row_cfg.get("date") or date_tag_from_names(
            [f.name for f in Path(self.cur["vk4_dir"]).rglob("*.vk4")]) or ""
        return _safe_name(row_out_name(tag, self._row_number()))

    def _start_row_run(self):
        """Launch run_row.py for the selected wafer row: ONE subprocess for the whole row, reusing
        the existing console/Stop/results plumbing (self.proc, _reader, _drain_console)."""
        n = self._row_number()
        if not n:
            return messagebox.showerror("Run row", "Pick a wafer row in the toolbar first.")
        map_path = self._find_wafer_map(self.cur["vk4_dir"])
        if map_path is None:
            return messagebox.showerror(
                "Run row", f"No {DEFAULT_MAP_NAME} found beside the VK4 folder, in its parent, or "
                f"in CSV\\. Use the “Wafer map” box in the Inputs drawer to pick one.")
        # Pre-flight: six samples is a long run, so fail in the first second, not the tenth minute.
        entries, _meta, problems = read_wafer_map(map_path)
        hard = [p for p in problems if not p.startswith("WARNING")]
        if hard:
            return messagebox.showerror("Wafer map", f"{map_path}\n\n" + "\n".join(hard[:12]))
        if n not in rows_present(entries):
            return messagebox.showerror(
                "Run row", f"The wafer map has no entries for row {n} "
                f"(rows present: {', '.join(str(r) for r in rows_present(entries))}).")
        row_name = self._row_name()
        collision = _sample_name_collision(row_name, self.samples)
        if collision:
            return messagebox.showerror(
                "Run row", f"The row folder Results\\{row_name} collides with sample "
                f"'{collision}'. Rename one so neither can overwrite the other.")
        if row_name.casefold() == CAL_OUT_NAME.casefold():
            return messagebox.showerror(
                "Run row", f"'{CAL_OUT_NAME}' is reserved for the depth-calibration output.")
        out_dir = DEF_OUT / row_name
        if out_dir.is_dir() and ((out_dir / "legacy").is_dir() or (out_dir / "figures").is_dir()
                                 or (out_dir / ".pflm-results.json").is_file()):
            return messagebox.showerror(
                "Run row", f"Results\\{row_name} is an existing SINGLE-sample result, not a wafer-row "
                f"container. Pick a different row name so that result is not put at risk.")
        dxf_dir = str(Path(self.cur["dxf"]).parent) if self.cur.get("dxf") else ""
        cmd = [sys.executable, "-u", str(HERE / ROW_SCRIPT), "--row", str(n),
               "--map", str(map_path), "--vk4", str(self.cur["vk4_dir"]),
               "--out", str(out_dir)]
        if dxf_dir:
            cmd += ["--dxf-dir", dxf_dir]
        self.console.delete("1.0", "end")
        self._log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except Exception as e:
            self.proc = None
            return messagebox.showerror("Run row", str(e))
        self.run_btn.config(text="■  Stop"); self.status.config(text=f"running row {n}…")
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        try:
            for line in proc.stdout:
                self.q.put(line)
        finally:
            proc.stdout.close()
            self.q.put(("__done__", proc.wait()))

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._log("\n[stopped]\n")
        self.run_btn.config(text="▶  Run"); self.status.config(text="stopped")

    # ------------------------------------------------------ depth calibration #
    def _discover_samples(self):
        """Sample folders under results/ that hold a legacy/measurements.csv — exactly what
        calibrate_depth pools. The calibration output folder is skipped (it has no legacy/).

        Mirrors calibrate_depth.discover_samples, INCLUDING its one-level descent into a wafer-row
        container: a row's samples are listed as '072426 Row 1/c1 D50 P100 P25_S400' so the tokens
        passed to --include and the --cell-filters JSON keys match what that module discovers."""
        if not DEF_OUT.is_dir():
            return []
        out = []
        for sub in sorted(p for p in DEF_OUT.iterdir() if p.is_dir()):
            if sub.name == CAL_OUT_NAME or sub.name.startswith("."):
                continue
            if (sub / MEAS_REL).is_file():
                out.append(sub.name)
            else:                                # a wafer-row container -> one level deeper
                out.extend(f"{sub.name}/{c.name}"
                           for c in sorted(p for p in sub.iterdir()
                                           if p.is_dir() and not p.name.startswith("."))
                           if (c / MEAS_REL).is_file())
        return out

    def _refresh_cal_samples(self):
        """Repopulate the depth-calibration sample table, preserving the current selection AND each
        sample's inline cell filter. On the first populate select all (default = pool everything). A
        brand-new sample (e.g. one just run) is auto-selected ONLY when everything was already
        selected, so an untouched 'all' selection stays 'all' after a new run — while an explicit
        user subset is left as-is."""
        children = self.cal_list.get_children()
        prev_names = [self.cal_list.item(i, "text") for i in children]
        prev_sel = {self.cal_list.item(i, "text") for i in self.cal_list.selection()}
        all_were_selected = bool(prev_names) and prev_sel == set(prev_names)
        first = not prev_names
        names = self._discover_samples()
        # forget cell specs for samples that no longer exist under results/
        self.cal_cell_specs = {n: s for n, s in self.cal_cell_specs.items() if n in names}
        # tear down any in-flight inline editor before the rows it sat on are deleted
        if self._cal_cell_editor is not None:
            try:
                self._cal_cell_editor.destroy()
            except tk.TclError:
                pass
            self._cal_cell_editor = None
        self.cal_list.delete(*children)
        prev_set, to_select = set(prev_names), []
        for n in names:
            spec = self.cal_cell_specs.get(n, "")
            iid = self.cal_list.insert("", "end", text=n, values=(spec if spec else "(all)",))
            if first or n in prev_sel or (all_were_selected and n not in prev_set):
                to_select.append(iid)
        if to_select:
            self.cal_list.selection_set(to_select)

    @staticmethod
    def _validate_cell_spec(spec):
        """(ok, message) for a cell-filter spec. Uses calibrate_depth.parse_cell_spec when
        importable (single source of truth for the grammar); if its heavier deps aren't installed
        on a UI-only box, accept the text and let the calibrate subprocess validate it."""
        if not spec.strip():
            return True, ""
        try:
            from calibrate_depth import parse_cell_spec
        except Exception:                                   # pragma: no cover - UI-only machine
            return True, ""
        try:
            parse_cell_spec(spec)
            return True, ""
        except ValueError as e:
            return False, str(e)

    def _edit_cal_cells(self, event):
        """Double-click handler: open an inline Entry over the 'cells' column to edit that sample's
        cell filter. Commits on Return / focus-out (only if valid), cancels on Escape."""
        if self.cal_list.identify_region(event.x, event.y) != "cell":
            return
        if self.cal_list.identify_column(event.x) != "#1":     # only the 'cells' data column
            return
        iid = self.cal_list.identify_row(event.y)
        if not iid:
            return
        bbox = self.cal_list.bbox(iid, "cells")
        if not bbox:                                           # row scrolled out of view
            return
        x, y, w, h = bbox
        name = self.cal_list.item(iid, "text")
        if self._cal_cell_editor is not None:
            try:
                self._cal_cell_editor.destroy()
            except tk.TclError:
                pass
        var = tk.StringVar(value=self.cal_cell_specs.get(name, ""))
        ent = tk.Entry(self.cal_list, textvariable=var)
        ent.place(x=x, y=y, width=w, height=h)
        ent.focus_set(); ent.icursor("end")
        self._cal_cell_editor = ent
        state = {"closed": False}

        def close():
            if state["closed"]:
                return
            state["closed"] = True
            if self._cal_cell_editor is ent:
                self._cal_cell_editor = None
            try:
                ent.destroy()
            except tk.TclError:
                pass

        def commit():
            raw = var.get().strip()
            ok, msg = self._validate_cell_spec(raw)
            if not ok:
                self.status.config(text=f"cell filter '{raw}': {msg}")
                try:
                    ent.configure(foreground="#b00020")
                except tk.TclError:
                    pass
                return False
            if raw:
                self.cal_cell_specs[name] = raw
            else:
                self.cal_cell_specs.pop(name, None)
            self.cal_list.set(iid, "cells", raw if raw else "(all)")
            return True

        def on_return(_e):
            if commit():                                       # keep editor open if invalid
                close()
            return "break"

        def on_focusout(_e):
            if state["closed"]:
                return
            commit()                                           # save if valid; else discard edit
            close()

        ent.bind("<Return>", on_return)
        ent.bind("<KP_Enter>", on_return)
        ent.bind("<FocusOut>", on_focusout)
        ent.bind("<Escape>", lambda _e: (close(), "break")[1])
        return "break"

    def _calibrate_depth(self):
        if self.cal_proc and self.cal_proc.poll() is None:
            return                                          # already running (button is disabled)
        if self.proc and self.proc.poll() is None:
            return messagebox.showerror("Depth calibration",
                                        "A sample run is in progress — wait for it to finish.")
        names = self._discover_samples()
        if not names:
            return messagebox.showerror(
                "Depth calibration", "No samples with legacy/measurements.csv under results/. "
                "Run at least one sample first.")
        raw = self.cal_target.get().strip()
        try:
            vals = [float(x) for x in raw.split(",") if x.strip()]
            if not vals:
                raise ValueError("empty")
        except ValueError:
            return messagebox.showerror(
                "Depth calibration",
                f"Target depth must be a number or comma-separated numbers (got '{raw}').")
        selected = [self.cal_list.item(i, "text") for i in self.cal_list.selection()]
        out_dir = DEF_OUT / CAL_OUT_NAME
        cmd = [sys.executable, "-u", str(HERE / CAL_SCRIPT),
               "--results", str(DEF_OUT), "--out", str(out_dir),
               "--targets", ",".join(f"{v:g}" for v in vals)]
        if self.cal_allow_legacy_qc.get():
            cmd.append("--allow-legacy-qc")
        # a strict subset -> --include those; all/none selected -> omit so it pools everything.
        # Pass each name as its own token (calibrate_depth --include is nargs='*'), so names with
        # spaces or commas survive intact.
        subset = bool(selected) and len(selected) < len(names)
        if subset:
            cmd += ["--include", *selected]
        # per-sample cell filters -> workspace JSON + --cell-filters, only for the samples that are
        # actually pooled (the selected subset, or every discovered sample when not subsetting).
        pooled = selected if subset else names
        cell_filters = {n: self.cal_cell_specs[n] for n in pooled
                        if self.cal_cell_specs.get(n, "").strip()}
        if cell_filters:
            WORKSPACE.mkdir(exist_ok=True)
            cf_path = WORKSPACE / "cell_filters.json"
            cf_path.write_text(json.dumps(cell_filters, indent=2), encoding="utf-8")
            cmd += ["--cell-filters", str(cf_path)]
        # band definitions -> workspace CSV + --bands, but only if there is a real band row
        # (a blank / comments-only box means "use the measurements 'band' column")
        bands_text = self.cal_bands_text.get("1.0", "end-1c")
        if any(ln.strip() and not ln.strip().startswith("#") for ln in bands_text.splitlines()):
            WORKSPACE.mkdir(exist_ok=True)
            bands_path = WORKSPACE / "band_defs.csv"
            bands_path.write_text(bands_text, encoding="utf-8")
            cmd += ["--bands", str(bands_path)]
        self.console.delete("1.0", "end")
        self._log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        try:
            self.cal_proc = subprocess.Popen(
                cmd, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except Exception as e:
            self.cal_proc = None
            return messagebox.showerror("Depth calibration", str(e))
        self.cal_btn.config(state="disabled"); self.status.config(text="calibrating depth…")
        threading.Thread(target=self._cal_reader, args=(self.cal_proc,), daemon=True).start()

    def _cal_reader(self, proc):
        try:
            for line in proc.stdout:
                self.q.put(line)
        finally:
            proc.stdout.close()
            self.q.put(("__cal_done__", proc.wait()))

    def _drain_console(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__done__":
                    self.run_btn.config(text="▶  Run")
                    self.status.config(text=f"done (exit {item[1]})")
                    self._refresh_results()
                elif isinstance(item, tuple) and item and item[0] == "__cal_done__":
                    self.cal_btn.config(state="normal")
                    self.status.config(text=f"depth calibration done (exit {item[1]})")
                    self._refresh_results()
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self.root.after(120, self._drain_console)

    def _log(self, s):
        self.console.insert("end", s)
        self.console.see("end")

    # ------------------------------------------------- results browser + preview #
    def _refresh_results(self):
        self._refresh_cal_samples()             # keep the depth-calibration sample list in sync
        self.tree.delete(*self.tree.get_children())
        self._tree_paths = {}
        if not DEF_OUT.exists():
            return

        def add(parent, path):
            for child in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                node = self.tree.insert(parent, "end", text=child.name, open=False)  # start collapsed
                self._tree_paths[node] = child
                if child.is_dir():
                    add(node, child)

        add("", DEF_OUT)

    def _on_tree_select(self, _e):
        sel = self.tree.selection()
        if not sel:
            return
        path = self._tree_paths.get(sel[0])
        if path and path.is_file():
            self._preview_file(path)

    def _fit_preview_169(self, e):
        """Fill the available preview frame with a centered 16:9 host."""
        aw = max(e.width - 10, 40)
        ah = max(e.height - 10, 40)

        # Start width-limited, then switch to height-limited when necessary.
        w = aw
        h = round(w * 9 / 16)
        if h > ah:
            h = ah
            w = round(h * 16 / 9)

        self.prev_host.place_configure(width=w, height=h)

    def _show_canvas(self):
        self.text_frame.pack_forget()
        if not self.preview.winfo_ismapped():
            self.preview.pack(fill="both", expand=True)

    def _preview_file(self, path):
        ext = path.suffix.lower()
        if ext in IMG_EXT:
            self._show_canvas()
            self._preview_path = path
            self._show_image(path)
        elif ext in TEXT_EXT:
            self.preview.pack_forget()
            self.text_frame.pack(fill="both", expand=True)
            self._preview_path = None
            self.preview_text.delete("1.0", "end")
            try:
                self.preview_text.insert("1.0", path.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                self.preview_text.insert("1.0", f"(cannot read: {e})")
        else:
            self._show_canvas()
            self._preview_path = None
            self.preview.delete("all")
            self.preview.create_text(16, 16, anchor="nw", fill="#aaa",
                                     text=f"No preview for {path.name}")

    def _reshow_image(self):
        if self._preview_path and self.preview.winfo_ismapped():
            self._show_image(self._preview_path)

    def _show_image(self, path):
        c = self.preview
        c.delete("all")
        cw, ch = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        try:
            if _PIL:
                im = Image.open(path)
                iw, ih = im.size
                sc = min(cw / iw, ch / ih) if iw and ih else 1.0
                im = im.resize((max(1, int(iw * sc)), max(1, int(ih * sc))))
                self._preview_img = ImageTk.PhotoImage(im)
            else:
                img = tk.PhotoImage(file=str(path))
                fac = max(1, int(max(img.width() / cw, img.height() / ch)))
                self._preview_img = img.subsample(fac, fac) if fac > 1 else img
            c.create_image(cw // 2, ch // 2, image=self._preview_img)
        except Exception as ex:
            c.create_text(16, 16, anchor="nw", fill="#f88", text=f"(cannot show image: {ex})")

    # ----------------------------------------------------------------- export #
    def _export_zip(self):
        # A wafer-row result is a CONTAINER: it has no figures/ of its own, but it does have
        # row_figures/ plus one full dataset per sample. Zip the whole row in that case.
        vk4 = self.cur.get("vk4_dir", "")
        is_row = bool(vk4) and Path(vk4).is_dir() and _classify_vk4_folder(vk4)[0] == "row"
        if is_row and self._row_number():
            name = self._row_name()
            root = DEF_OUT / name
            if not (root / ROW_FIGURES_DIR).is_dir():
                return messagebox.showerror("Export", f"No row results for '{name}'. Run it first.")
        else:
            name = self._dataset_name()
            root = self._dataset_out_dir() / "figures"
            if not root.is_dir():
                return messagebox.showerror("Export",
                                            f"No figures to export for '{name}'. Run it first.")
        out = filedialog.asksaveasfilename(title="Save figures zip", defaultextension=".zip",
                                           initialfile=f"{name}_figures.zip",
                                           filetypes=[("Zip archive", "*.zip")])
        if not out:
            return
        try:
            out_resolved = Path(out).resolve()               # don't let the archive include itself
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in root.rglob("*"):
                    if p.is_file() and p.resolve() != out_resolved:
                        zf.write(p, arcname=str(p.relative_to(root)))
        except Exception as e:
            return messagebox.showerror("Export", str(e))
        self.status.config(text=f"exported {Path(out).name}")

    def _on_close(self):
        for p in (self.proc, self.cal_proc):
            if p and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        self.root.destroy()


def _ask_name(root, initial=""):
    dlg = tk.Toplevel(root); dlg.title("Sample name"); dlg.transient(root); dlg.grab_set()
    dlg.configure(bg="#1b1b1b")              # match the app's soft-gray dark theme
    ttk.Label(dlg, text="Name:").grid(row=0, column=0, padx=6, pady=8)
    var = tk.StringVar(value=initial)
    e = ttk.Entry(dlg, textvariable=var, width=26); e.grid(row=0, column=1, padx=6, pady=8)
    e.focus(); e.select_range(0, "end")
    res = {"v": None}

    def ok():
        res["v"] = var.get().strip() or None
        dlg.destroy()

    ttk.Button(dlg, text="OK", command=ok).grid(row=1, column=0, columnspan=2, pady=(0, 8))
    e.bind("<Return>", lambda _event: ok())
    e.bind("<Escape>", lambda _event: dlg.destroy())
    root.wait_window(dlg)
    return res["v"]


def main():
    root = TkinterDnD.Tk() if _DND else tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
