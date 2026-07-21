"""
PFLM sample-tester UI — a simple Tkinter front-end for ``run_sample.py``.

Keep a library of samples (each = a DXF + a VK4 tile folder + a cell_params grid), switch
between them, edit the laser-parameter grid, run/stop the tiled analysis while watching the
console live, browse the Results folder + preview figures, and export a zip of ``figures/``.
Each run writes under ``Results/<sample name>/`` (a sample must be selected to run), so different
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

HERE = Path(__file__).resolve().parent
SAMPLES_JSON = HERE / ".ui_samples.json"
WORKSPACE = HERE / ".ui_workspace"
DEF_OUT = HERE / "Results"
IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".bmp")
TEXT_EXT = (".txt", ".csv", ".log", ".json")

# Depth-calibration tool (calibrate_depth.py). Keep these in sync with that module's OUT_NAME /
# MEAS_REL: it pools the per-sample legacy CSVs and writes the cross-sample analysis here.
CAL_SCRIPT = "calibrate_depth.py"
CAL_OUT_NAME = "etch depth"
MEAS_REL = Path("legacy") / "measurements.csv"

# Prefilled band definitions (one band per line: min_Ø, max_Ø, pitch in µm). Matches the provided
# 4x4 single-cell DXF: Ø 50–67.5 µm @100 µm pitch and Ø 100–125 µm @150 µm pitch. Editable; blank
# (or comments only) tells calibrate_depth.py to fall back to the measurements' own 'band' column.
DEFAULT_BAND_DEFS = ("# min_Ø, max_Ø, pitch (µm) — one band per line; blank = use CSV 'band' column\n"
                     "50, 67.5, 100\n"
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


_SLASH = re.compile(r"[/\\]")
_INVALID_NAME = re.compile(r'[<>:"|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def _safe_name(name):
    """Turn a UI sample name into a filesystem-safe folder name (Windows-safe): slashes become
    spaces; any other character illegal in a Windows path becomes an underscore; a Windows reserved
    device name (CON, PRN, COM1, ...) gets a trailing underscore so it can be a real folder.

    NOTE: '/' -> space is intentional (per the sample-naming convention) and is not one-to-one, so
    e.g. 'D100/D50' and 'D100 D50' still map to the same folder -- that collision is inherent to the
    space convention, not a sanitizer bug."""
    name = _SLASH.sub(" ", str(name))
    name = _INVALID_NAME.sub("_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if name.upper().split(".")[0] in _RESERVED:
        name += "_"
    return name or "unnamed"


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


class App:
    def __init__(self, root):
        self.root = root
        root.title("PFLM sample tester")
        root.geometry("1520x900")
        self.proc = None
        self.cal_proc = None                # depth-calibration subprocess (separate from a run)
        self.q = queue.Queue()
        self.samples = self._load_samples()
        self.cur = {"dxf": "", "vk4_dir": ""}
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
        """Pre-fill from the repo's DXF/ VK4/ CSV/ so the currently-loaded sample runs at once."""
        dxfs = sorted((HERE / "DXF").glob("*.dxf")) if (HERE / "DXF").is_dir() else []
        if dxfs:
            self._set_dxf(dxfs[0])
        if (HERE / "VK4").is_dir() and any((HERE / "VK4").glob("*.vk4")):
            self._set_vk4_dir(HERE / "VK4")
        csv = HERE / "CSV" / "cell_params.csv"
        if csv.exists():
            self.csv_text.delete("1.0", "end")
            self.csv_text.insert("1.0", csv.read_text(encoding="utf-8-sig"))
        rad = HERE / "CSV" / "radial_sets.csv"
        if rad.exists():
            self.radial_text.delete("1.0", "end")
            self.radial_text.insert("1.0", rad.read_text(encoding="utf-8-sig"))
        # band definitions for the depth calibration: from CSV/band_defs.csv if present, else default
        bands = HERE / "CSV" / "band_defs.csv"
        self.cal_bands_text.delete("1.0", "end")
        self.cal_bands_text.insert("1.0", bands.read_text(encoding="utf-8-sig")
                                   if bands.exists() else DEFAULT_BAND_DEFS)

    # ------------------------------------------------------------------ layout #
    def _build(self):
        root = self.root
        # Keep the two control columns compact and give all surplus width to
        # the preview.  Widgets inside the side columns also use bounded
        # requested widths below so long paths/text cannot squeeze this column.
        root.columnconfigure(0, weight=0, minsize=350)
        root.columnconfigure(1, weight=1, minsize=480)
        root.columnconfigure(2, weight=0, minsize=300)
        root.rowconfigure(0, weight=1)

        left = ttk.Frame(root, padding=6); left.grid(row=0, column=0, sticky="nsew")
        mid = ttk.Frame(root, padding=6); mid.grid(row=0, column=1, sticky="nsew")
        right = ttk.Frame(root, padding=6); right.grid(row=0, column=2, sticky="nsew")

        # ---------- LEFT: sample library + inputs ----------
        left.columnconfigure(0, weight=1)
        lib = ttk.LabelFrame(left, text="Samples", padding=6)
        lib.grid(row=0, column=0, sticky="ew", pady=(0, 6)); lib.columnconfigure((0, 1, 2), weight=1)
        self.sample_var = tk.StringVar()
        self.sample_combo = ttk.Combobox(lib, textvariable=self.sample_var, state="readonly")
        self.sample_combo.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.sample_combo.bind("<<ComboboxSelected>>",
                               lambda e: self._load_sample(self.sample_var.get()))
        ttk.Button(lib, text="Save as…", command=self._save_as).grid(row=1, column=0, sticky="ew", pady=4, padx=1)
        ttk.Button(lib, text="Update", command=self._update_sample).grid(row=1, column=1, sticky="ew", pady=4, padx=1)
        ttk.Button(lib, text="Delete", command=self._delete_sample).grid(row=1, column=2, sticky="ew", pady=4, padx=1)

        dxff = ttk.LabelFrame(left, text="DXF file (drag/drop)", padding=6)
        dxff.grid(row=1, column=0, sticky="ew", pady=(0, 6)); dxff.columnconfigure(0, weight=1)
        self.dxf_lbl = ttk.Label(dxff, text="(none)", relief="sunken", anchor="w",
                                 padding=4, width=1)
        self.dxf_lbl.grid(row=0, column=0, sticky="ew")
        ttk.Button(dxff, text="Browse…", command=self._browse_dxf).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._make_drop(self.dxf_lbl, self._on_drop_dxf)

        vk4f = ttk.LabelFrame(left, text="VK4 files (drag/drop folder or files)", padding=6)
        vk4f.grid(row=2, column=0, sticky="nsew", pady=(0, 6)); left.rowconfigure(2, weight=1)
        vk4f.columnconfigure(0, weight=1); vk4f.rowconfigure(1, weight=1)
        self.vk4_dir_lbl = ttk.Label(vk4f, text="(no folder)", relief="sunken", anchor="w",
                                     padding=4, width=1)
        self.vk4_dir_lbl.grid(row=0, column=0, sticky="ew")
        lbf = ttk.Frame(vk4f); lbf.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        lbf.columnconfigure(0, weight=1); lbf.rowconfigure(0, weight=1)
        self.vk4_list = tk.Listbox(lbf, height=6, activestyle="none")
        self.vk4_list.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lbf, orient="vertical", command=self.vk4_list.yview)
        sb.grid(row=0, column=1, sticky="ns"); self.vk4_list.config(yscrollcommand=sb.set)
        ttk.Button(vk4f, text="Browse folder…", command=self._browse_vk4).grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self._make_drop(self.vk4_list, self._on_drop_vk4)

        lpf = ttk.LabelFrame(left, text="Laser parameters (cell_params grid — edit freely)", padding=6)
        lpf.grid(row=3, column=0, sticky="nsew"); left.rowconfigure(3, weight=1)
        lpf.columnconfigure(0, weight=1); lpf.rowconfigure(0, weight=1)
        self.csv_text = tk.Text(lpf, height=6, width=34, wrap="none", font=("Consolas", 10),
                                undo=True)
        self.csv_text.grid(row=0, column=0, sticky="nsew")
        cs = ttk.Scrollbar(lpf, orient="vertical", command=self.csv_text.yview)
        cs.grid(row=0, column=1, sticky="ns"); self.csv_text.config(yscrollcommand=cs.set)
        ttk.Label(lpf, text="grid = design orientation (DXF top-left = line 1, col 1); "
                            "each cell 'P{passes}_S{speed}'",
                  foreground="#666", wraplength=320).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # ---------- MIDDLE: preview (locked 16:9) + console ----------
        mid.columnconfigure(0, weight=1); mid.rowconfigure(0, weight=3); mid.rowconfigure(1, weight=1)
        prevf = ttk.LabelFrame(mid, text="Preview (select a file in the Results browser →)", padding=4)
        prevf.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        prevf.bind("<Configure>", self._fit_preview_169)
        # host kept at 16:9 (letterboxed, centred in prevf); canvas + text overlay it
        self.prev_host = tk.Frame(prevf, background="#202020")
        self.prev_host.place(relx=0.5, rely=0.5, anchor="center", width=320, height=180)
        self.preview = tk.Canvas(self.prev_host, background="#202020", highlightthickness=0)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Configure>", lambda e: self._reshow_image())
        self.text_frame = tk.Frame(self.prev_host)          # shown for txt/csv, with H+V scrollbars
        self.text_frame.columnconfigure(0, weight=1); self.text_frame.rowconfigure(0, weight=1)
        self.preview_text = tk.Text(self.text_frame, wrap="none", font=("Consolas", 10))
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        tvs = ttk.Scrollbar(self.text_frame, orient="vertical", command=self.preview_text.yview)
        tvs.grid(row=0, column=1, sticky="ns")
        ths = ttk.Scrollbar(self.text_frame, orient="horizontal", command=self.preview_text.xview)
        ths.grid(row=1, column=0, sticky="ew")
        self.preview_text.config(yscrollcommand=tvs.set, xscrollcommand=ths.set)

        consf = ttk.LabelFrame(mid, text="Console", padding=4)
        consf.grid(row=1, column=0, sticky="nsew")
        consf.columnconfigure(0, weight=1); consf.rowconfigure(0, weight=1)
        self.console = tk.Text(consf, wrap="none", background="#111", foreground="#ddd",
                               font=("Consolas", 9), height=8)
        self.console.grid(row=0, column=0, sticky="nsew")
        kv = ttk.Scrollbar(consf, orient="vertical", command=self.console.yview)
        kv.grid(row=0, column=1, sticky="ns")
        kh = ttk.Scrollbar(consf, orient="horizontal", command=self.console.xview)
        kh.grid(row=1, column=0, sticky="ew")
        self.console.config(yscrollcommand=kv.set, xscrollcommand=kh.set)

        # ---------- RIGHT: results browser + radial-average sets + actions ----------
        right.columnconfigure(0, weight=1); right.rowconfigure(0, weight=1)
        resf = ttk.LabelFrame(right, text="Results folder", padding=4)
        resf.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        resf.columnconfigure(0, weight=1); resf.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(resf, show="tree")
        self.tree.grid(row=0, column=0, sticky="nsew")
        tb = ttk.Scrollbar(resf, orient="vertical", command=self.tree.yview)
        tb.grid(row=0, column=1, sticky="ns")
        thb = ttk.Scrollbar(resf, orient="horizontal", command=self.tree.xview)
        thb.grid(row=1, column=0, sticky="ew")
        self.tree.config(yscrollcommand=tb.set, xscrollcommand=thb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        ttk.Button(resf, text="Refresh", command=self._refresh_results).grid(row=2, column=0, sticky="ew", pady=(4, 0))

        radf = ttk.LabelFrame(right, text="Radial-average sets  (one set per line)", padding=4)
        radf.grid(row=1, column=0, sticky="nsew", pady=(0, 6))
        radf.columnconfigure(0, weight=1); radf.rowconfigure(0, weight=1)
        self.radial_text = tk.Text(radf, height=5, width=32, wrap="none",
                                   font=("Consolas", 9), undo=True)
        self.radial_text.grid(row=0, column=0, sticky="nsew")
        rvs = ttk.Scrollbar(radf, orient="vertical", command=self.radial_text.yview)
        rvs.grid(row=0, column=1, sticky="ns"); self.radial_text.config(yscrollcommand=rvs.set)
        ttk.Label(radf, text="each line = one overlay set, e.g. P30_S400,P40_S400,P50_S400 · "
                             "empty = overlay every parameter present",
                  foreground="#666", wraplength=280).grid(row=1, column=0, columnspan=2, sticky="w")

        # ---------- RIGHT: depth calibration (pool completed samples, post-hoc) ----------
        # Shells out to calibrate_depth.py on the samples selected here; output streams to the
        # console and the report/figures land under Results/_depth_calibration (browsable at left).
        calf = ttk.LabelFrame(right, text="Depth calibration  (pool samples → depth = f(passes, speed))",
                              padding=4)
        calf.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        calf.columnconfigure(0, weight=1)
        clf = ttk.Frame(calf); clf.grid(row=0, column=0, columnspan=2, sticky="nsew")
        clf.columnconfigure(0, weight=1); clf.rowconfigure(0, weight=1)
        # exportselection=False keeps the multi-selection when focus moves to another text widget
        self.cal_list = tk.Listbox(clf, height=5, selectmode="extended", activestyle="none",
                                   exportselection=False)
        self.cal_list.grid(row=0, column=0, sticky="nsew")
        cls = ttk.Scrollbar(clf, orient="vertical", command=self.cal_list.yview)
        cls.grid(row=0, column=1, sticky="ns"); self.cal_list.config(yscrollcommand=cls.set)
        ttk.Label(calf, text="samples to include (Ctrl/Shift-click for multi; none selected = all "
                             "discovered)", foreground="#666", wraplength=280).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(2, 2))
        ttk.Label(calf, text="bands — one per line: min Ø, max Ø, pitch (µm)  (a pin joins a band "
                             "only if its Ø is in range AND its pitch matches).  blank = use CSV "
                             "'band' column", foreground="#666", wraplength=280).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))
        bdf = ttk.Frame(calf); bdf.grid(row=3, column=0, columnspan=2, sticky="ew")
        bdf.columnconfigure(0, weight=1)
        self.cal_bands_text = tk.Text(bdf, height=3, width=30, wrap="none", font=("Consolas", 9),
                                      undo=True)
        self.cal_bands_text.grid(row=0, column=0, sticky="nsew")
        bds = ttk.Scrollbar(bdf, orient="vertical", command=self.cal_bands_text.yview)
        bds.grid(row=0, column=1, sticky="ns"); self.cal_bands_text.config(yscrollcommand=bds.set)
        tgtf = ttk.Frame(calf); tgtf.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 2))
        ttk.Label(tgtf, text="target depth (µm):").grid(row=0, column=0, sticky="w")
        self.cal_target = tk.StringVar(value="55")
        ttk.Entry(tgtf, textvariable=self.cal_target, width=16).grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Label(tgtf, text="(comma-sep OK)", foreground="#666").grid(row=0, column=2, sticky="w", padx=(4, 0))
        self.cal_btn = ttk.Button(calf, text="Calibrate depth", command=self._calibrate_depth)
        self.cal_btn.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        act = ttk.Frame(right); act.grid(row=3, column=0, sticky="ew"); act.columnconfigure(0, weight=1)
        self.run_btn = ttk.Button(act, text="▶  Run", command=self._toggle_run)
        self.run_btn.grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(act, text="Export figures .zip", command=self._export_zip).grid(row=1, column=0, sticky="ew", pady=2)
        self.status = ttk.Label(act, text="idle", anchor="w", relief="groove", padding=3)
        self.status.grid(row=2, column=0, sticky="ew", pady=(4, 0))

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

    def _set_vk4_dir(self, d):
        d = Path(d)
        self.cur["vk4_dir"] = str(d)
        self.vk4_dir_lbl.config(text=str(d))
        self.vk4_list.delete(0, tk.END)
        vks = sorted(d.glob("*.vk4"))
        for f in vks:
            self.vk4_list.insert(tk.END, f.name)
        self.vk4_dir_lbl.config(text=f"{d}   ({len(vks)} .vk4)")

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
        if s.get("vk4_dir") and Path(s["vk4_dir"]).is_dir():
            self._set_vk4_dir(s["vk4_dir"])
        self.csv_text.delete("1.0", "end")
        self.csv_text.insert("1.0", s.get("csv_text", ""))
        self.radial_text.delete("1.0", "end")
        self.radial_text.insert("1.0", s.get("radial_text", ""))     # old samples: absent -> empty
        self.status.config(text=f"loaded '{name}'")

    def _snapshot(self):
        return {"dxf": self.cur["dxf"], "vk4_dir": self.cur["vk4_dir"],
                "csv_text": self._csv(), "radial_text": self._radial()}

    def _save_as(self):
        name = _ask_name(self.root, self.sample_var.get())
        if not name:
            return
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
        """Per-dataset output root — Results/<dataset name>/ — holding this run's figures/ + legacy/."""
        return DEF_OUT / self._dataset_name()

    def _toggle_run(self):
        if self.proc and self.proc.poll() is None:
            self._stop()
            return
        if self.cal_proc and self.cal_proc.poll() is None:
            return messagebox.showerror("Run", "Depth calibration is in progress — wait for it to "
                                               "finish before starting a sample run.")
        if not self.sample_var.get().strip():
            return messagebox.showerror(
                "Run", "Select a sample in the Samples dropdown first — its name is used for the "
                "Results subfolder.  Use “Save as…” to name the current setup as a sample.")
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
        # each run writes under Results/<dataset name>/ so datasets don't overwrite each other
        # (run_sample clears only this subfolder, leaving other datasets' results intact)
        out_dir = self._dataset_out_dir()
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
        """Sample folders under Results/ that hold a legacy/measurements.csv — exactly what
        calibrate_depth pools. The calibration output folder is skipped (it has no legacy/)."""
        if not DEF_OUT.is_dir():
            return []
        out = []
        for sub in sorted(p for p in DEF_OUT.iterdir() if p.is_dir()):
            if sub.name == CAL_OUT_NAME:
                continue
            if (sub / MEAS_REL).is_file():
                out.append(sub.name)
        return out

    def _refresh_cal_samples(self):
        """Repopulate the depth-calibration sample list, preserving the current selection. On the
        first populate select all (default = pool everything). A brand-new sample (e.g. one just
        run) is auto-selected ONLY when everything was already selected, so an untouched 'all'
        selection stays 'all' after a new run — while an explicit user subset is left as-is."""
        prev_names = [self.cal_list.get(i) for i in range(self.cal_list.size())]
        prev_sel = {self.cal_list.get(i) for i in self.cal_list.curselection()}
        all_were_selected = bool(prev_names) and prev_sel == set(prev_names)
        first = not prev_names
        names = self._discover_samples()
        self.cal_list.delete(0, tk.END)
        for n in names:
            self.cal_list.insert(tk.END, n)
        for i, n in enumerate(names):
            if first or n in prev_sel or (all_were_selected and n not in set(prev_names)):
                self.cal_list.selection_set(i)

    def _calibrate_depth(self):
        if self.cal_proc and self.cal_proc.poll() is None:
            return                                          # already running (button is disabled)
        if self.proc and self.proc.poll() is None:
            return messagebox.showerror("Depth calibration",
                                        "A sample run is in progress — wait for it to finish.")
        names = self._discover_samples()
        if not names:
            return messagebox.showerror(
                "Depth calibration", "No samples with legacy/measurements.csv under Results/. "
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
        selected = [self.cal_list.get(i) for i in self.cal_list.curselection()]
        out_dir = DEF_OUT / CAL_OUT_NAME
        cmd = [sys.executable, "-u", str(HERE / CAL_SCRIPT),
               "--results", str(DEF_OUT), "--out", str(out_dir),
               "--targets", ",".join(f"{v:g}" for v in vals)]
        # a strict subset -> --include those; all/none selected -> omit so it pools everything.
        # Pass each name as its own token (calibrate_depth --include is nargs='*'), so names with
        # spaces or commas survive intact.
        if selected and len(selected) < len(names):
            cmd += ["--include", *selected]
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
                node = self.tree.insert(parent, "end", text=child.name,
                                        open=(parent == "" or child.name == "figures"))
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
        name = self._dataset_name()
        figs = self._dataset_out_dir() / "figures"
        if not figs.is_dir():
            return messagebox.showerror("Export", f"No figures to export for '{name}'. Run it first.")
        out = filedialog.asksaveasfilename(title="Save figures zip", defaultextension=".zip",
                                           initialfile=f"{name}_figures.zip",
                                           filetypes=[("Zip archive", "*.zip")])
        if not out:
            return
        try:
            out_resolved = Path(out).resolve()               # don't let the archive include itself
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in figs.rglob("*"):
                    if p.is_file() and p.resolve() != out_resolved:
                        zf.write(p, arcname=str(p.relative_to(figs)))
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
    ttk.Label(dlg, text="Name:").grid(row=0, column=0, padx=6, pady=8)
    var = tk.StringVar(value=initial)
    e = ttk.Entry(dlg, textvariable=var, width=26); e.grid(row=0, column=1, padx=6, pady=8)
    e.focus(); e.select_range(0, "end")
    res = {"v": None}

    def ok():
        res["v"] = var.get().strip() or None
        dlg.destroy()

    ttk.Button(dlg, text="OK", command=ok).grid(row=1, column=0, columnspan=2, pady=(0, 8))
    e.bind("<Return>", lambda ev: ok())
    e.bind("<Escape>", lambda ev: dlg.destroy())
    root.wait_window(dlg)
    return res["v"]


def main():
    root = TkinterDnD.Tk() if _DND else tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
