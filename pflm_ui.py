"""
PFLM sample-tester UI — a simple Tkinter front-end for ``run_sample.py``.

Keep a library of samples (each = a DXF + a VK4 tile folder + a cell_params grid), switch
between them, edit the laser-parameter grid, run/stop the tiled analysis while watching the
console live, browse the Results folder + preview figures, and export a zip of ``figures/``.

Drag-and-drop uses ``tkinterdnd2`` if installed (``pip install tkinterdnd2``); otherwise use the
Browse buttons. Image preview uses Pillow if installed, else Tk's built-in PNG support.

Run:  python pflm_ui.py
"""
from __future__ import annotations

import json
import os
import queue
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

    # ------------------------------------------------------------------ layout #
    def _build(self):
        root = self.root
        root.columnconfigure(0, weight=0, minsize=350)
        root.columnconfigure(1, weight=1)
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
        self.dxf_lbl = ttk.Label(dxff, text="(none)", relief="sunken", anchor="w", padding=4)
        self.dxf_lbl.grid(row=0, column=0, sticky="ew")
        ttk.Button(dxff, text="Browse…", command=self._browse_dxf).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._make_drop(self.dxf_lbl, self._on_drop_dxf)

        vk4f = ttk.LabelFrame(left, text="VK4 files (drag/drop folder or files)", padding=6)
        vk4f.grid(row=2, column=0, sticky="nsew", pady=(0, 6)); left.rowconfigure(2, weight=1)
        vk4f.columnconfigure(0, weight=1); vk4f.rowconfigure(1, weight=1)
        self.vk4_dir_lbl = ttk.Label(vk4f, text="(no folder)", relief="sunken", anchor="w", padding=4)
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
                               font=("Consolas", 9))
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
        self.radial_text = tk.Text(radf, height=5, wrap="none", font=("Consolas", 9), undo=True)
        self.radial_text.grid(row=0, column=0, sticky="nsew")
        rvs = ttk.Scrollbar(radf, orient="vertical", command=self.radial_text.yview)
        rvs.grid(row=0, column=1, sticky="ns"); self.radial_text.config(yscrollcommand=rvs.set)
        ttk.Label(radf, text="each line = one overlay set, e.g. P30_S400,P40_S400,P50_S400 · "
                             "empty = overlay every parameter present",
                  foreground="#666", wraplength=280).grid(row=1, column=0, columnspan=2, sticky="w")

        act = ttk.Frame(right); act.grid(row=2, column=0, sticky="ew"); act.columnconfigure(0, weight=1)
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
    def _toggle_run(self):
        if self.proc and self.proc.poll() is None:
            self._stop()
            return
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
        cmd = [sys.executable, "-u", str(HERE / "run_sample.py"),
               str(vk4), str(DEF_OUT), str(dxf), str(csv_path)]
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

    def _drain_console(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__done__":
                    self.run_btn.config(text="▶  Run")
                    self.status.config(text=f"done (exit {item[1]})")
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
        """Keep the preview host locked to 16:9, letterboxed and centred inside its frame."""
        aw, ah = max(e.width - 10, 40), max(e.height - 10, 40)
        if aw / ah > 16 / 9:
            h, w = ah, int(ah * 16 / 9)
        else:
            w, h = aw, int(aw * 9 / 16)
        self.prev_host.place_configure(width=w, height=h)
        self._reshow_image()

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
        figs = DEF_OUT / "figures"
        if not figs.is_dir():
            return messagebox.showerror("Export", "No Results/figures folder to export.")
        out = filedialog.asksaveasfilename(title="Save figures zip", defaultextension=".zip",
                                           initialfile="figures.zip",
                                           filetypes=[("Zip archive", "*.zip")])
        if not out:
            return
        try:
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in figs.rglob("*"):
                    if p.is_file():
                        zf.write(p, arcname=str(p.relative_to(figs)))
        except Exception as e:
            return messagebox.showerror("Export", str(e))
        self.status.config(text=f"exported {Path(out).name}")

    def _on_close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
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
