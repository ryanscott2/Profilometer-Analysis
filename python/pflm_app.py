"""PySide6 + QML desktop front end for the PFLM sample tester.

    python pflm_app.py

The Windows 11 Fluent-style desktop UI: keep a library of named samples, run the
tiled analysis while watching the console, and browse the figures it produces.
Covers the single-sample run, the wafer-row batch mode, depth calibration with
per-sample cell filters, and the figures zip export.

The run commands' shared pieces -- the sample library and workspace paths, the
results-folder naming, and the VK4-folder classification -- come from `ui_shared`
and `wafer_map` by import rather than being reimplemented here. `ui_shared` is
stdlib-only, so importing it never pulls numpy/pandas/matplotlib into UI startup.

Needs `pip install PySide6` alongside this project's own requirements. On Windows,
enable long paths first (`LongPathsEnabled`), or the wheel half-extracts and leaves
a tree with no QML modules at all.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import PySide6

# The QML plugins link against the Qt6*.dll files in the PySide6 package root, and
# Windows will not search that root for a DLL loaded from a nested directory. Must
# happen before Qt loads any plugin.
_PYSIDE_DIR = str(Path(PySide6.__file__).parent)
os.environ["PATH"] = _PYSIDE_DIR + os.pathsep + os.environ.get("PATH", "")
try:
    os.add_dll_directory(_PYSIDE_DIR)
except OSError:
    pass

from PySide6.QtCore import (Property, QObject, QProcess, Qt, QUrl,  # noqa: E402
                            Signal, Slot)
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtQml import QQmlApplicationEngine  # noqa: E402
from PySide6.QtQuickControls2 import QQuickStyle  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repo root: modules live in python/, data sits beside it
sys.path.insert(0, str(HERE))

# Shared launch helpers: results/sample-library paths, VK4 classification, name guards.
import ui_shared  # noqa: E402
from wafer_map import (DEFAULT_MAP_NAME, date_tag_from_names,  # noqa: E402
                       read_wafer_map, row_out_name, rows_present)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".bmp")


class Bridge(QObject):
    statusChanged = Signal()
    busyChanged = Signal()
    samplesChanged = Signal()
    figuresChanged = Signal()
    logAppended = Signal(str)
    logCleared = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._status = "Pick a sample, or fill the fields and save one."
        self._busy = False
        self._figures: list[str] = []
        self._process: QProcess | None = None
        self._samples = self._read_samples()

    # ------------------------------------------------------------- samples

    def _read_samples(self) -> dict:
        path = ui_shared.SAMPLES_JSON
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return loaded
            except (OSError, ValueError):
                pass
        return {}

    def _write_samples(self) -> None:
        try:
            ui_shared.SAMPLES_JSON.write_text(
                json.dumps(self._samples, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            self._set_status(f"Could not save samples: {exc}")

    def _get_sample_names(self) -> list:
        return sorted(self._samples)

    sampleNames = Property(list, _get_sample_names, notify=samplesChanged)

    @Slot(str, result="QVariantMap")
    def loadSample(self, name: str) -> dict:
        stored = self._samples.get(name)
        if not stored:
            return {"ok": False}
        result = {
            "ok": True,
            "dxf": str(stored.get("dxf") or ""),
            "vk4_dir": str(stored.get("vk4_dir") or ""),
            "csv_text": str(stored.get("csv_text") or ""),
            "radial_text": str(stored.get("radial_text") or ""),
        }
        self.refreshFigures(name)
        mode = self.classify(result["vk4_dir"])
        self._set_status(f"'{name}' loaded  ·  VK4 folder classifies as {mode}")
        return result

    @Slot(str, "QVariantMap")
    def saveSample(self, name: str, values) -> None:
        name = name.strip()
        if not name:
            self._set_status("Give the sample a name first.")
            return
        existing = dict(self._samples.get(name) or {})
        existing.update({
            "dxf": str(values.get("dxf") or ""),
            "vk4_dir": str(values.get("vk4_dir") or ""),
            "csv_text": str(values.get("csv_text") or ""),
            "radial_text": str(values.get("radial_text") or ""),
        })
        self._samples[name] = existing
        self._write_samples()
        self.samplesChanged.emit()
        self._set_status(f"Saved '{name}'.")

    @Slot(str)
    def deleteSample(self, name: str) -> None:
        if self._samples.pop(name, None) is not None:
            self._write_samples()
            self.samplesChanged.emit()
            self._set_status(f"Deleted '{name}'.")

    @Slot(str, result=str)
    def classify(self, vk4_dir: str) -> str:
        """Snapshots or tiles, decided by ui_shared's VK4-folder classifier."""
        path = Path(vk4_dir)
        if not path.is_dir():
            return "no folder"
        try:
            mode, _labels = ui_shared._classify_vk4_folder(path)
            return str(mode)
        except Exception:  # noqa: BLE001 - informational only
            return "unknown"

    # ------------------------------------------------------------- figures

    def _get_figures(self) -> list:
        return self._figures

    figures = Property(list, _get_figures, notify=figuresChanged)

    @Slot(str)
    def refreshFigures(self, name: str) -> None:
        found: list[str] = []
        out_dir = ui_shared.DEF_OUT / ui_shared._safe_name(name)
        for directory in (out_dir / "figures", out_dir):
            if directory.is_dir():
                found.extend(
                    QUrl.fromLocalFile(str(p)).toString()
                    for p in sorted(directory.iterdir())
                    if p.suffix.lower() in IMAGE_SUFFIXES
                )
            if found:
                break
        self._figures = found
        self.figuresChanged.emit()

    @Slot(str, result=str)
    def resultsPath(self, name: str) -> str:
        return str(ui_shared.DEF_OUT / ui_shared._safe_name(name))

    # -------------------------------------------------------------- rows

    @Slot(str, str, result="QVariantMap")
    def rowInfo(self, vk4_dir: str, map_override: str) -> dict:
        """Which wafer rows the map offers for this VK4 folder, plus any hard error.

        Searches where run_row.py searches, so the UI shows what the run will read:
        beside the VK4 folder, its parent, then csv/.
        """
        result = {"mode": self.classify(vk4_dir), "rows": [], "mapPath": "", "problem": ""}
        chosen = Path(map_override) if map_override else None
        if chosen is not None and chosen.is_file():
            found = chosen
        else:
            folder = Path(vk4_dir)
            found = next((c for c in (folder / DEFAULT_MAP_NAME,
                                      folder.parent / DEFAULT_MAP_NAME,
                                      ui_shared.ROOT / "csv" / DEFAULT_MAP_NAME)
                          if c.is_file()), None)
        if found is None:
            result["problem"] = (f"No {DEFAULT_MAP_NAME} beside the VK4 folder, in its parent, "
                                 "or in csv/. Pick one explicitly.")
            return result
        result["mapPath"] = str(found)
        try:
            entries, _meta, problems = read_wafer_map(found)
        except Exception as exc:  # noqa: BLE001 - reported to the user
            result["problem"] = f"Could not read {found.name}: {exc}"
            return result
        hard = [p for p in problems if not p.startswith("WARNING")]
        if hard:
            result["problem"] = "\n".join(hard[:8])
            return result
        result["rows"] = [int(r) for r in rows_present(entries)]
        return result

    @Slot(str, int, result=str)
    def rowName(self, vk4_dir: str, row: int) -> str:
        """The Results folder a row run writes, named the same way run_row.py names it."""
        if not row:
            return ""
        tag = date_tag_from_names([f.name for f in Path(vk4_dir).rglob("*.vk4")]) or ""
        return ui_shared._safe_name(row_out_name(tag, row))

    @Slot(str, "QStringList", int, str)
    def runRow(self, vk4_dir: str, dxfs, row: int, map_override: str) -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return
        if not row:
            self._set_status("Pick a wafer row first.")
            return
        info = self.rowInfo(vk4_dir, map_override)
        if info["problem"]:
            self._set_status(info["problem"].splitlines()[0])
            self.logAppended.emit(info["problem"] + "\n")
            return
        if row not in info["rows"]:
            present = ", ".join(str(r) for r in info["rows"]) or "none"
            self._set_status(f"The map has no entries for row {row} (present: {present}).")
            return

        name = self.rowName(vk4_dir, row)
        # Three guards, so a row run cannot overwrite a single-sample result or the
        # reserved calibration folder.
        collision = ui_shared._sample_name_collision(name, list(self._samples))
        if collision:
            self._set_status(f"Row folder '{name}' collides with sample '{collision}'.")
            return
        if name.casefold() == ui_shared.CAL_OUT_NAME.casefold():
            self._set_status(f"'{ui_shared.CAL_OUT_NAME}' is reserved for depth calibration.")
            return
        out_dir = ui_shared.DEF_OUT / name
        if out_dir.is_dir() and ((out_dir / "legacy").is_dir() or (out_dir / "figures").is_dir()
                                 or (out_dir / ".pflm-results.json").is_file()):
            self._set_status(f"results/{name} is an existing single-sample result, not a row "
                             "container. Pick a different row.")
            return

        arguments = ["-u", str(HERE / ui_shared.ROW_SCRIPT), "--row", str(row),
                     "--map", info["mapPath"], "--vk4", str(vk4_dir), "--out", str(out_dir)]
        dxf_files = [d for d in (dxfs or []) if d and Path(d).is_file()]
        if len(dxf_files) == 1:
            # One drawing picked: keep the whole folder as the candidate pool (the shipped
            # behaviour), so a multi-geometry row still finds every drawing it needs by content.
            arguments += ["--dxf-dir", str(Path(dxf_files[0]).parent)]
        elif dxf_files:
            # Several drawings picked: pass exactly those as the pool -- no stray or unreadable DXFs
            # from other wafer generations can slip in. This is the reliable path the UI now offers.
            arguments += ["--dxf-dir", *[str(Path(d)) for d in dxf_files]]
        self._launch(arguments, f"row {row} into {name}", figures_for=name)

    # ------------------------------------------------------- depth calibration

    @Slot(result=str)
    def defaultBands(self) -> str:
        return str(ui_shared.DEFAULT_BAND_DEFS)

    @Slot(result=list)
    def calibrationCandidates(self) -> list:
        """Result entries the pool is built from, each carrying a legacy measurements CSV.

        Mirrors calibrate_depth.discover_samples (keep in sync): a top-level result folder that
        directly holds legacy/measurements.csv is one entry; a wafer-row CONTAINER (which has none
        at the top) contributes its per-column samples one level deeper, named '<container>/<sample>'
        -- exactly the names discover_samples pools and that --include/--exclude and cell filters
        match on. Without this descent, row samples were pooled by an 'all results' run but could
        never be individually picked."""
        root = ui_shared.DEF_OUT
        if not root.is_dir():
            return []
        out = []
        for sub in sorted(p for p in root.iterdir()
                          if p.is_dir() and not p.name.startswith(".")):
            if (sub / ui_shared.MEAS_REL).is_file():
                out.append(sub.name)
            else:                                    # a wafer-row container -> one level deeper
                out += [f"{sub.name}/{c.name}"
                        for c in sorted(p for p in sub.iterdir()
                                        if p.is_dir() and not p.name.startswith("."))
                        if (c / ui_shared.MEAS_REL).is_file()]
        return out

    @Slot(str, result=str)
    def validateCellSpec(self, spec: str) -> str:
        """Empty string when the spec is fine, otherwise the complaint.

        Uses calibrate_depth.parse_cell_spec so the grammar has one definition. If
        that module's heavier dependencies are missing, accept the text and let the
        calibrate subprocess be the judge.
        """
        if not str(spec).strip():
            return ""
        try:
            from calibrate_depth import parse_cell_spec
        except Exception:  # noqa: BLE001 - numpy and friends may not be installed here
            return ""
        try:
            parse_cell_spec(str(spec))
        except ValueError as exc:
            return str(exc)
        return ""

    @Slot(str, list, str, bool, "QVariantMap", str, str)
    def runCalibration(self, targets: str, include: list, bands_text: str,
                       allow_legacy_qc: bool, cell_filters, speed: str = "",
                       max_passes: str = "") -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return
        try:
            values = [float(v) for v in str(targets).replace(";", ",").split(",") if v.strip()]
        except ValueError:
            self._set_status(f"Target depth must be numbers, got '{targets}'.")
            return
        if not values:
            self._set_status("Give at least one target depth.")
            return
        # Optional fixed-speed depth-vs-passes slice. Both fields blank -> no --speed passed -> the
        # figure is not produced (the CLI default), so an ordinary calibration is unchanged.
        speed_val = str(speed).strip()
        if speed_val:
            try:
                sp = float(speed_val)
                if not sp > 0:
                    raise ValueError
            except ValueError:
                self._set_status(f"Scan speed must be a positive number, got '{speed}'.")
                return
        max_passes_val = str(max_passes).strip()
        if max_passes_val:
            try:
                mp = float(max_passes_val)
                if not mp > 0:
                    raise ValueError
            except ValueError:
                self._set_status(f"Max passes must be a positive number, got '{max_passes}'.")
                return

        out_dir = ui_shared.DEF_OUT / ui_shared.CAL_OUT_NAME
        arguments = ["-u", str(HERE / ui_shared.CAL_SCRIPT),
                     "--results", str(ui_shared.DEF_OUT), "--out", str(out_dir),
                     "--targets", ",".join(f"{v:g}" for v in values)]
        if speed_val:
            arguments += ["--speed", f"{sp:g}"]
            if max_passes_val:                           # --max-passes only affects the --speed figure
                arguments += ["--max-passes", f"{mp:g}"]
        if allow_legacy_qc:
            arguments.append("--allow-legacy-qc")
        # A strict subset restricts the pool; all or none pools everything.
        chosen = [str(n) for n in include if str(n).strip()]
        if chosen and len(chosen) < len(self.calibrationCandidates()):
            arguments += ["--include", *chosen]
        # Per-sample cell filters, but only for the samples actually pooled: the
        # selected subset, or every discovered sample when not subsetting. A blank
        # spec means "every cell of that sample", so it is simply omitted.
        pooled = chosen if (chosen and len(chosen) < len(self.calibrationCandidates())) \
            else self.calibrationCandidates()
        filters = {name: str(cell_filters.get(name) or "").strip() for name in pooled}
        filters = {name: spec for name, spec in filters.items() if spec}
        if filters:
            ui_shared.WORKSPACE.mkdir(exist_ok=True)
            filters_path = ui_shared.WORKSPACE / "cell_filters.json"
            filters_path.write_text(json.dumps(filters, indent=2), encoding="utf-8")
            arguments += ["--cell-filters", str(filters_path)]
        # Blank or comments-only means "use the measurements' own band column".
        if any(line.strip() and not line.strip().startswith("#")
               for line in str(bands_text).splitlines()):
            ui_shared.WORKSPACE.mkdir(exist_ok=True)
            bands_path = ui_shared.WORKSPACE / "band_defs.csv"
            bands_path.write_text(str(bands_text), encoding="utf-8")
            arguments += ["--bands", str(bands_path)]
        self._launch(arguments, "depth calibration", figures_for=ui_shared.CAL_OUT_NAME)

    # ------------------------------------------------------------ zip export

    @Slot(str, str, bool, result=str)
    def exportZip(self, name: str, destination: str, is_row: bool) -> str:
        """Zip a sample's figures/, or a whole row container, which has no figures/."""
        import zipfile

        target = Path(QUrl(destination).toLocalFile()
                      if destination.startswith("file:") else destination)
        folder = ui_shared.DEF_OUT / ui_shared._safe_name(name)
        root = folder if is_row else folder / "figures"
        if is_row and not (folder / ui_shared.ROW_FIGURES_DIR).is_dir():
            return f"No row results for '{name}'. Run it first."
        if not root.is_dir():
            return f"Nothing to export for '{name}'. Run it first."
        try:
            resolved = target.resolve()
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for item in sorted(root.rglob("*")):
                    # Never let the archive swallow itself.
                    if item.is_file() and item.resolve() != resolved:
                        archive.write(item, item.relative_to(root))
        except OSError as exc:
            return f"Could not write {target.name}: {exc}"
        self._set_status(f"Exported {target.name}")
        return ""

    # ----------------------------------------------------------- run / stop

    def _get_status(self) -> str:
        return self._status

    status = Property(str, _get_status, notify=statusChanged)

    def _set_status(self, text: str) -> None:
        self._status = text
        self.statusChanged.emit()

    def _get_busy(self) -> bool:
        return self._busy

    busy = Property(bool, _get_busy, notify=busyChanged)

    def _set_busy(self, value: bool) -> None:
        if value != self._busy:
            self._busy = value
            self.busyChanged.emit()

    @Slot(str, "QVariantMap")
    def run(self, name: str, values) -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return
        name = name.strip()
        if not name:
            self._set_status("A run needs a selected sample, so results do not collide.")
            return
        dxf = Path(str(values.get("dxf") or ""))
        vk4 = Path(str(values.get("vk4_dir") or ""))
        if not dxf.is_file():
            self._set_status(f"DXF not found: {dxf}")
            return
        if not vk4.is_dir():
            self._set_status(f"VK4 folder not found: {vk4}")
            return

        # The params files live beside each other in the workspace, and run_sample.py
        # reads radial_sets.csv from there.
        ui_shared.WORKSPACE.mkdir(exist_ok=True)
        csv_path = ui_shared.WORKSPACE / "cell_params.csv"
        csv_path.write_text(str(values.get("csv_text") or ""), encoding="utf-8")
        (ui_shared.WORKSPACE / "radial_sets.csv").write_text(
            str(values.get("radial_text") or ""), encoding="utf-8"
        )

        out_dir = ui_shared.DEF_OUT / ui_shared._safe_name(name)
        mode, _labels = ui_shared._classify_vk4_folder(vk4)
        if mode == "snapshots":
            dose = ui_shared._first_ps_label(str(values.get("csv_text") or ""))
            arguments = ["-u", str(HERE / "run_sample.py"), "--snapshots",
                         str(vk4), str(out_dir), str(dxf), dose]
        else:
            arguments = ["-u", str(HERE / "run_sample.py"),
                         str(vk4), str(out_dir), str(dxf), str(csv_path)]

        self._launch(arguments, f"'{name}' ({mode})", figures_for=name)

    def _launch(self, arguments: list[str], label: str, figures_for: str = "") -> None:
        """One console, one Stop button and one finished handler for all three runs."""
        self.logCleared.emit()
        self.logAppended.emit(
            "> " + " ".join(f'"{a}"' if " " in a else a for a in arguments) + "\n\n"
        )
        self._set_busy(True)
        self._set_status(f"Running {label}...")
        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(HERE))
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.setProcessEnvironment(self._process_environment())
        self._process.readyReadStandardOutput.connect(self._drain)
        self._process.finished.connect(
            lambda code, _s: self._finished(figures_for or label, code)
        )
        self._process.start(sys.executable, arguments)

    @staticmethod
    def _process_environment():
        from PySide6.QtCore import QProcessEnvironment

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        return environment

    @Slot()
    def stop(self) -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._set_status("Stopped.")

    def _drain(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        if chunk:
            self.logAppended.emit(chunk)

    def _finished(self, name: str, code: int) -> None:
        self._set_busy(False)
        if code == 0:
            self.logAppended.emit("\nDone.\n")
            self._set_status(f"'{name}' finished.")
        else:
            self.logAppended.emit(f"\nExited with code {code}.\n")
            self._set_status(f"'{name}' failed with code {code}. See the console.")
        self.refreshFigures(name)
        self._process = None


def main() -> int:
    QGuiApplication.setApplicationName("PFLM Profilometer Analysis")
    QQuickStyle.setStyle(os.environ.get("PFLM_QML_STYLE", "FluentWinUI3"))
    app = QGuiApplication(sys.argv)
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)

    engine = QQmlApplicationEngine()
    bridge = Bridge()
    engine.rootContext().setContextProperty("bridge", bridge)
    engine.load(QUrl.fromLocalFile(str(ROOT / "qml" / "Main.qml")))
    if not engine.rootObjects():
        print("Failed to load the QML interface.", file=sys.stderr)
        return 1

    window = engine.rootObjects()[0]
    if "--tab" in sys.argv:
        window.setProperty("initialTab", int(sys.argv[sys.argv.index("--tab") + 1]))
    if "--sample" in sys.argv:
        window.setProperty("initialSample", sys.argv[sys.argv.index("--sample") + 1])

    if "--screenshot" in sys.argv:
        from PySide6.QtCore import QTimer

        target = Path(sys.argv[sys.argv.index("--screenshot") + 1])

        def capture() -> None:
            image = window.grabWindow()
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"{'saved' if image.save(str(target)) else 'FAILED'} {target} "
                  f"({image.width()}x{image.height()})", flush=True)
            app.quit()

        QTimer.singleShot(3500, capture)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
