"""PyQt6 main window: file open, series picker, tri-planar viewports, side tools."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from dicom_reader.ai import is_available as seg_available
from dicom_reader.ai.masks import SegmentationSet
from dicom_reader.imaging.display import DisplayOptions
from dicom_reader.imaging.fusion import resample_to
from dicom_reader.imaging.mip import SlabMode
from dicom_reader.imaging.reslice import Plane
from dicom_reader.imaging.suv import build_suv_conversion, is_bqml
from dicom_reader.imaging.windowing import WindowLevel
from dicom_reader.io.loader import load_path
from dicom_reader.io.series import Modality, Series
from dicom_reader.ui.dicomweb_dialog import DicomWebDialog
from dicom_reader.ui.seg_runner import SegWorker
from dicom_reader.ui.structures_panel import StructuresPanel
from dicom_reader.ui.tag_browser import TagBrowser
from dicom_reader.ui.tools_panel import ToolsPanel
from dicom_reader.ui.viewport import Viewport


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DICOM Reader — CT / PET-CT")
        self.resize(1500, 950)

        self._series_list: list[Series] = []
        self._ct: Series | None = None
        self._pet: Series | None = None
        self._pet_resampled_to_ct = None  # np.ndarray
        self._suv_factor: float | None = None
        self._segmentation: SegmentationSet | None = None
        self._seg_thread: QtCore.QThread | None = None
        self._seg_worker: SegWorker | None = None
        self._seg_progress: QtWidgets.QProgressDialog | None = None

        self._build_menu()
        self._build_central()
        self._build_statusbar()
        self._wire_signals()

    # --- construction ---

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")

        open_folder = QtGui.QAction("Open DICOM folder…", self)
        open_folder.setShortcut("Ctrl+O")
        open_folder.triggered.connect(self._open_folder)
        file_menu.addAction(open_folder)

        open_file = QtGui.QAction("Open DICOM file…", self)
        open_file.triggered.connect(self._open_file)
        file_menu.addAction(open_file)

        open_web = QtGui.QAction("Open from DICOMweb…", self)
        open_web.triggered.connect(self._open_dicomweb)
        file_menu.addAction(open_web)

        file_menu.addSeparator()
        quit_act = QtGui.QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = bar.addMenu("&View")
        self._tag_toggle = QtGui.QAction("DICOM Tag Browser", self, checkable=True)
        self._tag_toggle.setChecked(True)
        view_menu.addAction(self._tag_toggle)

    def _build_central(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._series_picker = QtWidgets.QListWidget()
        self._series_picker.setMinimumWidth(240)
        self._series_picker.setMaximumWidth(320)

        self._structures_panel = StructuresPanel()

        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        left_splitter.addWidget(self._series_picker)
        left_splitter.addWidget(self._structures_panel)
        left_splitter.setStretchFactor(0, 0)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setSizes([180, 500])

        self._axial = Viewport(Plane.AXIAL)
        self._coronal = Viewport(Plane.CORONAL)
        self._sagittal = Viewport(Plane.SAGITTAL)

        viewports_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(viewports_widget)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(2)
        grid.addWidget(self._axial, 0, 0)
        grid.addWidget(self._coronal, 0, 1)
        grid.addWidget(self._sagittal, 1, 0)

        self._tag_browser = TagBrowser()
        grid.addWidget(self._tag_browser, 1, 1)

        self._tools = ToolsPanel()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_splitter)
        splitter.addWidget(viewports_widget)
        splitter.addWidget(self._tools)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 900, 340])

        root.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_statusbar(self) -> None:
        self._probe_label = QtWidgets.QLabel("—")
        self._measure_label = QtWidgets.QLabel("—")
        self.statusBar().addWidget(self._probe_label, 1)
        self.statusBar().addPermanentWidget(self._measure_label, 1)

    def _wire_signals(self) -> None:
        self._series_picker.currentRowChanged.connect(self._on_series_selected)
        self._tools.windowChanged.connect(self._apply_window)
        self._tools.toolChosen.connect(self._apply_tool)
        self._tools.fusionChanged.connect(self._apply_fusion)
        self._tools.suvMethodChanged.connect(self._apply_suv_method)
        self._tools.slabChanged.connect(self._apply_slab)
        self._tools.segmentRequested.connect(self._run_segmentation)
        self._tools.segAlphaChanged.connect(self._apply_seg_alpha)
        self._tools.displayChanged.connect(self._apply_display)
        self._tools.resetRequested.connect(self._reset_views)
        for vp in self._viewports():
            vp.probed.connect(self._probe_label.setText)
            vp.measured.connect(self._measure_label.setText)
            vp.crosshairMoved.connect(self._broadcast_crosshair)
            vp.windowDragged.connect(self._sync_window_from_drag)
        self._tag_toggle.toggled.connect(self._tag_browser.setVisible)
        self._structures_panel.visibilityChanged.connect(self._on_structures_changed)
        self._structures_panel.computeStatsRequested.connect(self._compute_structure_stats)
        self._refresh_seg_status()

    # --- helpers ---

    def _viewports(self) -> list[Viewport]:
        return [self._axial, self._coronal, self._sagittal]

    # --- actions ---

    def _open_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open DICOM folder")
        if path:
            self._load(Path(path))

    def _open_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open DICOM file")
        if path:
            self._load(Path(path))

    def _load(self, path: Path) -> None:
        try:
            series = load_path(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return
        if not series:
            QtWidgets.QMessageBox.information(
                self, "No series", "No DICOM series found at that path."
            )
            return
        self._series_list = series
        self._ct = next((s for s in series if s.modality == Modality.CT), None)
        self._pet = next((s for s in series if s.modality == Modality.PT), None)
        self._pet_resampled_to_ct = None
        self._segmentation = None
        self._structures_panel.set_segmentation(None)
        for vp in self._viewports():
            vp.set_segmentation(None)
            vp.set_crosshair_voxel(None)
        self._series_picker.clear()
        for s in series:
            label = f"{s.modality.value}  {s.series_description or s.series_uid[-8:]}  ({s.n_slices} sl)"
            self._series_picker.addItem(label)
        if self._ct is not None:
            self._series_picker.setCurrentRow(series.index(self._ct))
        else:
            self._series_picker.setCurrentRow(0)
        self._update_suv_status()
        self._refresh_tag_browser(path)

    def _on_series_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._series_list):
            return
        series = self._series_list[row]
        for vp in self._viewports():
            vp.set_series(series)
            vp.set_suv_factor(self._suv_factor if series.modality == Modality.PT else None)
            # Only show the overlay on the series it was computed against
            if self._segmentation is not None and series.shape == self._segmentation.reference.shape:
                vp.set_segmentation(self._segmentation)
            else:
                vp.set_segmentation(None)
        if series.modality == Modality.CT:
            self._tools.set_window_values(WindowLevel(400, 40))
        elif series.modality == Modality.PT:
            vmax = float(series.pixels.max()) if series.pixels.size else 1.0
            self._tools.set_window_values(WindowLevel(vmax, vmax / 2.0))

    def _apply_window(self, wl: WindowLevel) -> None:
        for vp in self._viewports():
            vp.set_window(wl)

    def _apply_tool(self, tool: str) -> None:
        for vp in self._viewports():
            vp.set_tool(tool)

    def _apply_fusion(self, params: dict) -> None:
        if not params.get("enabled"):
            for vp in self._viewports():
                vp.set_fusion(None, 0.0, params.get("cmap", "hot"), 0.0, 1.0)
            return
        if self._ct is None or self._pet is None:
            QtWidgets.QMessageBox.information(
                self,
                "Fusion requires CT and PET",
                "Load a study that includes both a CT and a PET series.",
            )
            self._tools.fusion_enabled.setChecked(False)
            return
        if self._pet_resampled_to_ct is None:
            self.statusBar().showMessage("Resampling PET onto CT grid…")
            QtWidgets.QApplication.processEvents()
            self._pet_resampled_to_ct = resample_to(self._pet, self._ct, order=1)
            self.statusBar().clearMessage()
        vmax_suv = params["vmax"]
        # If we have an SUV factor and user entered SUV-max, convert to Bq/ml scale
        vmax_activity = vmax_suv / self._suv_factor if self._suv_factor else vmax_suv
        for vp in self._viewports():
            vp.set_fusion(
                self._pet_resampled_to_ct,
                params["alpha"],
                params["cmap"],
                params["threshold"],
                vmax_activity,
            )

    def _apply_suv_method(self, method: str) -> None:
        self._suv_factor = None
        if self._pet is None or self._pet.pet is None:
            self._tools.set_suv_status("No PET metadata available.")
            return
        if not is_bqml(self._pet.pet.units):
            self._tools.set_suv_status(
                f"PET units are {self._pet.pet.units!r}; expected BQML."
            )
            return
        try:
            conv = build_suv_conversion(self._pet.pet, method)
        except ValueError as exc:
            self._tools.set_suv_status(f"Cannot compute SUV_{method}: {exc}")
            return
        self._suv_factor = conv.factor
        self._tools.set_suv_status(
            f"SUV_{conv.method}: factor {conv.factor:.3e} ({conv.unit})"
        )
        for vp in self._viewports():
            if vp._series and vp._series.modality == Modality.PT:
                vp.set_suv_factor(self._suv_factor)

    def _update_suv_status(self) -> None:
        method = self._tools.suv_method.currentText()
        self._apply_suv_method(method)

    def _reset_views(self) -> None:
        for vp in self._viewports():
            vp._plot.getPlotItem().getViewBox().autoRange()

    # --- crosshair ---

    def _broadcast_crosshair(self, k: float, j: float, i: float) -> None:
        voxel = (float(k), float(j), float(i))
        for vp in self._viewports():
            vp.set_crosshair_voxel(voxel)

    # --- display + W/L drag ---

    def _apply_display(self, params: dict) -> None:
        opts = DisplayOptions(
            invert=bool(params.get("invert", False)),
            flip_v=bool(params.get("flip_v", False)),
            flip_h=bool(params.get("flip_h", False)),
            rotations=int(params.get("rotations", 0)) % 4,
        )
        for vp in self._viewports():
            vp.set_display_options(opts)

    def _sync_window_from_drag(self, wl: WindowLevel) -> None:
        # The drag changes the WL on the viewport that received it; mirror
        # to the tools panel and propagate to the other viewports so all
        # MPR planes stay in sync.
        self._tools.set_window_values(wl)
        sender = self.sender()
        for vp in self._viewports():
            if vp is sender:
                continue
            vp.set_window(wl)

    # --- slab ---

    def _apply_slab(self, params: dict) -> None:
        mode = SlabMode(params["mode"])
        thickness = int(params["thickness"])
        for vp in self._viewports():
            vp.set_slab(thickness, mode)

    # --- segmentation ---

    def _refresh_seg_status(self) -> None:
        if seg_available():
            self._tools.set_seg_status("TotalSegmentator available.", ok=True)
        else:
            self._tools.set_seg_status(
                "TotalSegmentator not installed. Install with "
                "`pip install 'dicom-reader[ai]'` to enable anatomical segmentation.",
                ok=False,
            )

    def _run_segmentation(self, params: dict) -> None:
        if self._ct is None:
            QtWidgets.QMessageBox.warning(
                self, "No CT", "Load a CT series before running segmentation."
            )
            return
        if self._seg_thread is not None and self._seg_thread.isRunning():
            return

        self._seg_progress = QtWidgets.QProgressDialog(
            "Preparing…", "Cancel", 0, 100, self
        )
        self._seg_progress.setWindowTitle("Segmentation")
        self._seg_progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self._seg_progress.setAutoClose(False)
        self._seg_progress.setValue(1)

        self._seg_thread = QtCore.QThread(self)
        self._seg_worker = SegWorker(self._ct, params["task"], params["fast"])
        self._seg_worker.moveToThread(self._seg_thread)
        self._seg_thread.started.connect(self._seg_worker.run)
        self._seg_worker.progress.connect(self._on_seg_progress)
        self._seg_worker.finished.connect(self._on_seg_finished)
        self._seg_worker.failed.connect(self._on_seg_failed)
        self._seg_thread.start()

    def _on_seg_progress(self, msg: str, frac: float) -> None:
        if self._seg_progress is not None:
            self._seg_progress.setLabelText(msg)
            self._seg_progress.setValue(int(frac * 100))

    def _on_seg_finished(self, segmentation) -> None:
        self._close_seg_thread()
        self._segmentation = segmentation
        self._structures_panel.set_segmentation(segmentation)
        for vp in self._viewports():
            vp.set_segmentation(segmentation)

    def _on_seg_failed(self, message: str) -> None:
        self._close_seg_thread()
        QtWidgets.QMessageBox.critical(self, "Segmentation failed", message)

    def _close_seg_thread(self) -> None:
        if self._seg_progress is not None:
            self._seg_progress.close()
            self._seg_progress = None
        if self._seg_thread is not None:
            self._seg_thread.quit()
            self._seg_thread.wait()
            self._seg_thread = None
        self._seg_worker = None

    def _apply_seg_alpha(self, alpha: float) -> None:
        for vp in self._viewports():
            vp.set_seg_alpha(alpha)

    def _on_structures_changed(self) -> None:
        for vp in self._viewports():
            vp.refresh()

    def _compute_structure_stats(self, label_id: int) -> None:
        if self._segmentation is None:
            return
        stats = self._segmentation.structure_stats(label_id)
        if stats is None:
            self._structures_panel.set_stats(label_id, "—")
            return
        ref = self._segmentation.reference
        unit = "HU" if ref.modality == Modality.CT else (
            "Bq/ml" if ref.modality == Modality.PT else "val"
        )
        text = (
            f"V={stats['volume_ml']:.1f} ml  "
            f"mean={stats['mean']:.1f} {unit}  "
            f"min={stats['min']:.1f}  max={stats['max']:.1f}"
        )
        if (
            ref.modality == Modality.PT
            and self._suv_factor is not None
        ):
            f = self._suv_factor
            text += (
                f"  | SUV mean={stats['mean'] * f:.2f}  "
                f"max={stats['max'] * f:.2f}"
            )
        self._structures_panel.set_stats(label_id, text)

    # --- DICOMweb ---

    def _open_dicomweb(self) -> None:
        dlg = DicomWebDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            path = dlg.retrieved_dir()
            if path is not None:
                self._load(path)

    def _refresh_tag_browser(self, path: Path) -> None:
        if path.is_file():
            self._tag_browser.load_file(path)
            return
        for p in path.rglob("*"):
            if p.is_file():
                self._tag_browser.load_file(p)
                break


def launch(path: Path | None = None) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if path is not None and path.exists():
        window._load(path)
    return app.exec()
