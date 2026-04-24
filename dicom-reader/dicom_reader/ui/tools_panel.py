"""Right-hand tools panel: window/level, CT presets, PET SUV, fusion, tools."""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from dicom_reader.imaging.windowing import CT_PRESETS, WindowLevel


class ToolsPanel(QtWidgets.QWidget):
    windowChanged = QtCore.pyqtSignal(object)  # WindowLevel
    presetChosen = QtCore.pyqtSignal(str)
    toolChosen = QtCore.pyqtSignal(str)
    fusionChanged = QtCore.pyqtSignal(dict)
    suvMethodChanged = QtCore.pyqtSignal(str)
    resetRequested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._building = True
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(self._build_window_group())
        layout.addWidget(self._build_presets_group())
        layout.addWidget(self._build_pet_group())
        layout.addWidget(self._build_fusion_group())
        layout.addWidget(self._build_tools_group())
        layout.addStretch(1)
        self._building = False

    # --- groups ---

    def _build_window_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Window / Level")
        form = QtWidgets.QFormLayout(box)
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(1, 10000)
        self.width_spin.setValue(400)
        self.level_spin = QtWidgets.QSpinBox()
        self.level_spin.setRange(-2000, 10000)
        self.level_spin.setValue(40)
        for s in (self.width_spin, self.level_spin):
            s.valueChanged.connect(self._emit_window)
        form.addRow("Width", self.width_spin)
        form.addRow("Level", self.level_spin)
        return box

    def _build_presets_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("CT Presets")
        grid = QtWidgets.QGridLayout(box)
        for i, name in enumerate(CT_PRESETS.keys()):
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(lambda _, n=name: self._on_preset(n))
            grid.addWidget(btn, i // 2, i % 2)
        return box

    def _build_pet_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("PET SUV")
        form = QtWidgets.QFormLayout(box)
        self.suv_method = QtWidgets.QComboBox()
        self.suv_method.addItems(["BW", "BSA", "LBM"])
        self.suv_method.currentTextChanged.connect(self.suvMethodChanged.emit)
        form.addRow("Normalization", self.suv_method)
        self.suv_status = QtWidgets.QLabel("—")
        self.suv_status.setWordWrap(True)
        form.addRow("Factor", self.suv_status)
        return box

    def _build_fusion_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("PET/CT Fusion")
        form = QtWidgets.QFormLayout(box)
        self.fusion_enabled = QtWidgets.QCheckBox("Enable fusion overlay")
        self.fusion_enabled.toggled.connect(self._emit_fusion)
        form.addRow(self.fusion_enabled)
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(40)
        self.alpha_slider.valueChanged.connect(self._emit_fusion)
        form.addRow("Alpha", self.alpha_slider)
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(10)
        self.threshold_slider.valueChanged.connect(self._emit_fusion)
        form.addRow("Threshold", self.threshold_slider)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["hot", "pet", "gray"])
        self.cmap_combo.currentTextChanged.connect(self._emit_fusion)
        form.addRow("Colormap", self.cmap_combo)
        self.pet_vmax = QtWidgets.QDoubleSpinBox()
        self.pet_vmax.setRange(0.1, 100.0)
        self.pet_vmax.setDecimals(2)
        self.pet_vmax.setValue(10.0)
        self.pet_vmax.valueChanged.connect(self._emit_fusion)
        form.addRow("SUV max", self.pet_vmax)
        return box

    def _build_tools_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Examination Tools")
        layout = QtWidgets.QVBoxLayout(box)
        self.tool_group = QtWidgets.QButtonGroup(self)
        tools = [
            ("probe", "Pixel probe"),
            ("distance", "Distance"),
            ("rect", "Rectangular ROI"),
            ("ellipse", "Elliptical ROI"),
        ]
        for i, (key, label) in enumerate(tools):
            rb = QtWidgets.QRadioButton(label)
            if i == 0:
                rb.setChecked(True)
            rb.toggled.connect(lambda on, k=key: on and self.toolChosen.emit(k))
            self.tool_group.addButton(rb, i)
            layout.addWidget(rb)
        reset_btn = QtWidgets.QPushButton("Reset view")
        reset_btn.clicked.connect(self.resetRequested.emit)
        layout.addWidget(reset_btn)
        return box

    # --- emit helpers ---

    def _emit_window(self) -> None:
        if self._building:
            return
        wl = WindowLevel(float(self.width_spin.value()), float(self.level_spin.value()))
        self.windowChanged.emit(wl)

    def _on_preset(self, name: str) -> None:
        wl = CT_PRESETS[name]
        self.width_spin.blockSignals(True)
        self.level_spin.blockSignals(True)
        self.width_spin.setValue(int(wl.width))
        self.level_spin.setValue(int(wl.level))
        self.width_spin.blockSignals(False)
        self.level_spin.blockSignals(False)
        self.windowChanged.emit(wl)
        self.presetChosen.emit(name)

    def _emit_fusion(self) -> None:
        if self._building:
            return
        self.fusionChanged.emit(
            {
                "enabled": self.fusion_enabled.isChecked(),
                "alpha": self.alpha_slider.value() / 100.0,
                "threshold": self.threshold_slider.value() / 100.0,
                "cmap": self.cmap_combo.currentText(),
                "vmax": float(self.pet_vmax.value()),
            }
        )

    def set_suv_status(self, text: str) -> None:
        self.suv_status.setText(text)

    def set_window_values(self, wl: WindowLevel) -> None:
        self.width_spin.blockSignals(True)
        self.level_spin.blockSignals(True)
        self.width_spin.setValue(int(wl.width))
        self.level_spin.setValue(int(wl.level))
        self.width_spin.blockSignals(False)
        self.level_spin.blockSignals(False)
