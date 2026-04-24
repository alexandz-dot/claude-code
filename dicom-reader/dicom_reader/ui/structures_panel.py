"""Structures panel: list of segmented organs with visibility and color.

Each row has a visibility checkbox, a color swatch (click to edit), the
structure name, and CT HU or PET SUV stats on hover / "Compute stats".
"""

from __future__ import annotations

from typing import Callable

from PyQt6 import QtCore, QtGui, QtWidgets

from dicom_reader.ai.masks import SegmentationSet, Structure


class StructuresPanel(QtWidgets.QWidget):
    visibilityChanged = QtCore.pyqtSignal()
    computeStatsRequested = QtCore.pyqtSignal(int)  # label_id

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._segmentation: SegmentationSet | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QtWidgets.QHBoxLayout()
        self._title = QtWidgets.QLabel("No segmentation loaded")
        header.addWidget(self._title, 1)

        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText("Filter structures…")
        self._filter.textChanged.connect(self._apply_filter)
        header.addWidget(self._filter, 2)
        layout.addLayout(header)

        action_row = QtWidgets.QHBoxLayout()
        self._show_all = QtWidgets.QPushButton("Show all")
        self._hide_all = QtWidgets.QPushButton("Hide all")
        self._show_all.clicked.connect(lambda: self._set_all(True))
        self._hide_all.clicked.connect(lambda: self._set_all(False))
        action_row.addWidget(self._show_all)
        action_row.addWidget(self._hide_all)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setRootIsDecorated(False)
        self._tree.setHeaderLabels(["", "", "Structure", "Stats"])
        self._tree.setColumnWidth(0, 30)
        self._tree.setColumnWidth(1, 30)
        self._tree.setColumnWidth(2, 180)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._tree, 1)

    # --- API ---

    def set_segmentation(self, segmentation: SegmentationSet | None) -> None:
        self._segmentation = segmentation
        self._tree.clear()
        if segmentation is None:
            self._title.setText("No segmentation loaded")
            return
        self._title.setText(f"{segmentation.source}  ({len(segmentation.structures)} structures)")
        for structure in segmentation.structures:
            self._add_row(structure)
        self._tree.sortItems(2, QtCore.Qt.SortOrder.AscendingOrder)

    def set_stats(self, label_id: int, text: str) -> None:
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item and int(item.data(0, QtCore.Qt.ItemDataRole.UserRole)) == label_id:
                item.setText(3, text)
                return

    # --- rows ---

    def _add_row(self, structure: Structure) -> None:
        item = QtWidgets.QTreeWidgetItem(["", "", structure.name, ""])
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, structure.label_id)
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            0,
            QtCore.Qt.CheckState.Checked if structure.visible else QtCore.Qt.CheckState.Unchecked,
        )
        self._set_swatch(item, structure.color_rgb)
        self._tree.addTopLevelItem(item)
        self._tree.itemChanged.connect(self._on_item_changed)

    def _set_swatch(self, item: QtWidgets.QTreeWidgetItem, rgb: tuple[int, int, int]) -> None:
        pix = QtGui.QPixmap(16, 16)
        pix.fill(QtGui.QColor(*rgb))
        item.setIcon(1, QtGui.QIcon(pix))

    # --- events ---

    def _on_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if self._segmentation is None or column != 0:
            return
        label_id = int(item.data(0, QtCore.Qt.ItemDataRole.UserRole))
        structure = self._segmentation.by_id(label_id)
        if structure is None:
            return
        structure.visible = item.checkState(0) == QtCore.Qt.CheckState.Checked
        self.visibilityChanged.emit()

    def _on_double_click(
        self, item: QtWidgets.QTreeWidgetItem, column: int
    ) -> None:
        if self._segmentation is None:
            return
        label_id = int(item.data(0, QtCore.Qt.ItemDataRole.UserRole))
        structure = self._segmentation.by_id(label_id)
        if structure is None:
            return
        if column == 1:
            color = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(*structure.color_rgb), self, "Structure color"
            )
            if color.isValid():
                structure.color_rgb = (color.red(), color.green(), color.blue())
                self._set_swatch(item, structure.color_rgb)
                self.visibilityChanged.emit()
        else:
            self.computeStatsRequested.emit(label_id)

    def _apply_filter(self, text: str) -> None:
        text = text.strip().lower()
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item is None:
                continue
            name = item.text(2).lower()
            item.setHidden(bool(text) and text not in name)

    def _set_all(self, visible: bool) -> None:
        if self._segmentation is None:
            return
        state = QtCore.Qt.CheckState.Checked if visible else QtCore.Qt.CheckState.Unchecked
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item is None or item.isHidden():
                continue
            item.setCheckState(0, state)
