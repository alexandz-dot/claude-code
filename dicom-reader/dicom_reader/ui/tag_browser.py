"""DICOM tag browser panel.

Reads the first DICOM file of a series and displays all public tags in a tree
view. Large binary values are elided.
"""

from __future__ import annotations

from pathlib import Path

import pydicom
from PyQt6 import QtCore, QtWidgets


class TagBrowser(QtWidgets.QTreeWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderLabels(["Tag", "VR", "Name", "Value"])
        self.setColumnWidth(0, 120)
        self.setColumnWidth(1, 40)
        self.setColumnWidth(2, 200)

    def load_file(self, path: Path) -> None:
        self.clear()
        try:
            ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
        except Exception as exc:
            self.addTopLevelItem(QtWidgets.QTreeWidgetItem(["Error", "", "", str(exc)]))
            return
        self._populate(ds, self.invisibleRootItem())
        self.expandToDepth(0)

    def _populate(self, ds, parent: QtWidgets.QTreeWidgetItem) -> None:
        for elem in ds:
            tag = f"({elem.tag.group:04X},{elem.tag.element:04X})"
            vr = elem.VR or ""
            name = elem.name or ""
            if elem.VR == "SQ":
                node = QtWidgets.QTreeWidgetItem([tag, vr, name, f"<{len(elem.value)} item(s)>"])
                parent.addChild(node)
                for i, item in enumerate(elem.value):
                    item_node = QtWidgets.QTreeWidgetItem(["", "", f"Item #{i}", ""])
                    node.addChild(item_node)
                    self._populate(item, item_node)
            else:
                value = self._format_value(elem.value)
                parent.addChild(QtWidgets.QTreeWidgetItem([tag, vr, name, value]))

    def _format_value(self, value) -> str:
        if isinstance(value, bytes):
            return f"<{len(value)} bytes>"
        s = str(value)
        if len(s) > 200:
            return s[:200] + "…"
        return s
