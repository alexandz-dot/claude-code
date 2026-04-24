"""Dialog to browse a DICOMweb server and pick a series to load.

Flow:
  1. User enters server URL + optional token/auth
  2. Search studies (optional PatientID filter)
  3. Pick a study -> list its series
  4. Pick a series -> retrieve to a temp dir; dialog returns the dir path
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from PyQt6 import QtCore, QtWidgets

from dicom_reader.io.dicomweb import DicomWebClient


def _first(ds: dict[str, Any], tag: str, default: str = "") -> str:
    node = ds.get(tag)
    if not node or "Value" not in node or not node["Value"]:
        return default
    v = node["Value"][0]
    if isinstance(v, dict):
        return v.get("Alphabetic", default)
    return str(v)


class DicomWebDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open from DICOMweb")
        self.resize(900, 600)
        self._client: DicomWebClient | None = None
        self._dest_dir: Path | None = None

        layout = QtWidgets.QVBoxLayout(self)

        # Connection row
        conn = QtWidgets.QGroupBox("Server")
        form = QtWidgets.QFormLayout(conn)
        self.url = QtWidgets.QLineEdit()
        self.url.setPlaceholderText("https://dicomweb.example/dicom-web")
        self.token = QtWidgets.QLineEdit()
        self.token.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("Optional bearer token")
        self.user = QtWidgets.QLineEdit()
        self.user.setPlaceholderText("Optional basic-auth user")
        self.password = QtWidgets.QLineEdit()
        self.password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.verify_tls = QtWidgets.QCheckBox("Verify TLS certificate")
        self.verify_tls.setChecked(True)
        form.addRow("URL", self.url)
        form.addRow("Bearer token", self.token)
        form.addRow("Username", self.user)
        form.addRow("Password", self.password)
        form.addRow(self.verify_tls)
        connect_btn = QtWidgets.QPushButton("Connect")
        connect_btn.clicked.connect(self._connect)
        form.addRow(connect_btn)
        layout.addWidget(conn)

        # Study search row
        search_row = QtWidgets.QHBoxLayout()
        self.patient_id = QtWidgets.QLineEdit()
        self.patient_id.setPlaceholderText("Filter by PatientID (optional)")
        search_btn = QtWidgets.QPushButton("Search studies")
        search_btn.clicked.connect(self._search)
        search_row.addWidget(self.patient_id, 1)
        search_row.addWidget(search_btn)
        layout.addLayout(search_row)

        # Studies + series lists
        lists = QtWidgets.QHBoxLayout()
        self.studies = QtWidgets.QTreeWidget()
        self.studies.setHeaderLabels(["Date", "Patient", "Study UID", "Modalities"])
        self.studies.itemSelectionChanged.connect(self._on_study)
        self.series = QtWidgets.QTreeWidget()
        self.series.setHeaderLabels(["Modality", "Description", "Series UID", "#"])
        lists.addWidget(self.studies, 1)
        lists.addWidget(self.series, 1)
        layout.addLayout(lists)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.fetch_btn = QtWidgets.QPushButton("Retrieve series")
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.clicked.connect(self._retrieve)
        btns.addButton(self.fetch_btn, QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self.series.itemSelectionChanged.connect(
            lambda: self.fetch_btn.setEnabled(len(self.series.selectedItems()) > 0)
        )

    # --- API ---

    def retrieved_dir(self) -> Path | None:
        return self._dest_dir

    # --- actions ---

    def _connect(self) -> None:
        url = self.url.text().strip()
        if not url:
            QtWidgets.QMessageBox.warning(self, "URL required", "Enter a server URL.")
            return
        self._client = DicomWebClient(
            base_url=url,
            token=self.token.text().strip() or None,
            username=self.user.text().strip() or None,
            password=self.password.text() or None,
            verify_tls=self.verify_tls.isChecked(),
        )
        self.studies.clear()
        self.series.clear()
        QtWidgets.QMessageBox.information(self, "Connected", "Credentials saved. Use Search to list studies.")

    def _search(self) -> None:
        if self._client is None:
            QtWidgets.QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        pid = self.patient_id.text().strip()
        filters = {"PatientID": pid} if pid else None
        try:
            results = self._client.search_studies(filters)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Search failed", str(exc))
            return
        self.studies.clear()
        for ds in results:
            item = QtWidgets.QTreeWidgetItem(
                [
                    _first(ds, "00080020"),  # StudyDate
                    _first(ds, "00100010"),  # PatientName
                    _first(ds, "0020000D"),  # StudyInstanceUID
                    _first(ds, "00080061"),  # ModalitiesInStudy
                ]
            )
            self.studies.addTopLevelItem(item)

    def _on_study(self) -> None:
        self.series.clear()
        selected = self.studies.selectedItems()
        if not selected or self._client is None:
            return
        study_uid = selected[0].text(2)
        try:
            results = self._client.search_series(study_uid)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Series fetch failed", str(exc))
            return
        for ds in results:
            item = QtWidgets.QTreeWidgetItem(
                [
                    _first(ds, "00080060"),  # Modality
                    _first(ds, "0008103E"),  # SeriesDescription
                    _first(ds, "0020000E"),  # SeriesInstanceUID
                    _first(ds, "00201209"),  # NumberOfSeriesRelatedInstances
                ]
            )
            self.series.addTopLevelItem(item)

    def _retrieve(self) -> None:
        if self._client is None:
            return
        study_sel = self.studies.selectedItems()
        series_sel = self.series.selectedItems()
        if not study_sel or not series_sel:
            return
        study_uid = study_sel[0].text(2)
        series_uid = series_sel[0].text(2)
        dest = Path(tempfile.mkdtemp(prefix="dicomweb_"))
        try:
            self._client.retrieve_series(study_uid, series_uid, dest)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Retrieve failed", str(exc))
            return
        self._dest_dir = dest
        self.accept()
