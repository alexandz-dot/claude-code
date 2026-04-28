"""Qt background worker that runs TotalSegmentator without blocking the UI."""

from __future__ import annotations

from PyQt6 import QtCore

from dicom_reader.ai.masks import SegmentationSet
from dicom_reader.ai.segmentation import run_totalsegmentator
from dicom_reader.io.series import Series


class SegWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(str, float)
    finished = QtCore.pyqtSignal(object)  # SegmentationSet or None
    failed = QtCore.pyqtSignal(str)

    def __init__(self, ct: Series, task: str, fast: bool) -> None:
        super().__init__()
        self._ct = ct
        self._task = task
        self._fast = fast

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            result = run_totalsegmentator(
                self._ct,
                task=self._task,
                fast=self._fast,
                progress=lambda msg, frac: self.progress.emit(msg, frac),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)
