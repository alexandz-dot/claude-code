"""Series data model: a 3D volume plus spatial and radiological metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class Modality(str, Enum):
    CT = "CT"
    PT = "PT"  # PET
    MR = "MR"
    OTHER = "OTHER"

    @classmethod
    def from_dicom(cls, value: str | None) -> "Modality":
        if not value:
            return cls.OTHER
        v = value.upper().strip()
        if v == "CT":
            return cls.CT
        if v in ("PT", "PET"):
            return cls.PT
        if v == "MR":
            return cls.MR
        return cls.OTHER


@dataclass
class PETMetadata:
    """Metadata needed to compute SUV from PET voxel activity.

    All units: activity in Bq/ml after rescale, times as datetime,
    half_life in seconds, weight in kg, height in meters, dose in Bq.
    """

    units: str = "BQML"
    patient_weight_kg: float | None = None
    patient_height_m: float | None = None
    patient_sex: str | None = None
    injected_dose_bq: float | None = None
    half_life_s: float | None = None
    injection_time: datetime | None = None
    acquisition_time: datetime | None = None
    decay_correction: str | None = None
    corrected_image: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class Series:
    """A reconstructed 3D image series.

    `pixels` is shape (n_slices, rows, cols) in float32.
    For CT the values are Hounsfield Units after rescale.
    For PET the values are activity concentration in Bq/ml (if UNITS=BQML).
    """

    pixels: np.ndarray
    spacing: tuple[float, float, float]  # (slice, row, col) in mm
    origin: tuple[float, float, float]  # patient coordinates of voxel [0,0,0]
    orientation: np.ndarray  # 3x3 direction cosines (row, col, slice axes)
    modality: Modality
    study_uid: str = ""
    series_uid: str = ""
    series_description: str = ""
    patient_id: str = ""
    patient_name: str = ""
    study_date: str = ""
    pet: PETMetadata | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.pixels.shape)  # type: ignore[return-value]

    @property
    def n_slices(self) -> int:
        return int(self.pixels.shape[0])

    def value_range(self) -> tuple[float, float]:
        return float(self.pixels.min()), float(self.pixels.max())

    def voxel_to_patient(self, k: float, j: float, i: float) -> np.ndarray:
        """Convert (slice, row, col) voxel index to patient (x,y,z) mm."""
        spacing = np.array(self.spacing)
        idx = np.array([k, j, i]) * spacing
        return self.orientation @ idx + np.array(self.origin)
