"""PET SUV conversion.

Given activity concentration in Bq/ml, convert to Standardized Uptake Value
normalized by body weight (BW), body surface area (BSA), or lean body mass
(LBM). The dose is decay-corrected from injection time to acquisition time
using the radionuclide half-life.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import log

import numpy as np

from dicom_reader.io.series import PETMetadata


@dataclass(frozen=True)
class SUVConversion:
    """Multiplicative factor to convert Bq/ml to SUV units.

    `unit` is "g/ml" for BW and LBM, "cm^2/ml" for BSA.
    """

    factor: float
    method: str
    unit: str

    def apply(self, activity: np.ndarray) -> np.ndarray:
        return activity * self.factor


def _decay_corrected_dose(
    dose_bq: float,
    half_life_s: float | None,
    injection: datetime | None,
    acquisition: datetime | None,
) -> float:
    if not half_life_s or injection is None or acquisition is None:
        return dose_bq
    delta = (acquisition - injection).total_seconds()
    if delta <= 0:
        return dose_bq
    return dose_bq * (0.5 ** (delta / half_life_s))


def suv_bw_factor(meta: PETMetadata) -> float:
    """Factor to convert Bq/ml into SUV_bw (g/ml)."""
    if not meta.patient_weight_kg or not meta.injected_dose_bq:
        raise ValueError("SUV_bw needs patient weight and injected dose.")
    dose = _decay_corrected_dose(
        meta.injected_dose_bq,
        meta.half_life_s,
        meta.injection_time,
        meta.acquisition_time,
    )
    if dose <= 0:
        raise ValueError("Decay-corrected dose is non-positive.")
    weight_g = meta.patient_weight_kg * 1000.0
    return weight_g / dose


def suv_bsa_factor(meta: PETMetadata) -> float:
    """Factor to convert Bq/ml into SUV_bsa (cm^2/ml).

    Uses the Du Bois formula: BSA (m^2) = 0.007184 * W^0.425 * H^0.725.
    """
    if not meta.patient_weight_kg or not meta.patient_height_m or not meta.injected_dose_bq:
        raise ValueError("SUV_bsa needs weight, height, and dose.")
    dose = _decay_corrected_dose(
        meta.injected_dose_bq,
        meta.half_life_s,
        meta.injection_time,
        meta.acquisition_time,
    )
    if dose <= 0:
        raise ValueError("Decay-corrected dose is non-positive.")
    height_cm = meta.patient_height_m * 100.0
    bsa_m2 = 0.007184 * (meta.patient_weight_kg ** 0.425) * (height_cm ** 0.725)
    bsa_cm2 = bsa_m2 * 10_000.0
    return bsa_cm2 / dose


def suv_lbm_factor(meta: PETMetadata) -> float:
    """Factor to convert Bq/ml into SUV_lbm (g/ml).

    Uses James formula:
      Male:   LBM = 1.10 * W - 128 * (W/H_cm)^2
      Female: LBM = 1.07 * W -  148 * (W/H_cm)^2
    """
    if (
        not meta.patient_weight_kg
        or not meta.patient_height_m
        or not meta.injected_dose_bq
        or not meta.patient_sex
    ):
        raise ValueError("SUV_lbm needs weight, height, sex, and dose.")
    dose = _decay_corrected_dose(
        meta.injected_dose_bq,
        meta.half_life_s,
        meta.injection_time,
        meta.acquisition_time,
    )
    if dose <= 0:
        raise ValueError("Decay-corrected dose is non-positive.")
    w = meta.patient_weight_kg
    h_cm = meta.patient_height_m * 100.0
    if h_cm <= 0:
        raise ValueError("Invalid height.")
    if meta.patient_sex.upper().startswith("M"):
        lbm_kg = 1.10 * w - 128.0 * (w / h_cm) ** 2
    else:
        lbm_kg = 1.07 * w - 148.0 * (w / h_cm) ** 2
    if lbm_kg <= 0:
        raise ValueError("Computed LBM is non-positive.")
    return (lbm_kg * 1000.0) / dose


def build_suv_conversion(meta: PETMetadata, method: str = "BW") -> SUVConversion:
    method = method.upper()
    if method == "BW":
        return SUVConversion(suv_bw_factor(meta), "BW", "g/ml")
    if method == "BSA":
        return SUVConversion(suv_bsa_factor(meta), "BSA", "cm^2/ml")
    if method == "LBM":
        return SUVConversion(suv_lbm_factor(meta), "LBM", "g/ml")
    raise ValueError(f"Unknown SUV method: {method}")


# Accepts "BQML" or close variants like "BQCC"
def is_bqml(units: str | None) -> bool:
    if not units:
        return False
    u = units.upper().strip()
    return u in ("BQML", "BQCC")
