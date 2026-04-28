"""SUV computation correctness tests.

References:
  SUV_bw = C(Bq/ml) * weight(g) / dose_at_scan(Bq)
  Dose decay: dose_at_scan = dose_injected * 0.5 ** (dt / half_life)
"""

from __future__ import annotations

from datetime import datetime

import pytest

from dicom_reader.imaging.suv import (
    build_suv_conversion,
    suv_bsa_factor,
    suv_bw_factor,
    suv_lbm_factor,
)
from dicom_reader.io.series import PETMetadata


def _fdg_meta(**over) -> PETMetadata:
    base = dict(
        units="BQML",
        patient_weight_kg=70.0,
        patient_height_m=1.75,
        patient_sex="M",
        injected_dose_bq=370_000_000.0,  # 370 MBq
        half_life_s=6586.2,  # F-18
        injection_time=datetime(2025, 1, 1, 9, 0, 0),
        acquisition_time=datetime(2025, 1, 1, 10, 0, 0),  # 60 min uptake
    )
    base.update(over)
    return PETMetadata(**base)


def test_suv_bw_matches_decay_formula():
    meta = _fdg_meta()
    f = suv_bw_factor(meta)
    dose_at_scan = 370_000_000.0 * 0.5 ** (3600 / 6586.2)
    expected = (70.0 * 1000.0) / dose_at_scan
    assert f == pytest.approx(expected, rel=1e-9)


def test_suv_bsa_is_positive_and_scales_with_dose():
    meta = _fdg_meta()
    f1 = suv_bsa_factor(meta)
    meta_half = _fdg_meta(injected_dose_bq=185_000_000.0)
    f2 = suv_bsa_factor(meta_half)
    assert f1 > 0 and f2 > 0
    # half dose -> double factor
    assert f2 == pytest.approx(2 * f1, rel=1e-9)


def test_suv_lbm_james_male_vs_female():
    meta_m = _fdg_meta(patient_sex="M")
    meta_f = _fdg_meta(patient_sex="F")
    fm = suv_lbm_factor(meta_m)
    ff = suv_lbm_factor(meta_f)
    assert fm != ff


def test_build_suv_conversion_switch():
    meta = _fdg_meta()
    bw = build_suv_conversion(meta, "BW")
    bsa = build_suv_conversion(meta, "BSA")
    lbm = build_suv_conversion(meta, "LBM")
    assert bw.method == "BW" and bw.unit == "g/ml"
    assert bsa.method == "BSA" and bsa.unit == "cm^2/ml"
    assert lbm.method == "LBM" and lbm.unit == "g/ml"


def test_missing_metadata_raises():
    meta = PETMetadata(units="BQML")
    with pytest.raises(ValueError):
        suv_bw_factor(meta)
