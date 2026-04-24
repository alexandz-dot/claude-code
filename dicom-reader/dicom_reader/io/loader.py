"""DICOM file discovery and series reconstruction.

Reads a folder of DICOM files, groups by SeriesInstanceUID, sorts by
ImagePositionPatient along the slice-normal axis, and builds a 3D volume with
correct spacing/origin/orientation. Applies RescaleSlope/RescaleIntercept so CT
is in HU and PET is in activity units.
"""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
from pydicom.dataset import FileDataset

from dicom_reader.io.series import Modality, PETMetadata, Series


def _iter_dicom_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            yield p


def _read_header(path: Path) -> FileDataset | None:
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def discover_series(root: Path) -> dict[str, list[Path]]:
    """Group DICOM files under `root` by SeriesInstanceUID."""
    groups: dict[str, list[Path]] = {}
    for path in _iter_dicom_files(root):
        ds = _read_header(path)
        if ds is None:
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if not uid:
            continue
        groups.setdefault(str(uid), []).append(path)
    return groups


def _parse_dicom_datetime(date_str: str | None, time_str: str | None) -> datetime | None:
    if not date_str or not time_str:
        return None
    try:
        ds = str(date_str).strip()
        ts = str(time_str).strip().split(".")[0]
        if len(ts) < 6:
            ts = ts.ljust(6, "0")
        return datetime.strptime(f"{ds}{ts[:6]}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _collect_pet_metadata(ds: FileDataset) -> PETMetadata:
    meta = PETMetadata()
    meta.units = str(getattr(ds, "Units", "BQML") or "BQML")
    meta.patient_weight_kg = _maybe_float(getattr(ds, "PatientWeight", None))
    height = _maybe_float(getattr(ds, "PatientSize", None))
    meta.patient_height_m = height
    meta.patient_sex = str(getattr(ds, "PatientSex", "") or "") or None
    meta.decay_correction = str(getattr(ds, "DecayCorrection", "") or "") or None
    ci = getattr(ds, "CorrectedImage", None)
    if ci:
        meta.corrected_image = tuple(str(x) for x in ci)

    rps = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if rps:
        item = rps[0]
        meta.injected_dose_bq = _maybe_float(
            getattr(item, "RadionuclideTotalDose", None)
        )
        meta.half_life_s = _maybe_float(
            getattr(item, "RadionuclideHalfLife", None)
        )
        meta.injection_time = _parse_injection_time(ds, item)

    meta.acquisition_time = _parse_dicom_datetime(
        getattr(ds, "AcquisitionDate", None) or getattr(ds, "SeriesDate", None),
        getattr(ds, "AcquisitionTime", None) or getattr(ds, "SeriesTime", None),
    )
    return meta


def _parse_injection_time(ds: FileDataset, item) -> datetime | None:
    start_dt = getattr(item, "RadiopharmaceuticalStartDateTime", None)
    if start_dt:
        try:
            s = str(start_dt).split(".")[0]
            return datetime.strptime(s[:14], "%Y%m%d%H%M%S")
        except ValueError:
            pass
    start_time = getattr(item, "RadiopharmaceuticalStartTime", None)
    if not start_time:
        return None
    date_part = (
        getattr(ds, "SeriesDate", None)
        or getattr(ds, "StudyDate", None)
        or getattr(ds, "AcquisitionDate", None)
    )
    return _parse_dicom_datetime(date_part, start_time)


def _maybe_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _image_orientation(ds: FileDataset) -> tuple[np.ndarray, np.ndarray]:
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        row = np.array([1.0, 0.0, 0.0])
        col = np.array([0.0, 1.0, 0.0])
        return row, col
    row = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
    col = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
    return row, col


def _pixel_spacing(ds: FileDataset) -> tuple[float, float]:
    ps = getattr(ds, "PixelSpacing", None)
    if ps is None or len(ps) != 2:
        return 1.0, 1.0
    return float(ps[0]), float(ps[1])


def load_series(files: list[Path]) -> Series:
    """Build a Series from a list of DICOM file paths belonging to one series.

    Handles both classic single-frame-per-file series and multi-frame DICOM
    (NumberOfFrames > 1, common for XA, US, NM and enhanced CT/MR).
    """
    if not files:
        raise ValueError("No files given.")

    datasets: list[FileDataset] = []
    for path in files:
        try:
            datasets.append(pydicom.dcmread(str(path), force=True))
        except Exception:
            continue

    if not datasets:
        raise ValueError("None of the provided files could be read as DICOM.")

    if len(datasets) == 1 and int(getattr(datasets[0], "NumberOfFrames", 1) or 1) > 1:
        return _load_multiframe(datasets[0])

    ref = datasets[0]
    row_dir, col_dir = _image_orientation(ref)
    slice_dir = np.cross(row_dir, col_dir)

    def _position_key(d: FileDataset) -> float:
        ipp = getattr(d, "ImagePositionPatient", None)
        if ipp is None or len(ipp) != 3:
            return float(getattr(d, "InstanceNumber", 0) or 0)
        pos = np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])])
        return float(np.dot(pos, slice_dir))

    datasets.sort(key=_position_key)

    slices: list[np.ndarray] = []
    for d in datasets:
        arr = d.pixel_array.astype(np.float32)
        slope = _maybe_float(getattr(d, "RescaleSlope", 1.0)) or 1.0
        intercept = _maybe_float(getattr(d, "RescaleIntercept", 0.0)) or 0.0
        slices.append(arr * slope + intercept)

    shape0 = slices[0].shape
    slices = [s for s in slices if s.shape == shape0]
    volume = np.stack(slices, axis=0).astype(np.float32)

    if len(datasets) >= 2:
        z0 = _position_key(datasets[0])
        z1 = _position_key(datasets[-1])
        slice_spacing = abs(z1 - z0) / max(len(datasets) - 1, 1)
        if slice_spacing <= 0:
            slice_spacing = _maybe_float(getattr(ref, "SliceThickness", 1.0)) or 1.0
    else:
        slice_spacing = _maybe_float(getattr(ref, "SliceThickness", 1.0)) or 1.0

    row_spacing, col_spacing = _pixel_spacing(ref)
    ipp = getattr(ref, "ImagePositionPatient", [0.0, 0.0, 0.0])
    origin = (float(ipp[0]), float(ipp[1]), float(ipp[2]))
    orientation = np.column_stack([col_dir, row_dir, slice_dir])

    modality = Modality.from_dicom(str(getattr(ref, "Modality", "") or ""))
    pet_meta = _collect_pet_metadata(ref) if modality == Modality.PT else None

    return Series(
        pixels=volume,
        spacing=(float(slice_spacing), float(row_spacing), float(col_spacing)),
        origin=origin,
        orientation=orientation,
        modality=modality,
        study_uid=str(getattr(ref, "StudyInstanceUID", "") or ""),
        series_uid=str(getattr(ref, "SeriesInstanceUID", "") or ""),
        series_description=str(getattr(ref, "SeriesDescription", "") or ""),
        patient_id=str(getattr(ref, "PatientID", "") or ""),
        patient_name=str(getattr(ref, "PatientName", "") or ""),
        study_date=str(getattr(ref, "StudyDate", "") or ""),
        pet=pet_meta,
    )


def _load_multiframe(ds: FileDataset) -> Series:
    """Load a single multi-frame DICOM (NumberOfFrames > 1) as a 3D Series."""
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"Unexpected multi-frame pixel shape: {arr.shape}")

    slope = _maybe_float(getattr(ds, "RescaleSlope", 1.0)) or 1.0
    intercept = _maybe_float(getattr(ds, "RescaleIntercept", 0.0)) or 0.0
    volume = arr * slope + intercept

    row_dir, col_dir = _image_orientation(ds)
    slice_dir = np.cross(row_dir, col_dir)
    row_spacing, col_spacing = _pixel_spacing(ds)

    spacing_between = (
        _maybe_float(getattr(ds, "SpacingBetweenSlices", None))
        or _maybe_float(getattr(ds, "SliceThickness", 1.0))
        or 1.0
    )

    ipp = getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])
    origin = (float(ipp[0]), float(ipp[1]), float(ipp[2]))
    orientation = np.column_stack([col_dir, row_dir, slice_dir])

    modality = Modality.from_dicom(str(getattr(ds, "Modality", "") or ""))
    pet_meta = _collect_pet_metadata(ds) if modality == Modality.PT else None

    return Series(
        pixels=volume.astype(np.float32),
        spacing=(float(spacing_between), float(row_spacing), float(col_spacing)),
        origin=origin,
        orientation=orientation,
        modality=modality,
        study_uid=str(getattr(ds, "StudyInstanceUID", "") or ""),
        series_uid=str(getattr(ds, "SeriesInstanceUID", "") or ""),
        series_description=str(getattr(ds, "SeriesDescription", "") or ""),
        patient_id=str(getattr(ds, "PatientID", "") or ""),
        patient_name=str(getattr(ds, "PatientName", "") or ""),
        study_date=str(getattr(ds, "StudyDate", "") or ""),
        pet=pet_meta,
        extra={"multiframe": True, "frames": int(volume.shape[0])},
    )


def load_path(path: Path) -> list[Series]:
    """Load every DICOM series found at `path` (file, folder, or DICOMDIR)."""
    if path.is_file():
        if path.name.upper() == "DICOMDIR":
            from dicom_reader.io.dicomdir import series_files_from_dicomdir

            grouped = series_files_from_dicomdir(path)
            return [load_series(files) for files in grouped.values() if files]
        return [load_series([path])]
    # Auto-detect a DICOMDIR at the root of a CD/DVD-style folder
    candidate = path / "DICOMDIR"
    if candidate.exists():
        from dicom_reader.io.dicomdir import series_files_from_dicomdir

        grouped = series_files_from_dicomdir(candidate)
        if grouped:
            return [load_series(files) for files in grouped.values() if files]
    groups = discover_series(path)
    return [load_series(files) for files in groups.values() if files]
