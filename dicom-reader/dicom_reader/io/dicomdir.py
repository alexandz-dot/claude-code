"""DICOMDIR support.

A DICOMDIR file is the directory record for a DICOM media (CD/DVD/USB).
It enumerates patients, studies, series, and the file paths (relative to
the DICOMDIR) of each instance. We expose a minimal browse + flatten API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pydicom


@dataclass
class DicomDirInstance:
    file_path: Path


@dataclass
class DicomDirSeries:
    series_uid: str
    description: str
    modality: str
    instances: list[DicomDirInstance] = field(default_factory=list)


@dataclass
class DicomDirStudy:
    study_uid: str
    description: str
    date: str
    series: list[DicomDirSeries] = field(default_factory=list)


@dataclass
class DicomDirPatient:
    patient_id: str
    patient_name: str
    studies: list[DicomDirStudy] = field(default_factory=list)


def _resolve_referenced(root_dir: Path, ref: list[str]) -> Path:
    """`ReferencedFileID` is a list of path segments, often with backslashes."""
    parts: list[str] = []
    for segment in ref:
        parts.extend(str(segment).replace("\\", "/").split("/"))
    candidate = root_dir.joinpath(*parts)
    if candidate.exists():
        return candidate
    # Some media use lowercase; try a case-insensitive lookup.
    cur = root_dir
    for part in parts:
        if not cur.is_dir():
            return candidate
        match = next((p for p in cur.iterdir() if p.name.lower() == part.lower()), None)
        if match is None:
            return candidate
        cur = match
    return cur


def parse_dicomdir(dicomdir_path: Path) -> list[DicomDirPatient]:
    """Parse a DICOMDIR and return its patient/study/series/instance tree."""
    ds = pydicom.dcmread(str(dicomdir_path), force=True)
    root_dir = dicomdir_path.parent
    records = getattr(ds, "DirectoryRecordSequence", None) or []

    patients: list[DicomDirPatient] = []
    cur_patient: DicomDirPatient | None = None
    cur_study: DicomDirStudy | None = None
    cur_series: DicomDirSeries | None = None

    for rec in records:
        rtype = (str(getattr(rec, "DirectoryRecordType", "") or "")).upper()
        if rtype == "PATIENT":
            cur_patient = DicomDirPatient(
                patient_id=str(getattr(rec, "PatientID", "") or ""),
                patient_name=str(getattr(rec, "PatientName", "") or ""),
            )
            patients.append(cur_patient)
            cur_study = None
            cur_series = None
        elif rtype == "STUDY":
            cur_study = DicomDirStudy(
                study_uid=str(getattr(rec, "StudyInstanceUID", "") or ""),
                description=str(getattr(rec, "StudyDescription", "") or ""),
                date=str(getattr(rec, "StudyDate", "") or ""),
            )
            if cur_patient is not None:
                cur_patient.studies.append(cur_study)
            cur_series = None
        elif rtype == "SERIES":
            cur_series = DicomDirSeries(
                series_uid=str(getattr(rec, "SeriesInstanceUID", "") or ""),
                description=str(getattr(rec, "SeriesDescription", "") or ""),
                modality=str(getattr(rec, "Modality", "") or ""),
            )
            if cur_study is not None:
                cur_study.series.append(cur_series)
        elif rtype == "IMAGE":
            ref = getattr(rec, "ReferencedFileID", None)
            if not ref or cur_series is None:
                continue
            cur_series.instances.append(
                DicomDirInstance(file_path=_resolve_referenced(root_dir, list(ref)))
            )
    return patients


def series_files_from_dicomdir(dicomdir_path: Path) -> dict[str, list[Path]]:
    """Flatten a DICOMDIR to {series_uid: [file paths]}."""
    out: dict[str, list[Path]] = {}
    for patient in parse_dicomdir(dicomdir_path):
        for study in patient.studies:
            for series in study.series:
                files = [inst.file_path for inst in series.instances if inst.file_path.exists()]
                if files:
                    out[series.series_uid] = files
    return out
