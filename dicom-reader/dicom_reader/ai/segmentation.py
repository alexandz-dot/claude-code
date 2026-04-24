"""TotalSegmentator wrapper.

Runs on a loaded CT Series. The heavy model + PyTorch deps are optional —
they are only imported when `run_totalsegmentator` is called, so the viewer
works without them. Output is a SegmentationSet registered to the input CT.

Uses TotalSegmentator's Python API (`totalsegmentator.python_api.totalsegmentator`)
which accepts and emits NIfTI. We write the CT to a temp NIfTI, run the model,
and read back the multi-label NIfTI. No per-structure round-trip on disk.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from dicom_reader.ai.masks import SegmentationSet, Structure, generate_palette
from dicom_reader.io.series import Modality, Series


ProgressFn = Callable[[str, float], None]  # (message, fraction 0..1)


def is_available() -> bool:
    try:
        import totalsegmentator  # noqa: F401
        import nibabel  # noqa: F401
        return True
    except Exception:
        return False


def _series_to_nifti(series: Series, out_path: Path) -> None:
    import nibabel as nib  # type: ignore

    # Volume is (slice, row, col). NIfTI is typically (x, y, z) with x=col.
    vol = np.transpose(series.pixels, (2, 1, 0)).astype(np.float32)

    col_sp, row_sp, slice_sp = series.spacing[2], series.spacing[1], series.spacing[0]
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = col_sp
    affine[1, 1] = row_sp
    affine[2, 2] = slice_sp
    affine[:3, 3] = series.origin

    img = nib.Nifti1Image(vol, affine)
    nib.save(img, str(out_path))


def _nifti_to_label_volume(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    import nibabel as nib  # type: ignore

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    # back from (x, y, z) to (slice, row, col)
    vol = np.transpose(data, (2, 1, 0)).astype(np.uint16)
    if vol.shape != expected_shape:
        raise RuntimeError(
            f"Segmentation shape {vol.shape} does not match CT {expected_shape}"
        )
    return vol


def _structures_from_mapping(mapping: dict[int, str]) -> list[Structure]:
    items = sorted(mapping.items())
    colors = generate_palette(len(items))
    return [
        Structure(label_id=int(lbl), name=str(name), color_rgb=color)
        for (lbl, name), color in zip(items, colors)
    ]


def _label_mapping_from_totalseg(task: str) -> dict[int, str]:
    """Pull the (label_id -> organ name) mapping from TotalSegmentator's maps.

    The location of `class_map` has moved between TotalSegmentator versions,
    so we try a few known paths before giving up.
    """
    class_map = None
    for module_path in (
        "totalsegmentator.map_to_binary",
        "totalsegmentator.map_to_binary_v2",
        "totalsegmentator.libs",
    ):
        try:
            mod = __import__(module_path, fromlist=["class_map"])
            if hasattr(mod, "class_map"):
                class_map = mod.class_map
                break
        except Exception:
            continue
    if class_map is None:
        raise RuntimeError(
            "Cannot locate TotalSegmentator class map. "
            "Install or upgrade the `totalsegmentator` package."
        )
    if task not in class_map:
        raise KeyError(f"Unknown TotalSegmentator task: {task}")
    return {int(k): str(v) for k, v in class_map[task].items()}


def run_totalsegmentator(
    ct: Series,
    task: str = "total",
    fast: bool = True,
    progress: ProgressFn | None = None,
) -> SegmentationSet:
    """Segment a CT series with TotalSegmentator.

    `task="total"` segments ~100 structures. `fast=True` uses the fast
    low-resolution model (good default for interactive work).

    Raises ImportError if TotalSegmentator or nibabel are not installed.
    """
    if ct.modality != Modality.CT:
        raise ValueError("TotalSegmentator expects a CT series.")
    if not is_available():
        raise ImportError(
            "TotalSegmentator is not installed. "
            "Install with: pip install 'dicom-reader[ai]'"
        )

    from totalsegmentator.python_api import totalsegmentator  # type: ignore

    _emit(progress, "Exporting CT to NIfTI…", 0.05)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ct_nii = tmp_path / "ct.nii.gz"
        out_nii = tmp_path / "seg.nii.gz"
        _series_to_nifti(ct, ct_nii)

        _emit(progress, "Running TotalSegmentator…", 0.15)
        totalsegmentator(
            input=str(ct_nii),
            output=str(out_nii),
            task=task,
            fast=fast,
            ml=True,  # output multi-label single file
            quiet=True,
        )

        _emit(progress, "Loading segmentation…", 0.85)
        labels = _nifti_to_label_volume(out_nii, ct.shape)

    mapping = _label_mapping_from_totalseg(task)
    structures = _structures_from_mapping(mapping)
    # drop structures that are absent in this volume, to keep the panel tidy
    present = set(np.unique(labels).tolist())
    structures = [s for s in structures if s.label_id in present]

    _emit(progress, "Done.", 1.0)
    return SegmentationSet(
        labels=labels,
        reference=ct,
        structures=structures,
        source=f"TotalSegmentator ({task}, {'fast' if fast else 'full'})",
    )


def _emit(progress: ProgressFn | None, msg: str, frac: float) -> None:
    if progress is not None:
        progress(msg, max(0.0, min(1.0, frac)))


# Qt-facing runner lives in the UI layer to avoid importing Qt here.
