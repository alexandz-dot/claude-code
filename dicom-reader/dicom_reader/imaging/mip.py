"""Thick-slab maximum/minimum intensity projection.

For a given center slice and slab thickness (in slices), reduce the slab
along the through-plane axis with max (MIP), min (MinIP), or mean. Works on
axial, coronal, or sagittal planes.

MIP is particularly useful for:
  - CT angiography (bright vessels over soft tissue)
  - PET (bright tumor foci over background) — whole-body MIP is a standard
    reading view
MinIP is useful for airway / lung fissure visualization.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from dicom_reader.imaging.reslice import Plane
from dicom_reader.io.series import Series


class SlabMode(str, Enum):
    MAX = "max"
    MIN = "min"
    MEAN = "mean"


def slab_projection(
    series: Series,
    plane: Plane,
    center_index: int,
    slab_thickness: int,
    mode: SlabMode = SlabMode.MAX,
) -> np.ndarray:
    """Reduce a slab of `slab_thickness` slices centered on `center_index`.

    `slab_thickness` is in voxels along the through-plane axis of `plane`.
    Minimum effective thickness is 1 (returns the single slice).
    """
    vol = series.pixels
    n_slices, n_rows, n_cols = vol.shape
    t = max(1, int(slab_thickness))
    half = t // 2

    if plane == Plane.AXIAL:
        lo = max(0, center_index - half)
        hi = min(n_slices, lo + t)
        block = vol[lo:hi, :, :]
        axis = 0
    elif plane == Plane.CORONAL:
        lo = max(0, center_index - half)
        hi = min(n_rows, lo + t)
        block = vol[:, lo:hi, :]
        axis = 1
    elif plane == Plane.SAGITTAL:
        lo = max(0, center_index - half)
        hi = min(n_cols, lo + t)
        block = vol[:, :, lo:hi]
        axis = 2
    else:
        raise ValueError(f"Unknown plane: {plane}")

    if block.size == 0:
        return np.zeros(vol.shape[1:] if axis == 0 else (vol.shape[0], vol.shape[2]) if axis == 1 else (vol.shape[0], vol.shape[1]))

    if mode == SlabMode.MAX:
        out = block.max(axis=axis)
    elif mode == SlabMode.MIN:
        out = block.min(axis=axis)
    else:
        out = block.mean(axis=axis)

    if plane in (Plane.CORONAL, Plane.SAGITTAL):
        out = np.flipud(out)
    return out.astype(np.float32)
