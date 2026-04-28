"""Multi-planar reformation.

Extracts axial, coronal, and sagittal slices from a Series. Interprets the
volume's voxel axes as (slice, row, col) = (axial, coronal-through-plane,
sagittal-through-plane) in the common case where the patient is scanned
roughly head-to-foot. The orientation matrix is stored on the Series for
future oblique reformations.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from dicom_reader.io.series import Series


class Plane(str, Enum):
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"


def extract_slice(series: Series, plane: Plane, index: int) -> np.ndarray:
    vol = series.pixels
    n_slices, n_rows, n_cols = vol.shape
    if plane == Plane.AXIAL:
        index = int(np.clip(index, 0, n_slices - 1))
        return vol[index, :, :]
    if plane == Plane.CORONAL:
        index = int(np.clip(index, 0, n_rows - 1))
        # flip vertically so head is at top
        return np.flipud(vol[:, index, :])
    if plane == Plane.SAGITTAL:
        index = int(np.clip(index, 0, n_cols - 1))
        return np.flipud(vol[:, :, index])
    raise ValueError(f"Unknown plane: {plane}")


def plane_extent(series: Series, plane: Plane) -> tuple[int, float, float]:
    """Return (n_slices, vertical_mm_per_pixel, horizontal_mm_per_pixel)."""
    slice_sp, row_sp, col_sp = series.spacing
    if plane == Plane.AXIAL:
        return series.shape[0], row_sp, col_sp
    if plane == Plane.CORONAL:
        return series.shape[1], slice_sp, col_sp
    if plane == Plane.SAGITTAL:
        return series.shape[2], slice_sp, row_sp
    raise ValueError(f"Unknown plane: {plane}")
