"""Geometry helpers used by measurement tools and crosshair linking."""

from __future__ import annotations

from math import acos, degrees, sqrt

import numpy as np

from dicom_reader.imaging.reslice import Plane
from dicom_reader.io.series import Series


def angle_between(
    p0: tuple[float, float],
    vertex: tuple[float, float],
    p1: tuple[float, float],
    spacing_v: float = 1.0,
    spacing_h: float = 1.0,
) -> float:
    """Angle in degrees at `vertex` between segments to `p0` and `p1`.

    Points are (row, col) pixel coordinates; spacings convert to mm so the
    angle is computed in physical space (matters when pixels aren't square).
    """
    v0 = np.array([(p0[0] - vertex[0]) * spacing_v, (p0[1] - vertex[1]) * spacing_h])
    v1 = np.array([(p1[0] - vertex[0]) * spacing_v, (p1[1] - vertex[1]) * spacing_h])
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 == 0 or n1 == 0:
        return 0.0
    cos_a = float(np.dot(v0, v1) / (n0 * n1))
    cos_a = max(-1.0, min(1.0, cos_a))
    return degrees(acos(cos_a))


def cobb_angle(
    line_a: tuple[tuple[float, float], tuple[float, float]],
    line_b: tuple[tuple[float, float], tuple[float, float]],
    spacing_v: float = 1.0,
    spacing_h: float = 1.0,
) -> float:
    """Acute angle (deg) between two line segments — used for spinal Cobb."""
    (a0, a1) = line_a
    (b0, b1) = line_b
    da = np.array([(a1[0] - a0[0]) * spacing_v, (a1[1] - a0[1]) * spacing_h])
    db = np.array([(b1[0] - b0[0]) * spacing_v, (b1[1] - b0[1]) * spacing_h])
    na = float(np.linalg.norm(da))
    nb = float(np.linalg.norm(db))
    if na == 0 or nb == 0:
        return 0.0
    cos_a = abs(float(np.dot(da, db) / (na * nb)))
    cos_a = max(-1.0, min(1.0, cos_a))
    return degrees(acos(cos_a))


def polygon_area_mm2(
    vertices: list[tuple[float, float]],
    spacing_v: float,
    spacing_h: float,
) -> float:
    """Shoelace area for a closed polygon, scaled to mm²."""
    if len(vertices) < 3:
        return 0.0
    ys = np.array([v[0] for v in vertices], dtype=np.float64) * spacing_v
    xs = np.array([v[1] for v in vertices], dtype=np.float64) * spacing_h
    area_pixels = 0.5 * abs(
        float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    )
    return area_pixels


def polygon_mask(
    shape: tuple[int, int], vertices: list[tuple[float, float]]
) -> np.ndarray:
    """Rasterize a polygon to a bool mask of shape (rows, cols)."""
    if len(vertices) < 3:
        return np.zeros(shape, dtype=bool)
    ny, nx = shape
    yy, xx = np.indices(shape)
    inside = np.zeros(shape, dtype=bool)
    n = len(vertices)
    j = n - 1
    for i in range(n):
        yi, xi = vertices[i]
        yj, xj = vertices[j]
        cond = ((xi > xx) != (xj > xx)) & (
            yy < (yj - yi) * (xx - xi) / ((xj - xi) + 1e-12) + yi
        )
        inside ^= cond
        j = i
    return inside


# --- Crosshair linking ---


def patient_to_voxel(series: Series, point_xyz: np.ndarray) -> np.ndarray:
    """Convert a patient-coordinate point (mm) to (slice, row, col) voxel index."""
    rel = np.asarray(point_xyz, dtype=np.float64) - np.asarray(series.origin)
    inv = np.linalg.inv(series.orientation)
    idx_mm = inv @ rel
    return idx_mm / np.asarray(series.spacing)


def voxel_to_patient(series: Series, k: float, j: float, i: float) -> np.ndarray:
    """Inverse of `patient_to_voxel`."""
    return series.voxel_to_patient(k, j, i)


def plane_pixel_to_voxel(
    series: Series,
    plane: Plane,
    slice_index: int,
    row: float,
    col: float,
) -> tuple[float, float, float]:
    """Map a (row, col) click on a 2D plane to (k, j, i) volume voxel.

    Inverse of the slicing in `extract_slice` — including the vertical flip
    we apply to coronal/sagittal so head sits at the top.
    """
    n_slices, n_rows, n_cols = series.shape
    if plane == Plane.AXIAL:
        return (float(slice_index), float(row), float(col))
    if plane == Plane.CORONAL:
        # we displayed flipud(vol[:, j, :]) — undo the flip: shown_row = (n_slices-1)-k
        k = (n_slices - 1) - row
        return (float(k), float(slice_index), float(col))
    if plane == Plane.SAGITTAL:
        k = (n_slices - 1) - row
        return (float(k), float(col), float(slice_index))
    raise ValueError(plane)


def voxel_to_plane_pixel(
    series: Series, plane: Plane, k: float, j: float, i: float
) -> tuple[int, int, int]:
    """Inverse: which (slice_index, row, col) shows the given (k,j,i) on `plane`?"""
    n_slices, n_rows, n_cols = series.shape
    if plane == Plane.AXIAL:
        return (int(round(k)), int(round(j)), int(round(i)))
    if plane == Plane.CORONAL:
        return (int(round(j)), int(round((n_slices - 1) - k)), int(round(i)))
    if plane == Plane.SAGITTAL:
        return (int(round(i)), int(round((n_slices - 1) - k)), int(round(j)))
    raise ValueError(plane)
