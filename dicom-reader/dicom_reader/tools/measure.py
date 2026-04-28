"""Measurement primitives: distance and ROI statistics."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


def distance_mm(
    p0: tuple[float, float],
    p1: tuple[float, float],
    spacing_v: float,
    spacing_h: float,
) -> float:
    """Euclidean distance between two image-space points, in millimeters.

    Points are (row, col) in pixel coordinates; spacings are mm/pixel on the
    displayed plane's vertical and horizontal axes.
    """
    dv = (p1[0] - p0[0]) * spacing_v
    dh = (p1[1] - p0[1]) * spacing_h
    return sqrt(dv * dv + dh * dh)


@dataclass(frozen=True)
class ROIStats:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    area_mm2: float


def _stats(values: np.ndarray, pixel_area_mm2: float) -> ROIStats:
    if values.size == 0:
        return ROIStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return ROIStats(
        count=int(values.size),
        mean=float(values.mean()),
        std=float(values.std()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        area_mm2=float(values.size) * pixel_area_mm2,
    )


def rect_roi_stats(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    spacing_v: float,
    spacing_h: float,
) -> ROIStats:
    r0, c0 = top_left
    r1, c1 = bottom_right
    r0, r1 = sorted((max(0, r0), min(image.shape[0], r1)))
    c0, c1 = sorted((max(0, c0), min(image.shape[1], c1)))
    if r1 <= r0 or c1 <= c0:
        return ROIStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    patch = image[r0:r1, c0:c1]
    return _stats(patch, spacing_v * spacing_h)


def ellipse_roi_stats(
    image: np.ndarray,
    center: tuple[float, float],
    radii: tuple[float, float],
    spacing_v: float,
    spacing_h: float,
) -> ROIStats:
    rr, cc = np.indices(image.shape)
    cy, cx = center
    ry, rx = radii
    if ry <= 0 or rx <= 0:
        return ROIStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mask = ((rr - cy) / ry) ** 2 + ((cc - cx) / rx) ** 2 <= 1.0
    values = image[mask]
    return _stats(values, spacing_v * spacing_h)
