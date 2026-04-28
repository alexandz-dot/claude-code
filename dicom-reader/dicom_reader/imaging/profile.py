"""Sample voxel intensity along an arbitrary line in a 2D slice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class LineProfile:
    distances_mm: np.ndarray
    values: np.ndarray


def sample_line(
    image: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    spacing_v: float,
    spacing_h: float,
    samples: int | None = None,
    order: int = 1,
) -> LineProfile:
    """Linearly interpolate `image` along the line p0 -> p1.

    Points are (row, col). `samples` defaults to the pixel-distance length so
    one sample per ~1 px is taken.
    """
    y0, x0 = p0
    y1, x1 = p1
    pixel_dist = float(np.hypot(y1 - y0, x1 - x0))
    n = max(2, int(samples if samples is not None else pixel_dist))
    ys = np.linspace(y0, y1, n)
    xs = np.linspace(x0, x1, n)
    vals = map_coordinates(image, np.vstack([ys, xs]), order=order, mode="nearest")
    # mm-distance from start
    dy = (ys - y0) * spacing_v
    dx = (xs - x0) * spacing_h
    dist = np.sqrt(dy * dy + dx * dx)
    return LineProfile(distances_mm=dist.astype(np.float32), values=vals.astype(np.float32))
