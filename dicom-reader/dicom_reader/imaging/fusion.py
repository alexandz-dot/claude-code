"""PET/CT fusion.

Resamples the PET volume into the CT grid using patient-coordinate sampling,
then blends the two as RGB. Nearest-neighbor by default for speed; linear
available via scipy.
"""

from __future__ import annotations

import numpy as np

from dicom_reader.io.series import Series


def _world_to_voxel(series: Series, points: np.ndarray) -> np.ndarray:
    """Convert (N, 3) patient (x, y, z) mm to (N, 3) voxel index (k, j, i)."""
    origin = np.asarray(series.origin)
    spacing = np.asarray(series.spacing)
    rel = points - origin
    inv = np.linalg.inv(series.orientation)
    idx_mm = rel @ inv.T
    return idx_mm / spacing


def resample_to(
    source: Series, target: Series, order: int = 1
) -> np.ndarray:
    """Resample `source` onto the grid of `target`. Returns a float32 array."""
    from scipy.ndimage import map_coordinates

    tk, tj, ti = np.indices(target.shape)
    k = tk.ravel().astype(np.float32)
    j = tj.ravel().astype(np.float32)
    i = ti.ravel().astype(np.float32)
    voxel = np.stack([k, j, i], axis=1)
    target_spacing = np.asarray(target.spacing)
    world_rel = voxel * target_spacing
    world = world_rel @ target.orientation.T + np.asarray(target.origin)

    src_voxel = _world_to_voxel(source, world)
    coords = src_voxel.T  # shape (3, N)
    sampled = map_coordinates(
        source.pixels, coords, order=order, mode="constant", cval=0.0, prefilter=False
    )
    return sampled.reshape(target.shape).astype(np.float32)


def _apply_colormap(values: np.ndarray, cmap: str) -> np.ndarray:
    """Map a (H, W) array in [0, 1] to an (H, W, 3) RGB uint8 array."""
    v = np.clip(values, 0.0, 1.0)
    if cmap == "hot":
        r = np.clip(v * 3.0, 0.0, 1.0)
        g = np.clip(v * 3.0 - 1.0, 0.0, 1.0)
        b = np.clip(v * 3.0 - 2.0, 0.0, 1.0)
    elif cmap == "gray":
        r = g = b = v
    else:  # "pet" — common PET rainbow, simplified
        r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def fuse_pet_ct(
    ct_slice_u8: np.ndarray,
    pet_slice: np.ndarray,
    pet_vmax: float,
    alpha: float = 0.4,
    cmap: str = "hot",
    threshold: float = 0.1,
) -> np.ndarray:
    """Blend a windowed CT slice (uint8) with a PET slice (float) as RGB.

    Values of PET below `threshold * pet_vmax` are treated as transparent.
    """
    if ct_slice_u8.ndim != 2:
        raise ValueError("CT slice must be 2D.")
    if pet_slice.shape != ct_slice_u8.shape:
        raise ValueError(
            f"PET and CT shapes disagree: {pet_slice.shape} vs {ct_slice_u8.shape}"
        )
    vmax = pet_vmax if pet_vmax > 0 else 1.0
    pet_norm = pet_slice / vmax
    pet_rgb = _apply_colormap(pet_norm, cmap).astype(np.float32)
    ct_rgb = np.repeat(ct_slice_u8[..., None], 3, axis=-1).astype(np.float32)
    mask = (pet_norm >= threshold).astype(np.float32)[..., None]
    blended = ct_rgb * (1.0 - alpha * mask) + pet_rgb * (alpha * mask)
    return np.clip(blended, 0, 255).astype(np.uint8)
