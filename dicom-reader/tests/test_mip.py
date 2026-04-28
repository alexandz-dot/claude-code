import numpy as np
import pytest

from dicom_reader.imaging.mip import SlabMode, slab_projection
from dicom_reader.imaging.reslice import Plane
from dicom_reader.io.series import Modality, Series


def _series(vol: np.ndarray) -> Series:
    return Series(
        pixels=vol.astype(np.float32),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        orientation=np.eye(3),
        modality=Modality.CT,
    )


def test_slab_max_picks_brightest_slice():
    # 5 axial slices, slice i has constant value i * 10
    vol = np.stack([np.full((4, 4), i * 10.0) for i in range(5)], axis=0)
    s = _series(vol)
    out = slab_projection(s, Plane.AXIAL, center_index=2, slab_thickness=5, mode=SlabMode.MAX)
    assert out.shape == (4, 4)
    assert np.all(out == 40.0)


def test_slab_min_picks_darkest_slice():
    vol = np.stack([np.full((4, 4), i * 10.0) for i in range(5)], axis=0)
    s = _series(vol)
    out = slab_projection(s, Plane.AXIAL, 2, 5, SlabMode.MIN)
    assert np.all(out == 0.0)


def test_slab_mean_averages_slices():
    vol = np.stack([np.full((2, 2), i * 2.0) for i in range(4)], axis=0)
    s = _series(vol)
    out = slab_projection(s, Plane.AXIAL, 1, 4, SlabMode.MEAN)
    assert np.allclose(out, np.full((2, 2), (0 + 2 + 4 + 6) / 4))


def test_slab_thickness_one_is_single_slice():
    vol = np.stack([np.full((3, 3), float(i)) for i in range(5)], axis=0)
    s = _series(vol)
    out = slab_projection(s, Plane.AXIAL, 2, 1, SlabMode.MAX)
    assert np.all(out == 2.0)


def test_coronal_slab_is_flipped_vertically():
    vol = np.zeros((3, 4, 5), dtype=np.float32)
    vol[2, :, :] = 99.0  # last axial slice bright
    s = _series(vol)
    out = slab_projection(s, Plane.CORONAL, 2, 1, SlabMode.MAX)
    # coronal flips top/bottom so top row comes from the last axial slice
    assert out[0, 0] == 99.0
