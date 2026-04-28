import numpy as np
import pytest

from dicom_reader.imaging.geometry import (
    angle_between,
    cobb_angle,
    plane_pixel_to_voxel,
    polygon_area_mm2,
    polygon_mask,
    voxel_to_plane_pixel,
)
from dicom_reader.imaging.reslice import Plane
from dicom_reader.io.series import Modality, Series


def test_angle_right_angle():
    # vertex at origin, arms along y and x
    a = angle_between((10, 0), (0, 0), (0, 10))
    assert a == pytest.approx(90.0, abs=1e-6)


def test_angle_zero_when_collinear():
    a = angle_between((0, 5), (0, 0), (0, 10))
    assert a == pytest.approx(0.0, abs=1e-6)


def test_angle_accounts_for_pixel_spacing():
    # Without spacing correction pixels appear square; with 2x vertical spacing
    # a 1:1 pixel right triangle becomes 1:2 in mm and the angle changes.
    a_no = angle_between((1, 0), (0, 0), (0, 1), spacing_v=1.0, spacing_h=1.0)
    a_yes = angle_between((1, 0), (0, 0), (0, 1), spacing_v=2.0, spacing_h=1.0)
    assert a_no == pytest.approx(90.0)
    assert a_yes == pytest.approx(90.0)  # still 90 because arms are orthogonal
    # Non-orthogonal case where spacing matters
    a1 = angle_between((1, 1), (0, 0), (0, 1), spacing_v=1.0, spacing_h=1.0)
    a2 = angle_between((1, 1), (0, 0), (0, 1), spacing_v=2.0, spacing_h=1.0)
    assert a1 != pytest.approx(a2)


def test_cobb_acute_and_reflex():
    line_a = ((0, 0), (0, 10))
    line_b = ((0, 0), (10, 0))
    assert cobb_angle(line_a, line_b) == pytest.approx(90.0, abs=1e-6)
    # Near-parallel should give ~0
    c = cobb_angle(((0, 0), (0, 10)), ((0, 1), (0, 11)))
    assert c == pytest.approx(0.0, abs=1e-6)


def test_polygon_area_unit_square():
    verts = [(0, 0), (0, 10), (10, 10), (10, 0)]
    assert polygon_area_mm2(verts, 1.0, 1.0) == pytest.approx(100.0)
    # Scale by spacing
    assert polygon_area_mm2(verts, 2.0, 1.0) == pytest.approx(200.0)


def test_polygon_area_degenerate():
    assert polygon_area_mm2([(0, 0), (1, 1)], 1.0, 1.0) == 0.0


def test_polygon_mask_covers_interior():
    # 10x10 square polygon in a 20x20 grid
    verts = [(5, 5), (5, 14), (14, 14), (14, 5)]
    mask = polygon_mask((20, 20), verts)
    # center of polygon is inside
    assert mask[10, 10]
    # corner of the image is outside
    assert not mask[0, 0]
    # total count should be close to the area
    assert mask.sum() > 50


def _series(shape=(4, 5, 6)) -> Series:
    return Series(
        pixels=np.zeros(shape, dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        orientation=np.eye(3),
        modality=Modality.CT,
    )


def test_plane_pixel_to_voxel_roundtrip_axial():
    s = _series()
    # axial: (slice_index=2, row=3, col=4) corresponds to volume (2, 3, 4)
    k, j, i = plane_pixel_to_voxel(s, Plane.AXIAL, 2, 3, 4)
    assert (k, j, i) == (2.0, 3.0, 4.0)
    assert voxel_to_plane_pixel(s, Plane.AXIAL, k, j, i) == (2, 3, 4)


def test_plane_pixel_to_voxel_roundtrip_coronal_is_flipped():
    s = _series(shape=(4, 5, 6))  # n_slices=4
    # In coronal view, shown_row 0 is the last axial slice (k=n_slices-1=3)
    k, j, i = plane_pixel_to_voxel(s, Plane.CORONAL, slice_index=2, row=0, col=5)
    assert (k, j, i) == (3.0, 2.0, 5.0)
    back = voxel_to_plane_pixel(s, Plane.CORONAL, k, j, i)
    assert back == (2, 0, 5)
