import numpy as np

from dicom_reader.tools.measure import (
    distance_mm,
    ellipse_roi_stats,
    rect_roi_stats,
)


def test_distance_axis_aligned():
    # 10 pixels apart vertically at 2 mm/pixel = 20 mm
    d = distance_mm((0, 0), (10, 0), spacing_v=2.0, spacing_h=1.0)
    assert d == 20.0


def test_distance_diagonal():
    d = distance_mm((0, 0), (3, 4), spacing_v=1.0, spacing_h=1.0)
    assert d == 5.0


def test_rect_roi_stats_constant():
    img = np.full((10, 10), 42.0, dtype=np.float32)
    stats = rect_roi_stats(img, (2, 2), (8, 8), spacing_v=1.0, spacing_h=1.0)
    assert stats.count == 36
    assert stats.mean == 42.0
    assert stats.std == 0.0
    assert stats.area_mm2 == 36.0


def test_ellipse_roi_stats_coverage():
    img = np.zeros((20, 20), dtype=np.float32)
    img[10, 10] = 100.0
    stats = ellipse_roi_stats(img, center=(10, 10), radii=(5, 5), spacing_v=1.0, spacing_h=1.0)
    assert stats.count > 0
    assert stats.maximum == 100.0
