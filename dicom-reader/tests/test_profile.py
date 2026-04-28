import numpy as np
import pytest

from dicom_reader.imaging.profile import sample_line


def test_profile_horizontal_line_constant():
    img = np.full((10, 10), 50.0, dtype=np.float32)
    prof = sample_line(img, (5, 0), (5, 9), spacing_v=1.0, spacing_h=1.0)
    assert prof.values.shape == prof.distances_mm.shape
    assert np.allclose(prof.values, 50.0)
    # Start distance is zero, end distance is 9 mm
    assert prof.distances_mm[0] == pytest.approx(0.0)
    assert prof.distances_mm[-1] == pytest.approx(9.0)


def test_profile_vertical_line_gradient():
    img = np.arange(100).reshape(10, 10).astype(np.float32)
    # image[r, c] = r*10 + c. A vertical line at col=0 goes 0, 10, 20,...
    prof = sample_line(img, (0, 0), (9, 0), spacing_v=2.0, spacing_h=1.0, samples=10)
    assert prof.values[0] == pytest.approx(0.0)
    assert prof.values[-1] == pytest.approx(90.0)
    # Distances are in mm using spacing_v=2
    assert prof.distances_mm[-1] == pytest.approx(18.0)


def test_profile_applies_spacing_to_diagonal():
    img = np.zeros((5, 5), dtype=np.float32)
    # 3 rows x 4 cols -> 9 mm x 16 mm -> sqrt(337) = 18.358 mm
    prof = sample_line(img, (0, 0), (3, 4), spacing_v=3.0, spacing_h=4.0, samples=2)
    assert prof.distances_mm[0] == pytest.approx(0.0)
    assert prof.distances_mm[-1] == pytest.approx(np.sqrt(9**2 + 16**2), rel=1e-5)
