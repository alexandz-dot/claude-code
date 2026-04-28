import numpy as np

from dicom_reader.imaging.windowing import CT_PRESETS, WindowLevel, apply_window, preset


def test_preset_known_values():
    lung = preset("Lung")
    assert lung.width == 1500 and lung.level == -600
    assert set(CT_PRESETS.keys()) >= {"Lung", "Bone", "Brain", "Soft Tissue"}


def test_apply_window_clips_and_scales():
    img = np.array([[-1000, -500, 0, 500, 1000]], dtype=np.float32)
    out = apply_window(img, WindowLevel(width=2000, level=0))
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    # monotonic
    assert list(out[0]) == sorted(out[0])


def test_apply_window_zero_width_guarded():
    img = np.array([[0, 1, 2]], dtype=np.float32)
    out = apply_window(img, WindowLevel(width=0, level=1))
    assert out.shape == img.shape
