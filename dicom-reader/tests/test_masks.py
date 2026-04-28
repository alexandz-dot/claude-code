import numpy as np
import pytest

from dicom_reader.ai.masks import SegmentationSet, Structure, generate_palette
from dicom_reader.io.series import Modality, Series


def _ct(shape=(3, 4, 5)) -> Series:
    return Series(
        pixels=np.arange(np.prod(shape)).reshape(shape).astype(np.float32),
        spacing=(1.0, 2.0, 3.0),
        origin=(0.0, 0.0, 0.0),
        orientation=np.eye(3),
        modality=Modality.CT,
    )


def test_palette_is_deterministic_and_high_contrast():
    a = generate_palette(10, seed=0)
    b = generate_palette(10, seed=0)
    assert a == b
    assert len(set(a)) == 10  # all distinct
    for r, g, b_ in a:
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b_ <= 255


def test_segmentation_set_rejects_shape_mismatch():
    ct = _ct(shape=(2, 2, 2))
    bad = np.zeros((3, 2, 2), dtype=np.uint16)
    with pytest.raises(ValueError):
        SegmentationSet(labels=bad, reference=ct)


def test_structure_stats_volume_and_mean():
    ct = _ct(shape=(2, 2, 2))
    labels = np.zeros((2, 2, 2), dtype=np.uint16)
    labels[0, 0, 0] = 1
    labels[1, 1, 1] = 1
    seg = SegmentationSet(
        labels=labels,
        reference=ct,
        structures=[Structure(1, "spleen", (255, 0, 0))],
    )
    stats = seg.structure_stats(1)
    assert stats is not None
    # 2 voxels at 1 * 2 * 3 = 6 mm^3 each -> 12 mm^3 = 0.012 ml
    assert stats["volume_ml"] == pytest.approx(0.012)
    # Values at (0,0,0)=0 and (1,1,1)=7 -> mean = 3.5
    assert stats["mean"] == pytest.approx(3.5)


def test_missing_label_returns_none():
    ct = _ct(shape=(2, 2, 2))
    seg = SegmentationSet(
        labels=np.zeros((2, 2, 2), dtype=np.uint16),
        reference=ct,
        structures=[Structure(1, "liver", (1, 2, 3))],
    )
    assert seg.structure_stats(1) is None
