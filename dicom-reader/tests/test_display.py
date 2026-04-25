"""Round-trip tests for the display transform + inverse coord mapping."""

import numpy as np
import pytest

from dicom_reader.imaging.display import (
    DisplayOptions,
    apply_display_transform,
    image_to_view_coords,
    transformed_shape,
    view_to_image_coords,
)


def _img() -> np.ndarray:
    # 3x4 unique values so we can identify any pixel by its content.
    return np.arange(12, dtype=np.uint8).reshape(3, 4)


@pytest.mark.parametrize(
    "opts",
    [
        DisplayOptions(),
        DisplayOptions(invert=True),
        DisplayOptions(flip_v=True),
        DisplayOptions(flip_h=True),
        DisplayOptions(flip_v=True, flip_h=True),
        DisplayOptions(rotations=1),
        DisplayOptions(rotations=2),
        DisplayOptions(rotations=3),
        DisplayOptions(flip_v=True, rotations=1),
        DisplayOptions(flip_h=True, rotations=3),
        DisplayOptions(flip_v=True, flip_h=True, rotations=2),
    ],
)
def test_view_to_image_coords_round_trips(opts: DisplayOptions):
    img = _img()
    # locate each original pixel on the transformed canvas using the
    # geometric-only transform (invert would change pixel values)
    geom_only = DisplayOptions(
        flip_v=opts.flip_v, flip_h=opts.flip_h, rotations=opts.rotations
    )
    transformed = apply_display_transform(img, geom_only)
    H, W = img.shape
    for yi in range(H):
        for xi in range(W):
            mask = transformed == img[yi, xi]
            ys, xs = np.where(mask)
            yv, xv = int(ys[0]), int(xs[0])
            yi2, xi2 = view_to_image_coords(yv, xv, img.shape, opts)
            assert (round(yi2), round(xi2)) == (yi, xi), (
                f"opts={opts} yi={yi} xi={xi} -> view ({yv},{xv}) -> back ({yi2},{xi2})"
            )


@pytest.mark.parametrize(
    "opts",
    [
        DisplayOptions(),
        DisplayOptions(flip_v=True),
        DisplayOptions(flip_h=True),
        DisplayOptions(rotations=1),
        DisplayOptions(rotations=2),
        DisplayOptions(rotations=3),
        DisplayOptions(flip_v=True, flip_h=True, rotations=1),
        DisplayOptions(flip_h=True, rotations=3),
    ],
)
def test_image_view_round_trip_inverse_of_view_image(opts: DisplayOptions):
    img = _img()
    H, W = img.shape
    for yi in range(H):
        for xi in range(W):
            yv, xv = image_to_view_coords(yi, xi, img.shape, opts)
            yi2, xi2 = view_to_image_coords(yv, xv, img.shape, opts)
            assert (round(yi2), round(xi2)) == (yi, xi), (
                f"opts={opts}  ({yi},{xi}) -> ({yv},{xv}) -> ({yi2},{xi2})"
            )


def test_invert_uint8_inverts_intensity():
    img = np.array([[0, 128, 255]], dtype=np.uint8)
    out = apply_display_transform(img, DisplayOptions(invert=True))
    assert list(out.ravel()) == [255, 127, 0]


def test_transformed_shape_swaps_for_quarter_turns():
    assert transformed_shape((3, 4), DisplayOptions()) == (3, 4)
    assert transformed_shape((3, 4), DisplayOptions(rotations=1)) == (4, 3)
    assert transformed_shape((3, 4), DisplayOptions(rotations=2)) == (3, 4)
    assert transformed_shape((3, 4), DisplayOptions(rotations=3)) == (4, 3)


def test_identity_helper():
    assert DisplayOptions().is_identity()
    assert not DisplayOptions(invert=True).is_identity()
    assert not DisplayOptions(rotations=4).is_identity() is False  # rotations=4 -> 0
    assert DisplayOptions(rotations=4).is_identity()


def test_apply_returns_contiguous():
    out = apply_display_transform(_img(), DisplayOptions(rotations=1, flip_v=True))
    assert out.flags["C_CONTIGUOUS"]
