"""Display-only transforms: invert, flip H/V, rotate 0/90/180/270.

Forward order applied to the rendered image: invert -> flip_v -> flip_h ->
rot90(k=rotations).

The viewport applies these only at the very end of rendering. Tools must
keep operating on original (row, col) coordinates, so we expose
`view_to_image_coords` as the inverse mapping for clicks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DisplayOptions:
    invert: bool = False
    flip_v: bool = False
    flip_h: bool = False
    rotations: int = 0  # number of CCW 90° turns; modulo 4

    def is_identity(self) -> bool:
        return (
            not self.invert
            and not self.flip_v
            and not self.flip_h
            and (self.rotations % 4) == 0
        )


def apply_display_transform(image: np.ndarray, opts: DisplayOptions) -> np.ndarray:
    """Apply display options to a 2D grayscale or 3D RGB image.

    Output is C-contiguous so it can be passed to QImage / pyqtgraph
    without a hidden copy.
    """
    out = image
    if opts.invert:
        if out.dtype == np.uint8:
            out = 255 - out
        else:
            out = -out
    if opts.flip_v:
        out = out[::-1, ...]
    if opts.flip_h:
        out = out[:, ::-1, ...]
    rotations = opts.rotations % 4
    if rotations:
        out = np.rot90(out, k=rotations)
    return np.ascontiguousarray(out)


def view_to_image_coords(
    yv: float,
    xv: float,
    image_shape: tuple[int, int],
    opts: DisplayOptions,
) -> tuple[float, float]:
    """Inverse coordinate mapping for `apply_display_transform`.

    `image_shape` is the ORIGINAL (rows, cols) before any transform.
    `(yv, xv)` is the click on the transformed image. Returns (yi, xi)
    in the original image. Invert is identity for coordinates.
    """
    H, W = image_shape[:2]
    rotations = opts.rotations % 4
    # Undo rotation first (it was applied last).
    if rotations == 0:
        y, x = float(yv), float(xv)
    elif rotations == 1:
        # rot90 k=1: rotated[i, j] = image[j, W-1-i] (shape becomes (W, H))
        y, x = float(xv), float(W - 1 - yv)
    elif rotations == 2:
        y, x = float(H - 1 - yv), float(W - 1 - xv)
    else:  # rotations == 3
        y, x = float(H - 1 - xv), float(yv)
    if opts.flip_h:
        x = (W - 1) - x
    if opts.flip_v:
        y = (H - 1) - y
    return y, x


def image_to_view_coords(
    yi: float,
    xi: float,
    image_shape: tuple[int, int],
    opts: DisplayOptions,
) -> tuple[float, float]:
    """Forward map: image-space (yi, xi) -> view-space (yv, xv) after `opts`."""
    H, W = image_shape[:2]
    y, x = float(yi), float(xi)
    if opts.flip_v:
        y = (H - 1) - y
    if opts.flip_h:
        x = (W - 1) - x
    k = opts.rotations % 4
    if k == 0:
        return y, x
    if k == 1:
        return (W - 1) - x, y
    if k == 2:
        return (H - 1) - y, (W - 1) - x
    return x, (H - 1) - y  # k == 3


def transformed_shape(
    image_shape: tuple[int, int], opts: DisplayOptions
) -> tuple[int, int]:
    """Shape of the image after applying `opts`. Useful for bounds checks."""
    H, W = image_shape[:2]
    if opts.rotations % 2 == 1:
        return (W, H)
    return (H, W)
