"""CT window/level presets and display mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowLevel:
    width: float
    level: float

    @property
    def low(self) -> float:
        return self.level - self.width / 2.0

    @property
    def high(self) -> float:
        return self.level + self.width / 2.0


CT_PRESETS: dict[str, WindowLevel] = {
    "Soft Tissue": WindowLevel(400, 40),
    "Lung": WindowLevel(1500, -600),
    "Mediastinum": WindowLevel(350, 50),
    "Bone": WindowLevel(1800, 400),
    "Brain": WindowLevel(80, 40),
    "Liver": WindowLevel(150, 60),
    "Abdomen": WindowLevel(400, 50),
    "Angiography": WindowLevel(600, 300),
}


def preset(name: str) -> WindowLevel:
    if name not in CT_PRESETS:
        raise KeyError(f"Unknown CT preset: {name}")
    return CT_PRESETS[name]


def apply_window(image: np.ndarray, wl: WindowLevel) -> np.ndarray:
    """Map a 2D slice to uint8 grayscale using the given W/L."""
    low, high = wl.low, wl.high
    if high <= low:
        high = low + 1.0
    clipped = np.clip(image, low, high)
    scaled = (clipped - low) / (high - low) * 255.0
    return scaled.astype(np.uint8)
