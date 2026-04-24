"""Mask data model and palette for anatomical segmentations.

A SegmentationSet owns a label volume (uint16) registered to a parent Series
and a list of `Structure` metadata (name, label id, color). Each structure
can be toggled on/off and recolored without touching the volume itself.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np

from dicom_reader.io.series import Series


@dataclass
class Structure:
    label_id: int
    name: str
    color_rgb: tuple[int, int, int]
    visible: bool = True


@dataclass
class SegmentationSet:
    """Label volume registered to `reference` (same shape, spacing, grid)."""

    labels: np.ndarray  # shape == reference.shape, dtype uint16
    reference: Series
    structures: list[Structure] = field(default_factory=list)
    source: str = ""  # e.g. "TotalSegmentator v2"

    def __post_init__(self) -> None:
        if self.labels.shape != self.reference.shape:
            raise ValueError(
                f"Label shape {self.labels.shape} != reference {self.reference.shape}"
            )
        if self.labels.dtype != np.uint16:
            self.labels = self.labels.astype(np.uint16)

    def by_id(self, label_id: int) -> Structure | None:
        for s in self.structures:
            if s.label_id == label_id:
                return s
        return None

    def structure_mask(self, label_id: int) -> np.ndarray:
        return self.labels == label_id

    def structure_stats(self, label_id: int) -> dict[str, float] | None:
        """Mean / min / max / std of the reference volume inside one structure."""
        mask = self.structure_mask(label_id)
        count = int(mask.sum())
        if count == 0:
            return None
        values = self.reference.pixels[mask]
        spacing = self.reference.spacing
        volume_ml = count * spacing[0] * spacing[1] * spacing[2] / 1000.0
        return {
            "count": float(count),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "volume_ml": float(volume_ml),
        }


def generate_palette(n: int, seed: int = 0) -> list[tuple[int, int, int]]:
    """Deterministic high-contrast palette using golden-ratio hue spacing."""
    golden = 0.61803398875
    colors: list[tuple[int, int, int]] = []
    h = (seed * golden) % 1.0
    for _ in range(n):
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
        h = (h + golden) % 1.0
    return colors
