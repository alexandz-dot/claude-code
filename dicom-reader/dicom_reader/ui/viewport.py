"""A single-plane DICOM viewport built on pyqtgraph.

Responsibilities:
  - render the current slice (grayscale CT or RGB fused)
  - expose signals for slice change, mouse probe, ROI, and distance tools
  - compute pixel value, HU, and SUV at cursor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from dicom_reader.ai.masks import SegmentationSet
from dicom_reader.imaging.geometry import (
    plane_pixel_to_voxel,
    voxel_to_plane_pixel,
)
from dicom_reader.imaging.mip import SlabMode, slab_projection
from dicom_reader.imaging.reslice import Plane, extract_slice, plane_extent
from dicom_reader.imaging.windowing import WindowLevel, apply_window
from dicom_reader.imaging.fusion import fuse_pet_ct
from dicom_reader.io.series import Modality, Series
from dicom_reader.tools.measure import (
    ROIStats,
    distance_mm,
    ellipse_roi_stats,
    rect_roi_stats,
)


@dataclass
class ViewState:
    wl: WindowLevel
    plane: Plane
    slice_index: int
    pet_vmax: float = 10.0
    fusion_alpha: float = 0.4
    fusion_cmap: str = "hot"
    fusion_threshold: float = 0.1
    suv_factor: float | None = None  # Bq/ml -> SUV, if applicable
    slab_thickness: int = 1  # 1 => single-slice
    slab_mode: SlabMode = SlabMode.MAX
    seg_alpha: float = 0.45
    crosshair_voxel: tuple[float, float, float] | None = None  # (k, j, i) volume


class Viewport(QtWidgets.QWidget):
    sliceChanged = QtCore.pyqtSignal(int)
    probed = QtCore.pyqtSignal(str)
    measured = QtCore.pyqtSignal(str)
    # Volume-voxel tuple (k, j, i). Emitted when the user picks a crosshair
    # point on this viewport; the main window rebroadcasts to the others.
    crosshairMoved = QtCore.pyqtSignal(float, float, float)

    def __init__(self, plane: Plane, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._plane = plane
        self._series: Series | None = None
        self._pet_overlay: np.ndarray | None = None  # PET volume resampled to CT grid
        self._segmentation: SegmentationSet | None = None
        self._state = ViewState(WindowLevel(400, 40), plane, 0)

        self._tool = "probe"  # probe | crosshair | distance | rect | ellipse
        self._tool_points: list[tuple[float, float]] = []
        self._last_roi_item: pg.GraphicsObject | None = None
        self._last_line_item: pg.GraphicsObject | None = None
        self._crosshair_h: pg.InfiniteLine | None = None
        self._crosshair_v: pg.InfiniteLine | None = None

        self._plot = pg.PlotWidget()
        self._plot.setBackground("k")
        self._plot.setAspectLocked(True)
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        self._plot.setMenuEnabled(False)

        self._img_item = pg.ImageItem(axisOrder="row-major")
        self._plot.addItem(self._img_item)

        self._label = QtWidgets.QLabel(plane.value.title())
        self._label.setStyleSheet("color:#eee;background:#222;padding:2px 6px;")

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(self._label)
        top_bar.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(top_bar)
        layout.addWidget(self._plot, 1)
        layout.addWidget(self._slider)

        self._plot.scene().sigMouseMoved.connect(self._on_mouse_move)
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_click)

    # --- API ---

    def set_series(self, series: Series, pet_overlay: np.ndarray | None = None) -> None:
        self._series = series
        self._pet_overlay = pet_overlay
        n_slices, _, _ = plane_extent(series, self._plane)
        self._state.slice_index = n_slices // 2
        if series.modality == Modality.CT:
            self._state.wl = WindowLevel(400, 40)
        elif series.modality == Modality.PT:
            vmax = float(np.percentile(series.pixels, 99.5)) or 1.0
            self._state.wl = WindowLevel(vmax, vmax / 2.0)
            self._state.pet_vmax = vmax
        self._slider.setEnabled(True)
        self._slider.setRange(0, n_slices - 1)
        self._slider.setValue(self._state.slice_index)
        self.refresh()

    def set_window(self, wl: WindowLevel) -> None:
        self._state.wl = wl
        self.refresh()

    def set_slice(self, index: int) -> None:
        self._slider.setValue(int(index))

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        self._tool_points.clear()
        self._clear_overlay_items()

    def set_fusion(
        self,
        pet_overlay: np.ndarray | None,
        alpha: float,
        cmap: str,
        threshold: float,
        vmax: float,
    ) -> None:
        self._pet_overlay = pet_overlay
        self._state.fusion_alpha = alpha
        self._state.fusion_cmap = cmap
        self._state.fusion_threshold = threshold
        self._state.pet_vmax = vmax
        self.refresh()

    def set_suv_factor(self, factor: float | None) -> None:
        self._state.suv_factor = factor

    def set_segmentation(self, segmentation: SegmentationSet | None) -> None:
        self._segmentation = segmentation
        self.refresh()

    def set_seg_alpha(self, alpha: float) -> None:
        self._state.seg_alpha = float(alpha)
        self.refresh()

    def set_slab(self, thickness: int, mode: SlabMode) -> None:
        self._state.slab_thickness = max(1, int(thickness))
        self._state.slab_mode = mode
        self.refresh()

    def set_crosshair_voxel(self, voxel: tuple[float, float, float] | None) -> None:
        """Pin the crosshair to a volume-voxel (k, j, i). Auto-snaps the slice."""
        self._state.crosshair_voxel = voxel
        if voxel is not None and self._series is not None:
            slice_idx, _, _ = voxel_to_plane_pixel(
                self._series, self._plane, voxel[0], voxel[1], voxel[2]
            )
            if 0 <= slice_idx <= self._slider.maximum():
                # Block the change signal; we are reacting to another viewport.
                self._slider.blockSignals(True)
                self._slider.setValue(slice_idx)
                self._slider.blockSignals(False)
                self._state.slice_index = slice_idx
        self.refresh()

    # --- Rendering ---

    def refresh(self) -> None:
        if self._series is None:
            return
        slice_ = self._extract_display_slice()
        gray = apply_window(slice_, self._state.wl)
        if self._pet_overlay is not None and self._series.modality == Modality.CT:
            pet_slice = self._extract_overlay_slice()
            rgb = fuse_pet_ct(
                gray,
                pet_slice,
                self._state.pet_vmax,
                self._state.fusion_alpha,
                self._state.fusion_cmap,
                self._state.fusion_threshold,
            )
        else:
            rgb = np.repeat(gray[..., None], 3, axis=-1)
        if self._segmentation is not None:
            rgb = self._composite_segmentation(rgb)
        self._img_item.setImage(rgb, autoLevels=False)
        self._update_crosshair()
        slab = self._state.slab_thickness
        slab_label = f"  slab {slab} ({self._state.slab_mode.value})" if slab > 1 else ""
        self._label.setText(
            f"{self._plane.value.title()}  "
            f"{self._state.slice_index + 1}/{self._slider.maximum() + 1}  "
            f"W/L {int(self._state.wl.width)}/{int(self._state.wl.level)}"
            f"{slab_label}"
        )

    def _update_crosshair(self) -> None:
        voxel = self._state.crosshair_voxel
        if voxel is None or self._series is None:
            self._hide_crosshair()
            return
        slice_idx, row, col = voxel_to_plane_pixel(
            self._series, self._plane, voxel[0], voxel[1], voxel[2]
        )
        # Only draw if the crosshair point is actually on the current slice
        if slice_idx != self._state.slice_index:
            self._hide_crosshair()
            return
        pen = pg.mkPen(color=(255, 235, 0, 200), width=1, style=QtCore.Qt.PenStyle.DashLine)
        if self._crosshair_h is None:
            self._crosshair_h = pg.InfiniteLine(angle=0, pen=pen, movable=False)
            self._crosshair_v = pg.InfiniteLine(angle=90, pen=pen, movable=False)
            self._plot.addItem(self._crosshair_h)
            self._plot.addItem(self._crosshair_v)
        self._crosshair_h.setPos(row)
        self._crosshair_v.setPos(col)

    def _hide_crosshair(self) -> None:
        if self._crosshair_h is not None:
            self._plot.removeItem(self._crosshair_h)
            self._crosshair_h = None
        if self._crosshair_v is not None:
            self._plot.removeItem(self._crosshair_v)
            self._crosshair_v = None

    def _extract_display_slice(self) -> np.ndarray:
        assert self._series is not None
        if self._state.slab_thickness > 1:
            return slab_projection(
                self._series,
                self._plane,
                self._state.slice_index,
                self._state.slab_thickness,
                self._state.slab_mode,
            )
        return extract_slice(self._series, self._plane, self._state.slice_index)

    def _composite_segmentation(self, rgb: np.ndarray) -> np.ndarray:
        assert self._segmentation is not None and self._series is not None
        if self._segmentation.reference.shape != self._series.shape:
            return rgb  # not registered to this series
        seg = Series(
            pixels=self._segmentation.labels.astype(np.float32),
            spacing=self._series.spacing,
            origin=self._series.origin,
            orientation=self._series.orientation,
            modality=Modality.OTHER,
        )
        label_slice = extract_slice(seg, self._plane, self._state.slice_index).astype(np.uint16)

        alpha = float(self._state.seg_alpha)
        if alpha <= 0:
            return rgb
        out = rgb.astype(np.float32)
        visible_ids = [s for s in self._segmentation.structures if s.visible]
        if not visible_ids:
            return rgb
        for structure in visible_ids:
            mask = label_slice == structure.label_id
            if not mask.any():
                continue
            color = np.array(structure.color_rgb, dtype=np.float32)
            m3 = mask[..., None]
            out = np.where(m3, out * (1.0 - alpha) + color * alpha, out)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _extract_overlay_slice(self) -> np.ndarray:
        assert self._series is not None and self._pet_overlay is not None
        tmp = Series(
            pixels=self._pet_overlay,
            spacing=self._series.spacing,
            origin=self._series.origin,
            orientation=self._series.orientation,
            modality=Modality.PT,
        )
        return extract_slice(tmp, self._plane, self._state.slice_index)

    # --- Events ---

    def _on_slider(self, value: int) -> None:
        self._state.slice_index = int(value)
        self.sliceChanged.emit(self._state.slice_index)
        self.refresh()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._slider.isEnabled():
            step = 1 if event.angleDelta().y() < 0 else -1
            self._slider.setValue(self._slider.value() + step)
        event.accept()

    def _image_coords(self, scene_pos: QtCore.QPointF) -> tuple[int, int] | None:
        vb = self._plot.getPlotItem().getViewBox()
        if vb is None:
            return None
        pt = vb.mapSceneToView(scene_pos)
        x = int(round(pt.x()))
        y = int(round(pt.y()))
        if self._series is None:
            return None
        shape = extract_slice(self._series, self._plane, self._state.slice_index).shape
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            return y, x
        return None

    def _on_mouse_move(self, scene_pos: QtCore.QPointF) -> None:
        if self._series is None:
            return
        coord = self._image_coords(scene_pos)
        if coord is None:
            return
        y, x = coord
        slice_ = self._extract_display_slice()
        value = float(slice_[y, x])
        msg = self._format_value(value, y, x)
        if self._segmentation is not None:
            organ = self._organ_at(y, x)
            if organ:
                msg += f"  [{organ}]"
        self.probed.emit(msg)

    def _organ_at(self, y: int, x: int) -> str | None:
        assert self._segmentation is not None and self._series is not None
        if self._segmentation.reference.shape != self._series.shape:
            return None
        seg = Series(
            pixels=self._segmentation.labels.astype(np.float32),
            spacing=self._series.spacing,
            origin=self._series.origin,
            orientation=self._series.orientation,
            modality=Modality.OTHER,
        )
        label_slice = extract_slice(seg, self._plane, self._state.slice_index)
        label = int(label_slice[y, x])
        if label == 0:
            return None
        structure = self._segmentation.by_id(label)
        return structure.name if structure else None

    def _format_value(self, value: float, y: int, x: int) -> str:
        assert self._series is not None
        parts = [f"{self._plane.value}  r={y} c={x}"]
        if self._series.modality == Modality.CT:
            parts.append(f"HU={value:.0f}")
        elif self._series.modality == Modality.PT:
            parts.append(f"Bq/ml={value:.1f}")
            if self._state.suv_factor:
                parts.append(f"SUV={value * self._state.suv_factor:.2f}")
        else:
            parts.append(f"val={value:.2f}")
        return "  ".join(parts)

    def _on_mouse_click(self, event: pg.Qt.QtGui.QMouseEvent) -> None:
        if self._series is None:
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        coord = self._image_coords(event.scenePos())
        if coord is None:
            return
        y, x = coord
        if self._tool == "crosshair":
            k, j, i = plane_pixel_to_voxel(
                self._series, self._plane, self._state.slice_index, float(y), float(x)
            )
            self.crosshairMoved.emit(k, j, i)
            return
        self._tool_points.append((float(y), float(x)))
        if self._tool == "distance" and len(self._tool_points) == 2:
            self._complete_distance()
            self._tool_points.clear()
        elif self._tool in ("rect", "ellipse") and len(self._tool_points) == 2:
            self._complete_roi()
            self._tool_points.clear()

    def _complete_distance(self) -> None:
        assert self._series is not None
        p0, p1 = self._tool_points
        _, sv, sh = plane_extent(self._series, self._plane)
        mm = distance_mm(p0, p1, sv, sh)
        self._clear_overlay_items()
        line = pg.PlotDataItem(
            x=[p0[1], p1[1]], y=[p0[0], p1[0]], pen=pg.mkPen("y", width=2)
        )
        self._plot.addItem(line)
        self._last_line_item = line
        self.measured.emit(f"Distance: {mm:.1f} mm")

    def _complete_roi(self) -> None:
        assert self._series is not None
        (y0, x0), (y1, x1) = self._tool_points
        slice_ = self._extract_display_slice()
        _, sv, sh = plane_extent(self._series, self._plane)
        self._clear_overlay_items()
        if self._tool == "rect":
            stats = rect_roi_stats(
                slice_, (int(y0), int(x0)), (int(y1), int(x1)), sv, sh
            )
            rect = QtCore.QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            roi = pg.RectROI(
                [rect.x(), rect.y()], [rect.width(), rect.height()],
                pen=pg.mkPen("c", width=2), movable=False,
            )
            roi.removeHandle(0)
        else:
            cy = (y0 + y1) / 2
            cx = (x0 + x1) / 2
            ry = abs(y1 - y0) / 2
            rx = abs(x1 - x0) / 2
            stats = ellipse_roi_stats(slice_, (cy, cx), (ry, rx), sv, sh)
            roi = pg.EllipseROI(
                [cx - rx, cy - ry], [rx * 2, ry * 2],
                pen=pg.mkPen("c", width=2), movable=False,
            )
            roi.removeHandle(0)
            roi.removeHandle(1)
        self._plot.addItem(roi)
        self._last_roi_item = roi
        self.measured.emit(self._format_roi_stats(stats))

    def _format_roi_stats(self, stats: ROIStats) -> str:
        assert self._series is not None
        unit = "HU" if self._series.modality == Modality.CT else (
            "Bq/ml" if self._series.modality == Modality.PT else "val"
        )
        msg = (
            f"ROI n={stats.count}  "
            f"mean={stats.mean:.2f} {unit}  "
            f"min={stats.minimum:.2f}  max={stats.maximum:.2f}  "
            f"std={stats.std:.2f}  area={stats.area_mm2:.1f} mm^2"
        )
        if (
            self._series.modality == Modality.PT
            and self._state.suv_factor is not None
        ):
            f = self._state.suv_factor
            msg += (
                f"  | SUV mean={stats.mean * f:.2f}  "
                f"max={stats.maximum * f:.2f}  min={stats.minimum * f:.2f}"
            )
        return msg

    def _clear_overlay_items(self) -> None:
        if self._last_roi_item is not None:
            self._plot.removeItem(self._last_roi_item)
            self._last_roi_item = None
        if self._last_line_item is not None:
            self._plot.removeItem(self._last_line_item)
            self._last_line_item = None
