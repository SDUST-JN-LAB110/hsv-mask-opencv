#!/usr/bin/env python3
"""
Near-real-time HSV object detector.

Pipeline:
    BGR frame -> pluggable frame processors -> HSV mask
    -> pluggable mask processors -> contours -> bounding boxes
    -> pixel-area filter -> closest box to image center

The public function to reuse from other code is:
    result = detect_objects(frame, config)
    print(result.all_rect_details)
    print(result.filtered_rect_details)
    print(result.closest_rect_details)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    cv2 = None
    np = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    Image = None
    ImageDraw = None
    ImageFont = None


FrameProcessor = Callable[[Any, "DetectionConfig"], Any]
MaskProcessor = Callable[[Any, "DetectionConfig"], Any]

VIEW_WINDOW = "HSV检测器"
CONTROL_WINDOW = "HSV控制面板"
MAX_HUE = 179
MAX_CHANNEL = 255
DEFAULT_LOWER_HSV = (170, 90, 80)
DEFAULT_UPPER_HSV = (18, 255, 255)

TB_H_LOW = "H下限"
TB_S_LOW = "S下限"
TB_V_LOW = "V下限"
TB_H_HIGH = "H上限"
TB_S_HIGH = "S上限"
TB_V_HIGH = "V上限"
TB_MIN_AREA = "最小像素面积"
TB_MAX_AREA = "最大像素面积"
TB_FRAME_BLUR = "原图滤波核"
TB_MASK_MEDIAN = "掩膜中值核"
TB_MORPH = "形态学核"
TB_OPEN_ITER = "开运算次数"
TB_CLOSE_ITER = "闭运算次数"
TB_PROCESS_SCALE = "处理缩放%"
TB_CENTER_RATIO = "中心区域%"
TB_SHOW_CENTER = "显示中心框"
TB_SHOW_RETICLE = "显示准星"
TB_PICK_RADIUS = "取色半径px"
TB_H_PICK_TOL = "H取色容差"
TB_S_PICK_TOL = "S取色容差"
TB_V_PICK_TOL = "V取色容差"
TB_DISPLAY_SCALE = "显示缩放%"


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x2, self.y2)

    def area_ratio(self, frame_shape: Sequence[int]) -> float:
        frame_h, frame_w = frame_shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return 0.0
        return self.area / float(frame_w * frame_h)

    def scale(self, scale_x: float, scale_y: float) -> "BoundingBox":
        x = int(round(self.x * scale_x))
        y = int(round(self.y * scale_y))
        w = max(1, int(round(self.w * scale_x)))
        h = max(1, int(round(self.h * scale_y)))
        return BoundingBox(x, y, w, h)


@dataclass(frozen=True)
class DetectionConfig:
    lower_hsv: tuple[int, int, int] = DEFAULT_LOWER_HSV
    upper_hsv: tuple[int, int, int] = DEFAULT_UPPER_HSV
    min_area_ratio: float = 0.0
    max_area_ratio: float = 1.0
    min_area_px: int = 0
    max_area_px: int = 10_000_000
    frame_blur_kernel: int = 5
    mask_median_kernel: int = 5
    morph_kernel: int = 5
    open_iterations: int = 1
    close_iterations: int = 2
    process_scale: float = 1.0
    center_region_ratio: float = 1.0


@dataclass(frozen=True)
class DetectionResult:
    mask: Any
    all_rects: list[BoundingBox]
    filtered_rects: list[BoundingBox]
    center_rects: list[BoundingBox]
    closest_rect: Optional[BoundingBox]

    @property
    def all_rect_coords(self) -> list[tuple[int, int, int, int]]:
        return [box.xywh for box in self.all_rects]

    @property
    def all_rect_details(self) -> list[tuple[int, int, int, int, int]]:
        return [(*box.xywh, box.area) for box in self.all_rects]

    @property
    def filtered_rect_coords(self) -> list[tuple[int, int, int, int]]:
        return [box.xywh for box in self.filtered_rects]

    @property
    def filtered_rect_details(self) -> list[tuple[int, int, int, int, int]]:
        return [(*box.xywh, box.area) for box in self.filtered_rects]

    @property
    def center_rect_coords(self) -> list[tuple[int, int, int, int]]:
        return [box.xywh for box in self.center_rects]

    @property
    def center_rect_details(self) -> list[tuple[int, int, int, int, int]]:
        return [(*box.xywh, box.area) for box in self.center_rects]

    @property
    def closest_rect_coords(self) -> Optional[tuple[int, int, int, int]]:
        return None if self.closest_rect is None else self.closest_rect.xywh

    @property
    def closest_rect_details(self) -> Optional[tuple[int, int, int, int, int]]:
        return None if self.closest_rect is None else (*self.closest_rect.xywh, self.closest_rect.area)

    @property
    def has_hsv_target(self) -> bool:
        return bool(self.all_rects)

    @property
    def has_final_target(self) -> bool:
        return self.closest_rect is not None


def require_opencv() -> None:
    if _IMPORT_ERROR is not None:
        raise SystemExit(
            "OpenCV is not installed. Install dependencies first:\n"
            "  python3 -m pip install -r requirements.txt"
        ) from _IMPORT_ERROR


def odd_kernel(value: int) -> int:
    value = max(0, int(value))
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def clamp_process_scale(value: float) -> float:
    return min(max(float(value), 0.1), 1.0)


def clamp_center_region_ratio(value: float) -> float:
    return min(max(float(value), 0.01), 1.0)


def clamp_display_scale(value: float) -> float:
    return min(max(float(value), 0.25), 3.0)


def clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(int(value), upper))


def gaussian_blur_frame(frame: Any, config: DetectionConfig) -> Any:
    kernel = odd_kernel(config.frame_blur_kernel)
    if kernel <= 1:
        return frame
    return cv2.GaussianBlur(frame, (kernel, kernel), 0)


def median_blur_mask(mask: Any, config: DetectionConfig) -> Any:
    kernel = odd_kernel(config.mask_median_kernel)
    if kernel <= 1:
        return mask
    return cv2.medianBlur(mask, kernel)


def morph_open_mask(mask: Any, config: DetectionConfig) -> Any:
    if config.open_iterations <= 0 or config.morph_kernel <= 1:
        return mask
    kernel_size = odd_kernel(config.morph_kernel)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=config.open_iterations
    )


def morph_close_mask(mask: Any, config: DetectionConfig) -> Any:
    if config.close_iterations <= 0 or config.morph_kernel <= 1:
        return mask
    kernel_size = odd_kernel(config.morph_kernel)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=config.close_iterations
    )


# Add, remove, or reorder functions here to change the processing pipeline.
FRAME_PROCESSORS: list[FrameProcessor] = [
    gaussian_blur_frame,
]

MASK_PROCESSORS: list[MaskProcessor] = [
    median_blur_mask,
    morph_open_mask,
    morph_close_mask,
]


def apply_frame_processors(frame: Any, config: DetectionConfig) -> Any:
    processed = frame
    for processor in FRAME_PROCESSORS:
        processed = processor(processed, config)
    return processed


def apply_mask_processors(mask: Any, config: DetectionConfig) -> Any:
    processed = mask
    for processor in MASK_PROCESSORS:
        processed = processor(processed, config)
    return processed


def hsv_bound(h: int, s: int, v: int) -> Any:
    return np.array((h, s, v), dtype=np.uint8)


def create_hsv_mask(frame: Any, config: DetectionConfig) -> Any:
    processed_frame = apply_frame_processors(frame, config)
    hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
    lower = hsv_bound(*config.lower_hsv)
    upper = hsv_bound(*config.upper_hsv)

    if lower[0] <= upper[0]:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        # Hue wraps at 179 in OpenCV HSV, so red-like ranges may cross zero.
        low_to_end = cv2.inRange(
            hsv,
            lower,
            hsv_bound(MAX_HUE, int(upper[1]), int(upper[2])),
        )
        start_to_high = cv2.inRange(
            hsv,
            hsv_bound(0, int(lower[1]), int(lower[2])),
            upper,
        )
        mask = cv2.bitwise_or(low_to_end, start_to_high)

    return apply_mask_processors(mask, config)


def find_bounding_boxes(mask: Any) -> list[BoundingBox]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [BoundingBox(*cv2.boundingRect(contour)) for contour in contours]
    return sorted(boxes, key=lambda box: (box.y, box.x))


def filter_boxes_by_area(
    boxes: Iterable[BoundingBox], frame_shape: Sequence[int], config: DetectionConfig
) -> list[BoundingBox]:
    min_ratio = min(max(config.min_area_ratio, 0.0), 1.0)
    max_ratio = min(max(config.max_area_ratio, min_ratio), 1.0)
    min_area_px = max(0, int(config.min_area_px))
    max_area_px = max(min_area_px, int(config.max_area_px))
    return [
        box
        for box in boxes
        if min_ratio <= box.area_ratio(frame_shape) <= max_ratio
        and min_area_px <= box.area <= max_area_px
    ]


def center_region_bounds(
    frame_shape: Sequence[int], center_region_ratio: float
) -> BoundingBox:
    frame_h, frame_w = frame_shape[:2]
    ratio = clamp_center_region_ratio(center_region_ratio)
    region_w = max(1, int(round(frame_w * ratio)))
    region_h = max(1, int(round(frame_h * ratio)))
    x = (frame_w - region_w) // 2
    y = (frame_h - region_h) // 2
    return BoundingBox(x, y, region_w, region_h)


def filter_boxes_by_center_region(
    boxes: Iterable[BoundingBox], frame_shape: Sequence[int], config: DetectionConfig
) -> list[BoundingBox]:
    region = center_region_bounds(frame_shape, config.center_region_ratio)
    return [
        box
        for box in boxes
        if region.x <= box.center[0] <= region.x2
        and region.y <= box.center[1] <= region.y2
    ]


def choose_closest_to_center(
    boxes: Sequence[BoundingBox], frame_shape: Sequence[int]
) -> Optional[BoundingBox]:
    if not boxes:
        return None
    frame_h, frame_w = frame_shape[:2]
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    return min(
        boxes,
        key=lambda box: (box.center[0] - cx) ** 2 + (box.center[1] - cy) ** 2,
    )


def detect_objects(frame: Any, config: DetectionConfig) -> DetectionResult:
    """Return all boxes, size-filtered boxes, and the center-nearest box."""
    require_opencv()
    original_h, original_w = frame.shape[:2]
    process_scale = clamp_process_scale(config.process_scale)

    if process_scale < 0.999:
        process_w = max(1, int(round(original_w * process_scale)))
        process_h = max(1, int(round(original_h * process_scale)))
        process_frame = cv2.resize(frame, (process_w, process_h), interpolation=cv2.INTER_AREA)
    else:
        process_frame = frame

    process_mask = create_hsv_mask(process_frame, config)
    process_rects = find_bounding_boxes(process_mask)

    if process_frame is frame:
        mask = process_mask
        all_rects = process_rects
    else:
        scale_x = original_w / float(process_frame.shape[1])
        scale_y = original_h / float(process_frame.shape[0])
        all_rects = [box.scale(scale_x, scale_y) for box in process_rects]
        mask = cv2.resize(process_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    filtered_rects = filter_boxes_by_area(all_rects, frame.shape, config)
    center_rects = filter_boxes_by_center_region(filtered_rects, frame.shape, config)
    closest_rect = choose_closest_to_center(center_rects, frame.shape)
    return DetectionResult(mask, all_rects, filtered_rects, center_rects, closest_rect)


def parse_hsv_triplet(raw: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HSV must look like 0,80,80") from exc

    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV must contain exactly 3 numbers")
    h, s, v = parts
    if not (0 <= h <= 179 and 0 <= s <= 255 and 0 <= v <= 255):
        raise argparse.ArgumentTypeError("HSV ranges are H=0..179, S=0..255, V=0..255")
    return (h, s, v)


class HsvPickerState:
    def __init__(self) -> None:
        self.frame: Optional[Any] = None
        self.display_scale = 1.0
        self.last_hsv: Optional[tuple[int, int, int]] = None
        self.last_xy: Optional[tuple[int, int]] = None
        self.lock = threading.Lock()

    def set_frame(self, frame: Any, display_scale: float) -> None:
        with self.lock:
            self.frame = frame.copy()
            self.display_scale = clamp_display_scale(display_scale)

    def on_mouse(
        self, event: int, x: int, y: int, flags: int, userdata: Optional[Any]
    ) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        with self.lock:
            if self.frame is None:
                return
            frame = self.frame.copy()
            display_scale = self.display_scale

        frame_h, frame_w = frame.shape[:2]
        x = int(round(x / display_scale))
        y = int(round(y / display_scale))
        if not (0 <= x < frame_w and 0 <= y < frame_h):
            return

        radius = max(0, cv2.getTrackbarPos(TB_PICK_RADIUS, CONTROL_WINDOW))
        x1 = clamp_int(x - radius, 0, frame_w - 1)
        x2 = clamp_int(x + radius + 1, 1, frame_w)
        y1 = clamp_int(y - radius, 0, frame_h - 1)
        y2 = clamp_int(y + radius + 1, 1, frame_h)

        patch = frame[y1:y2, x1:x2]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h, s, v = np.median(hsv_patch.reshape(-1, 3), axis=0).astype(int)
        hsv = (int(h), int(s), int(v))
        set_hsv_trackbars_from_pick(hsv)

        with self.lock:
            self.last_hsv = hsv
            self.last_xy = (x, y)

    def sample_text(self) -> str:
        with self.lock:
            if self.last_hsv is None or self.last_xy is None:
                return "取色=None | 鼠标左键点击目标颜色自动调整HSV"
            return f"取色坐标={self.last_xy} 取色HSV={self.last_hsv}"


def set_hsv_trackbars_from_pick(hsv: tuple[int, int, int]) -> None:
    h, s, v = hsv
    h_tol = cv2.getTrackbarPos(TB_H_PICK_TOL, CONTROL_WINDOW)
    s_tol = cv2.getTrackbarPos(TB_S_PICK_TOL, CONTROL_WINDOW)
    v_tol = cv2.getTrackbarPos(TB_V_PICK_TOL, CONTROL_WINDOW)

    if h_tol >= 90:
        low_h = 0
        high_h = MAX_HUE
    else:
        low_h = (h - h_tol) % (MAX_HUE + 1)
        high_h = (h + h_tol) % (MAX_HUE + 1)

    cv2.setTrackbarPos(TB_H_LOW, CONTROL_WINDOW, low_h)
    cv2.setTrackbarPos(TB_S_LOW, CONTROL_WINDOW, clamp_int(s - s_tol, 0, MAX_CHANNEL))
    cv2.setTrackbarPos(TB_V_LOW, CONTROL_WINDOW, clamp_int(v - v_tol, 0, MAX_CHANNEL))
    cv2.setTrackbarPos(TB_H_HIGH, CONTROL_WINDOW, high_h)
    cv2.setTrackbarPos(TB_S_HIGH, CONTROL_WINDOW, clamp_int(s + s_tol, 0, MAX_CHANNEL))
    cv2.setTrackbarPos(TB_V_HIGH, CONTROL_WINDOW, clamp_int(v + v_tol, 0, MAX_CHANNEL))


def create_controls(initial: DetectionConfig, initial_display_scale: float) -> None:
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 520, 420)

    values = {
        TB_H_LOW: (initial.lower_hsv[0], 179),
        TB_S_LOW: (initial.lower_hsv[1], 255),
        TB_V_LOW: (initial.lower_hsv[2], 255),
        TB_H_HIGH: (initial.upper_hsv[0], 179),
        TB_S_HIGH: (initial.upper_hsv[1], 255),
        TB_V_HIGH: (initial.upper_hsv[2], 255),
        TB_MIN_AREA: (int(initial.min_area_px), 10_000_000),
        TB_MAX_AREA: (int(initial.max_area_px), 10_000_000),
        TB_FRAME_BLUR: (initial.frame_blur_kernel, 31),
        TB_MASK_MEDIAN: (initial.mask_median_kernel, 31),
        TB_MORPH: (initial.morph_kernel, 31),
        TB_OPEN_ITER: (initial.open_iterations, 5),
        TB_CLOSE_ITER: (initial.close_iterations, 5),
        TB_PROCESS_SCALE: (int(clamp_process_scale(initial.process_scale) * 100), 100),
        TB_CENTER_RATIO: (
            int(clamp_center_region_ratio(initial.center_region_ratio) * 100),
            100,
        ),
        TB_SHOW_CENTER: (1, 1),
        TB_SHOW_RETICLE: (1, 1),
        TB_PICK_RADIUS: (3, 50),
        TB_H_PICK_TOL: (10, 90),
        TB_S_PICK_TOL: (60, 255),
        TB_V_PICK_TOL: (60, 255),
        TB_DISPLAY_SCALE: (int(clamp_display_scale(initial_display_scale) * 100), 300),
    }

    for name, (value, maximum) in values.items():
        cv2.createTrackbar(name, CONTROL_WINDOW, value, maximum, lambda _value: None)


def read_controls() -> DetectionConfig:
    lower = (
        cv2.getTrackbarPos(TB_H_LOW, CONTROL_WINDOW),
        cv2.getTrackbarPos(TB_S_LOW, CONTROL_WINDOW),
        cv2.getTrackbarPos(TB_V_LOW, CONTROL_WINDOW),
    )
    upper = (
        cv2.getTrackbarPos(TB_H_HIGH, CONTROL_WINDOW),
        cv2.getTrackbarPos(TB_S_HIGH, CONTROL_WINDOW),
        cv2.getTrackbarPos(TB_V_HIGH, CONTROL_WINDOW),
    )
    min_area_px = cv2.getTrackbarPos(TB_MIN_AREA, CONTROL_WINDOW)
    max_area_px = cv2.getTrackbarPos(TB_MAX_AREA, CONTROL_WINDOW)
    if max_area_px <= 0:
        max_area_px = 10_000_000
    if min_area_px > max_area_px:
        min_area_px = max_area_px

    return DetectionConfig(
        lower_hsv=lower,
        upper_hsv=upper,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
        frame_blur_kernel=cv2.getTrackbarPos(TB_FRAME_BLUR, CONTROL_WINDOW),
        mask_median_kernel=cv2.getTrackbarPos(TB_MASK_MEDIAN, CONTROL_WINDOW),
        morph_kernel=cv2.getTrackbarPos(TB_MORPH, CONTROL_WINDOW),
        open_iterations=cv2.getTrackbarPos(TB_OPEN_ITER, CONTROL_WINDOW),
        close_iterations=cv2.getTrackbarPos(TB_CLOSE_ITER, CONTROL_WINDOW),
        process_scale=clamp_process_scale(
            cv2.getTrackbarPos(TB_PROCESS_SCALE, CONTROL_WINDOW) / 100.0
        ),
        center_region_ratio=clamp_center_region_ratio(
            cv2.getTrackbarPos(TB_CENTER_RATIO, CONTROL_WINDOW) / 100.0
        ),
    )


def read_show_center_box() -> bool:
    return cv2.getTrackbarPos(TB_SHOW_CENTER, CONTROL_WINDOW) == 1


def read_show_reticle() -> bool:
    return cv2.getTrackbarPos(TB_SHOW_RETICLE, CONTROL_WINDOW) == 1


def read_display_scale() -> float:
    return clamp_display_scale(cv2.getTrackbarPos(TB_DISPLAY_SCALE, CONTROL_WINDOW) / 100.0)


def adjust_display_scale(delta: float) -> None:
    next_scale = read_display_scale() + delta
    cv2.setTrackbarPos(TB_DISPLAY_SCALE, CONTROL_WINDOW, int(clamp_display_scale(next_scale) * 100))


class CameraSource:
    def __init__(
        self,
        camera_index: int,
        max_camera_index: int,
        width: Optional[int],
        height: Optional[int],
    ) -> None:
        self.max_camera_index = max(0, max_camera_index)
        self.width = width
        self.height = height
        self.index = camera_index
        self.cap: Optional[Any] = None
        self.frame: Optional[Any] = None
        self.frame_id = 0
        self.device_width = 0
        self.device_height = 0
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        if not self.open(camera_index):
            raise RuntimeError(f"Could not open camera {camera_index}")

    @property
    def label(self) -> str:
        if self.device_width > 0 and self.device_height > 0:
            return f"摄像头 {self.index} ({self.device_width}x{self.device_height})"
        return f"摄像头 {self.index}"

    def open(self, index: int) -> bool:
        index = max(0, min(index, self.max_camera_index))
        self.release()

        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            return False

        self.cap = cap
        self.index = index
        self.device_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.device_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            ok, _frame = self.read()
            if ok:
                return True
            time.sleep(0.01)
        return True

    def _capture_loop(self) -> None:
        cap = self.cap
        while self.running and cap is not None:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.device_width <= 0 or self.device_height <= 0:
                self.device_height, self.device_width = frame.shape[:2]
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def next_camera(self, step: int = 1) -> bool:
        previous_index = self.index
        for offset in range(1, self.max_camera_index + 2):
            candidate = (self.index + step * offset) % (self.max_camera_index + 1)
            if self.open(candidate):
                return True
        self.open(previous_index)
        return False

    def read(self) -> tuple[bool, Any]:
        with self.lock:
            if self.frame is None:
                return False, None
            if self.device_width <= 0 or self.device_height <= 0:
                self.device_height, self.device_width = self.frame.shape[:2]
            return True, self.frame.copy()

    def release(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.5)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self.lock:
            self.frame = None


class SyncCameraSource(CameraSource):
    def open(self, index: int) -> bool:
        index = max(0, min(index, self.max_camera_index))
        self.release()

        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            return False

        self.cap = cap
        self.index = index
        self.device_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.device_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return True

    def read(self) -> tuple[bool, Any]:
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if ok and frame is not None and (self.device_width <= 0 or self.device_height <= 0):
            self.device_height, self.device_width = frame.shape[:2]
        return ok, frame


class ImageSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.frame = cv2.imread(str(path))
        if self.frame is None:
            raise RuntimeError(f"Could not read image: {path}")

    @property
    def label(self) -> str:
        return f"图片 {self.path.name}"

    def read(self) -> tuple[bool, Any]:
        return True, self.frame.copy()

    def next_camera(self, step: int = 1) -> bool:
        return False

    def release(self) -> None:
        return None


_FONT_CACHE: dict[int, Any] = {}


def get_chinese_font(size: int) -> Any:
    if ImageFont is None:
        return None
    size = max(12, int(size))
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            _FONT_CACHE[size] = ImageFont.truetype(font_path, size)
            return _FONT_CACHE[size]

    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_text_lines(image: Any, lines: Sequence[str], display_scale: float) -> None:
    scale = clamp_display_scale(display_scale)
    if Image is not None and ImageDraw is not None and ImageFont is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        font = get_chinese_font(int(round(18 * scale)))
        x = max(6, int(round(12 * scale)))
        y = max(6, int(round(12 * scale)))
        line_height = max(16, int(round(24 * scale)))

        for line in lines:
            shadow_offset = max(1, int(round(1 * scale)))
            draw.text(
                (x + shadow_offset, y + shadow_offset),
                line,
                font=font,
                fill=(20, 20, 20),
            )
            draw.text((x, y), line, font=font, fill=(255, 255, 255))
            y += line_height

        image[:, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return

    y = max(16, int(round(24 * scale)))
    x = max(6, int(round(12 * scale)))
    font_scale = max(0.35, 0.56 * scale)
    line_height = max(14, int(round(22 * scale)))
    for line in lines:
        safe_line = line.encode("ascii", "ignore").decode("ascii") or "Pillow missing"
        cv2.putText(
            image,
            safe_line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            max(1, int(round(3 * scale))),
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            safe_line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_height


def draw_center_reticle(image: Any, center: tuple[int, int]) -> None:
    cx, cy = center
    color = (255, 120, 0)
    shadow = (20, 20, 20)
    gap = 8
    length = 36
    circle_radius = 18

    segments = [
        ((cx - length, cy), (cx - gap, cy)),
        ((cx + gap, cy), (cx + length, cy)),
        ((cx, cy - length), (cx, cy - gap)),
        ((cx, cy + gap), (cx, cy + length)),
    ]

    for start, end in segments:
        cv2.line(image, start, end, shadow, 5, cv2.LINE_AA)
        cv2.line(image, start, end, color, 2, cv2.LINE_AA)

    cv2.circle(image, center, circle_radius, shadow, 4, cv2.LINE_AA)
    cv2.circle(image, center, circle_radius, color, 1, cv2.LINE_AA)
    cv2.circle(image, center, 2, (0, 240, 255), -1, cv2.LINE_AA)


def resize_for_display(image: Any, display_scale: float) -> Any:
    scale = clamp_display_scale(display_scale)
    if abs(scale - 1.0) < 0.001:
        return image
    height, width = image.shape[:2]
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def draw_detections(
    frame: Any,
    result: DetectionResult,
    config: DetectionConfig,
    source_label: str,
    display_mask: bool,
    box_mode: str,
    show_center_region: bool,
    show_reticle: bool,
    picker_text: str,
    display_scale: float,
    fps: float,
) -> Any:
    if display_mask:
        display = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    else:
        display = frame.copy()

    frame_h, frame_w = frame.shape[:2]
    center = (frame_w // 2, frame_h // 2)
    center_region = center_region_bounds(frame.shape, config.center_region_ratio)
    if show_center_region:
        cv2.rectangle(
            display,
            (center_region.x, center_region.y),
            (center_region.x2, center_region.y2),
            (255, 160, 0),
            2,
        )
    if show_reticle:
        draw_center_reticle(display, center)

    if box_mode == "all":
        boxes_to_draw = result.all_rects
        box_color = (90, 220, 90)
    elif box_mode == "center":
        boxes_to_draw = result.center_rects
        box_color = (0, 230, 255)
    else:
        boxes_to_draw = []
        box_color = (90, 220, 90)

    for box in boxes_to_draw:
        cv2.rectangle(display, (box.x, box.y), (box.x2, box.y2), box_color, 2)
        cv2.putText(
            display,
            f"{box.xywh} A={box.area}",
            (box.x, max(18, box.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
            cv2.LINE_AA,
        )

    if result.closest_rect is not None:
        box = result.closest_rect
        cv2.rectangle(display, (box.x, box.y), (box.x2, box.y2), (0, 240, 255), 3)
        cv2.circle(display, (int(box.center[0]), int(box.center[1])), 4, (0, 240, 255), -1)
        cv2.putText(
            display,
            f"{box.xywh} A={box.area}",
            (box.x, min(frame.shape[0] - 8, box.y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 240, 255),
            1,
            cv2.LINE_AA,
        )

    display = resize_for_display(display, display_scale)
    view_label = "HSV掩膜" if display_mask else "原画面"
    box_label = {"all": "全部框", "center": "中心候选框", "closest": "最终最近框"}[box_mode]
    draw_text_lines(
        display,
        [
            f"输入：{source_label} | FPS：{fps:.1f} | 显示：{view_label} | 矩形框：{box_label}",
            f"HSV下限={config.lower_hsv} HSV上限={config.upper_hsv}",
            f"像素面积：{config.min_area_px}..{config.max_area_px}",
            f"处理缩放：{clamp_process_scale(config.process_scale):.2f} | 显示缩放：{clamp_display_scale(display_scale):.2f}",
            f"中心区域比例：{clamp_center_region_ratio(config.center_region_ratio):.2f} 中心框={center_region.xywh}",
            f"全部矩形框={result.all_rect_details}",
            f"面积过滤后={result.filtered_rect_details}",
            f"中心候选框={result.center_rect_details}",
            f"最终最近框={result.closest_rect_details if result.has_final_target else '无'}",
            picker_text,
            "按键：q退出 | m切换画面/掩膜 | b切换框 | +/-缩放显示 | c/p切换摄像头 | 0-9选摄像头",
        ],
        display_scale,
    )
    return display


def print_detection_result(result: DetectionResult) -> None:
    print(
        "all_rects=",
        result.all_rect_details,
        "filtered_rects=",
        result.filtered_rect_details,
        "center_rects=",
        result.center_rect_details,
        "closest_rect=",
        result.closest_rect_details,
        "has_hsv_target=",
        result.has_hsv_target,
        "has_final_target=",
        result.has_final_target,
        flush=True,
    )


def run_headless(args: argparse.Namespace, config: DetectionConfig) -> int:
    source = ImageSource(Path(args.image))
    ok, frame = source.read()
    if not ok:
        raise RuntimeError("Could not load image")

    result = detect_objects(frame, config)
    print_detection_result(result)

    if args.save_output:
        output = draw_detections(
            frame,
            result,
            config,
            source.label,
            display_mask=args.display == "mask",
            box_mode=args.boxes,
            show_center_region=True,
            show_reticle=True,
            picker_text="取色=None",
            display_scale=1.0,
            fps=0.0,
        )
        cv2.imwrite(str(args.save_output), output)
        print(f"saved_output={args.save_output}")

    return 0


def run_gui(args: argparse.Namespace, config: DetectionConfig) -> int:
    if args.image:
        source: Any = ImageSource(Path(args.image))
    else:
        camera_cls = SyncCameraSource if args.sync_camera else CameraSource
        source = camera_cls(
            args.camera, args.max_camera_index, width=args.width, height=args.height
        )

    cv2.namedWindow(VIEW_WINDOW, cv2.WINDOW_NORMAL)
    create_controls(config, args.display_scale)
    hsv_picker = HsvPickerState()
    cv2.setMouseCallback(VIEW_WINDOW, hsv_picker.on_mouse)

    display_mask = args.display == "mask"
    box_mode = args.boxes
    last_frame_time = time.monotonic()
    last_print_time = 0.0
    last_display_size: Optional[tuple[int, int]] = None
    fps = 0.0

    try:
        while True:
            ok, frame = source.read()
            if not ok:
                print(f"Could not read from {source.label}", file=sys.stderr)
                time.sleep(0.1)
                continue

            now = time.monotonic()
            elapsed = max(now - last_frame_time, 1e-6)
            last_frame_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed

            current_config = read_controls()
            display_scale = read_display_scale()
            hsv_picker.set_frame(frame, display_scale)
            show_center_region = read_show_center_box()
            show_reticle = read_show_reticle()
            result = detect_objects(frame, current_config)
            display = draw_detections(
                frame,
                result,
                current_config,
                source.label,
                display_mask,
                box_mode,
                show_center_region,
                show_reticle,
                hsv_picker.sample_text(),
                display_scale,
                fps,
            )
            display_size = (display.shape[1], display.shape[0])
            if display_size != last_display_size:
                cv2.resizeWindow(VIEW_WINDOW, display_size[0], display_size[1])
                last_display_size = display_size
            cv2.imshow(VIEW_WINDOW, display)
            if args.print_interval > 0 and now - last_print_time >= args.print_interval:
                print_detection_result(result)
                last_print_time = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                display_mask = not display_mask
            elif key == ord("b"):
                box_modes = ("all", "center", "closest")
                box_mode = box_modes[(box_modes.index(box_mode) + 1) % len(box_modes)]
            elif key in (ord("+"), ord("=")):
                adjust_display_scale(0.1)
            elif key in (ord("-"), ord("_")):
                adjust_display_scale(-0.1)
            elif key == ord("c"):
                if not source.next_camera(step=1):
                    print("No next camera available", file=sys.stderr)
            elif key == ord("p"):
                if not source.next_camera(step=-1):
                    print("No previous camera available", file=sys.stderr)
            elif ord("0") <= key <= ord("9") and not args.image:
                camera_index = key - ord("0")
                previous_index = source.index
                if not source.open(camera_index):
                    source.open(previous_index)
                    print(f"Could not open camera {camera_index}", file=sys.stderr)
    finally:
        source.release()
        cv2.destroyAllWindows()

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Near-real-time HSV mask detector with center-nearest bounding box."
    )
    parser.add_argument("--image", type=Path, help="Run detection on one image instead of a camera.")
    parser.add_argument("--camera", type=int, default=0, help="Initial camera index.")
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=4,
        help="Highest camera index to try when switching cameras.",
    )
    parser.add_argument("--width", type=int, help="Kept for compatibility; camera resolution is device-defined.")
    parser.add_argument("--height", type=int, help="Kept for compatibility; camera resolution is device-defined.")
    parser.add_argument("--hsv-low", type=parse_hsv_triplet, default=DEFAULT_LOWER_HSV)
    parser.add_argument("--hsv-high", type=parse_hsv_triplet, default=DEFAULT_UPPER_HSV)
    parser.add_argument("--min-area-ratio", type=float, default=0.0, help="Compatibility ratio filter; pixel area filter is preferred.")
    parser.add_argument("--max-area-ratio", type=float, default=1.0, help="Compatibility ratio filter; pixel area filter is preferred.")
    parser.add_argument("--min-area-px", type=int, default=0, help="Minimum bounding-box pixel area.")
    parser.add_argument("--max-area-px", type=int, default=10_000_000, help="Maximum bounding-box pixel area.")
    parser.add_argument(
        "--center-region-ratio",
        type=float,
        default=1.0,
        help="Centered region width/height ratio used before choosing closest box.",
    )
    parser.add_argument(
        "--process-scale",
        type=float,
        default=1.0,
        help="Process a smaller frame for speed, then map boxes back. Range: 0.1..1.0.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Scale the displayed GUI image. Range: 0.25..3.0.",
    )
    parser.add_argument("--display", choices=("original", "mask"), default="original")
    parser.add_argument("--boxes", choices=("all", "center", "closest"), default="all")
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="Seconds between console coordinate prints in GUI mode. Use 0 to disable.",
    )
    parser.add_argument(
        "--sync-camera",
        action="store_true",
        help="Disable threaded latest-frame capture and read camera synchronously.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Print one image detection result without opening GUI. Requires --image.",
    )
    parser.add_argument(
        "--save-output",
        type=Path,
        help="Save an annotated output image. Useful with --image --headless.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    require_opencv()
    cv2.setUseOptimized(True)

    config = DetectionConfig(
        lower_hsv=args.hsv_low,
        upper_hsv=args.hsv_high,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_area_px=args.min_area_px,
        max_area_px=args.max_area_px,
        process_scale=clamp_process_scale(args.process_scale),
        center_region_ratio=clamp_center_region_ratio(args.center_region_ratio),
    )

    if args.headless:
        if not args.image:
            parser.error("--headless requires --image")
        return run_headless(args, config)

    return run_gui(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
