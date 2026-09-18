#!/usr/bin/env python3
"""Small smoke tests for the HSV detection core."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hsv_detector import (
    BoundingBox,
    DetectionConfig,
    ObjectStabilityTracker,
    center_coordinate_offset,
    center_coordinate_offset_rate,
    detect_objects,
    require_opencv,
)


def create_self_test_frame():
    import cv2
    import numpy as np

    frame = np.full((480, 640, 3), (44, 48, 54), dtype=np.uint8)
    cv2.rectangle(frame, (220, 140), (420, 340), (0, 80, 255), -1)
    cv2.rectangle(frame, (260, 180), (380, 300), (20, 120, 255), -1)
    return frame


def main() -> int:
    require_opencv()
    frame = create_self_test_frame()
    result = detect_objects(frame, DetectionConfig(center_region_ratio=1.0))

    assert result.mask.shape[:2] == frame.shape[:2]
    assert result.all_rects, "default red-orange HSV should detect the self-test target"
    assert result.closest_rect is not None
    assert center_coordinate_offset(BoundingBox(360, 170, 20, 40), frame.shape) == (50, 50)
    assert center_coordinate_offset_rate((50, -50), frame.shape) == (0.16, -0.21)
    assert result.closest_offset_xy is not None
    assert result.closest_offset_rate_xy is not None

    tracker = ObjectStabilityTracker(
        window_size=5,
        stable_confirm_frames=2,
        unstable_confirm_frames=2,
        max_accel_ratio=0.1,
    )
    frame_shape = (100, 100, 3)
    smooth_boxes = [
        BoundingBox(10, 10, 20, 20),
        BoundingBox(12, 10, 20, 20),
        BoundingBox(14, 10, 20, 20),
        BoundingBox(16, 10, 20, 20),
        BoundingBox(18, 10, 20, 20),
        BoundingBox(20, 10, 20, 20),
    ]
    stable_values = [tracker.update(box, frame_shape) for box in smooth_boxes]
    assert stable_values[-1] is True
    assert tracker.payload() == {"if-obj-stable": 1}
    assert tracker.update(None, frame_shape) is True
    assert tracker.update(None, frame_shape) is False
    assert tracker.payload() == {"if-obj-stable": 0}

    jump_tracker = ObjectStabilityTracker(
        window_size=5,
        stable_confirm_frames=1,
        unstable_confirm_frames=1,
        max_accel_ratio=0.1,
    )
    jump_boxes = [
        BoundingBox(10, 10, 20, 20),
        BoundingBox(12, 10, 20, 20),
        BoundingBox(14, 10, 20, 20),
        BoundingBox(16, 10, 20, 20),
        BoundingBox(80, 10, 20, 20),
    ]
    for box in jump_boxes:
        jump_is_stable = jump_tracker.update(box, frame_shape)
    assert jump_is_stable is False

    print("detector core smoke test passed")
    print("all_rects=", result.all_rect_details)
    print("closest_rect=", result.closest_rect_details)
    print("closest_offset_xy=", result.closest_offset_xy)
    print("closest_offset_rate_xy=", result.closest_offset_rate_xy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
