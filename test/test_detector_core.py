#!/usr/bin/env python3
"""Small smoke tests for the HSV detection core."""

from __future__ import annotations

from hsv_detector import DetectionConfig, detect_objects, require_opencv


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

    print("detector core smoke test passed")
    print("all_rects=", result.all_rect_details)
    print("closest_rect=", result.closest_rect_details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
