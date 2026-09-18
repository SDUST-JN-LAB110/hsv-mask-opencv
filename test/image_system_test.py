#!/usr/bin/env python3
"""
Image-based system test for the HSV detector.

This script runs the same detection pipeline on one or more image files without
opening a GUI. It is intended for quick regression checks while tuning HSV,
filters, center-region rules, and area thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hsv_detector import (
    DEFAULT_LOWER_HSV,
    DEFAULT_UPPER_HSV,
    DetectionConfig,
    detect_objects,
    draw_detections,
    parse_hsv_triplet,
    require_opencv,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run image system tests for HSV detection.")
    parser.add_argument("images", nargs="+", type=Path, help="Image paths to test.")
    parser.add_argument("--hsv-low", type=parse_hsv_triplet, default=DEFAULT_LOWER_HSV)
    parser.add_argument("--hsv-high", type=parse_hsv_triplet, default=DEFAULT_UPPER_HSV)
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-area-ratio", type=float, default=1.0)
    parser.add_argument("--min-area-px", type=int, default=0)
    parser.add_argument("--max-area-px", type=int, default=10_000_000)
    parser.add_argument("--process-scale", type=float, default=0.75)
    parser.add_argument("--center-region-ratio", type=float, default=0.5)
    parser.add_argument("--min-all", type=int, default=0)
    parser.add_argument("--min-filtered", type=int, default=0)
    parser.add_argument("--min-center", type=int, default=0)
    parser.add_argument("--require-closest", action="store_true")
    parser.add_argument("--save-dir", type=Path, help="Save annotated images into this folder.")
    parser.add_argument("--display", choices=("original", "mask"), default="original")
    parser.add_argument("--boxes", choices=("all", "center", "closest"), default="all")
    return parser


def result_payload(image_path: Path, frame_shape, result) -> dict:
    return {
        "image": str(image_path),
        "width": int(frame_shape[1]),
        "height": int(frame_shape[0]),
        "all_rects": result.all_rect_details,
        "filtered_rects": result.filtered_rect_details,
        "center_rects": result.center_rect_details,
        "closest_rect": result.closest_rect_details,
        "closest_offset_xy": result.closest_offset_xy,
        "closest_offset_rate_xy": result.closest_offset_rate_xy,
        "has_hsv_target": result.has_hsv_target,
        "has_final_target": result.has_final_target,
    }


def assert_result(payload: dict, args: argparse.Namespace) -> list[str]:
    failures = []
    if len(payload["all_rects"]) < args.min_all:
        failures.append(f"all_rects 数量小于 {args.min_all}")
    if len(payload["filtered_rects"]) < args.min_filtered:
        failures.append(f"filtered_rects 数量小于 {args.min_filtered}")
    if len(payload["center_rects"]) < args.min_center:
        failures.append(f"center_rects 数量小于 {args.min_center}")
    if args.require_closest and payload["closest_rect"] is None:
        failures.append("closest_rect 为空")
    return failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    require_opencv()
    import cv2

    cv2.setUseOptimized(True)

    config = DetectionConfig(
        lower_hsv=args.hsv_low,
        upper_hsv=args.hsv_high,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_area_px=args.min_area_px,
        max_area_px=args.max_area_px,
        process_scale=args.process_scale,
        center_region_ratio=args.center_region_ratio,
    )

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for image_path in args.images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(json.dumps({"image": str(image_path), "error": "无法读取图片"}, ensure_ascii=False))
            exit_code = 1
            continue

        result = detect_objects(frame, config)
        payload = result_payload(image_path, frame.shape, result)
        failures = assert_result(payload, args)
        payload["passed"] = not failures
        payload["failures"] = failures
        print(json.dumps(payload, ensure_ascii=False))

        if failures:
            exit_code = 1

        if args.save_dir:
            annotated = draw_detections(
                frame,
                result,
                config,
                f"图片 {image_path.name}",
                display_mask=args.display == "mask",
                box_mode=args.boxes,
                show_center_region=True,
                show_reticle=True,
                picker_text="图片系统测试",
                display_scale=1.0,
                fps=0.0,
            )
            output_path = args.save_dir / f"{image_path.stem}_detected{image_path.suffix}"
            cv2.imwrite(str(output_path), annotated)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
