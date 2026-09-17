#!/usr/bin/env python3
"""
Lightweight Tkinter GUI for HSV detection.

This is the recommended GUI for algorithm testing:
    - tkinter/ttk is included with most Python installations.
    - GUI runs on the Tk main thread.
    - camera/image reading and OpenCV detection run on a worker thread.
    - the frame queue keeps only the latest processed frame to avoid lag.
"""

from __future__ import annotations

import argparse
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional, Sequence

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    cv2 = None
    np = None
    _CV_IMPORT_ERROR = exc
else:
    _CV_IMPORT_ERROR = None

try:
    from PIL import Image, ImageTk
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    Image = None
    ImageTk = None
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None

from hsv_detector import (
    DEFAULT_LOWER_HSV,
    DEFAULT_UPPER_HSV,
    DetectionConfig,
    center_region_bounds,
    clamp_center_region_ratio,
    clamp_int,
    clamp_process_scale,
    detect_objects,
    draw_center_reticle,
    parse_hsv_triplet,
)


APP_TITLE = "HSV 轻量测试工具"
MAX_CAMERA_SCAN_INDEX = 9


def pil_bilinear_resample():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


@dataclass(frozen=True)
class AppState:
    source_mode: str = "camera"
    camera_index: int = 0
    image_path: Optional[Path] = None
    width: int = 640
    height: int = 480
    config: DetectionConfig = DetectionConfig()
    view_mode: str = "original"
    box_mode: str = "all"
    show_center_region: bool = True
    show_reticle: bool = True
    display_scale: float = 1.0
    pick_radius: int = 0
    h_pick_tol: int = 10
    s_pick_tol: int = 60
    v_pick_tol: int = 60
    print_interval: float = 0.5


@dataclass(frozen=True)
class CameraDevice:
    index: int
    label: str


def read_macos_camera_names() -> list[str]:
    if platform.system() != "Darwin":
        return []
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=2.5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    names: list[str] = []
    for line in result.stdout.splitlines():
        match = re.match(r"^\s{4}(.+):\s*$", line)
        if match:
            name = match.group(1).strip()
            if name and name not in names:
                names.append(name)
    return names


def discover_camera_devices(max_index: int = MAX_CAMERA_SCAN_INDEX) -> list[CameraDevice]:
    camera_names = read_macos_camera_names()
    devices: list[CameraDevice] = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                continue
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if (width <= 0 or height <= 0) and cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    height, width = frame.shape[:2]
            name = camera_names[len(devices)] if len(devices) < len(camera_names) else f"摄像头 {index}"
            label = f"{name}（设备 {index}）"
            if width > 0 and height > 0:
                label += f" {width}x{height}"
            devices.append(CameraDevice(index=index, label=label))
        finally:
            cap.release()
    return devices


def create_self_test_frame(width: int = 640, height: int = 480):
    frame = np.full((height, width, 3), (44, 48, 54), dtype=np.uint8)
    cv2.rectangle(frame, (220, 140), (420, 340), (0, 80, 255), -1)
    cv2.rectangle(frame, (260, 180), (380, 300), (20, 120, 255), -1)
    cv2.putText(
        frame,
        "HSV TEST",
        (230, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return frame


class LatestQueue:
    def __init__(self) -> None:
        self.queue: queue.Queue[dict] = queue.Queue(maxsize=1)

    def put(self, item: dict) -> None:
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(item)

    def get_nowait(self) -> Optional[dict]:
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None


class VideoWorker(threading.Thread):
    def __init__(self, state: AppState, frame_queue: LatestQueue, event_queue: LatestQueue) -> None:
        super().__init__(daemon=True)
        self._state = state
        self._state_lock = threading.Lock()
        self._frame_queue = frame_queue
        self._event_queue = event_queue
        self._latest_raw_frame = None
        self._latest_raw_frame_lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()

    def stop(self) -> None:
        self._running.clear()

    def snapshot(self) -> AppState:
        with self._state_lock:
            return self._state

    def update_state(self, **kwargs) -> None:
        with self._state_lock:
            self._state = replace(self._state, **kwargs)

    def pick_hsv(self, x: int, y: int) -> None:
        state = self.snapshot()
        with self._latest_raw_frame_lock:
            if self._latest_raw_frame is None:
                return
            frame = self._latest_raw_frame.copy()

        frame_h, frame_w = frame.shape[:2]
        if not (0 <= x < frame_w and 0 <= y < frame_h):
            return

        radius = max(0, state.pick_radius)
        x1 = clamp_int(x - radius, 0, frame_w - 1)
        x2 = clamp_int(x + radius + 1, 1, frame_w)
        y1 = clamp_int(y - radius, 0, frame_h - 1)
        y2 = clamp_int(y + radius + 1, 1, frame_h)

        patch = frame[y1:y2, x1:x2]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h, s, v = np.median(hsv_patch.reshape(-1, 3), axis=0).astype(int)

        if state.h_pick_tol >= 90:
            low_h, high_h = 0, 179
        else:
            low_h = (int(h) - state.h_pick_tol) % 180
            high_h = (int(h) + state.h_pick_tol) % 180

        lower_hsv = (
            low_h,
            clamp_int(int(s) - state.s_pick_tol, 0, 255),
            clamp_int(int(v) - state.v_pick_tol, 0, 255),
        )
        upper_hsv = (
            high_h,
            clamp_int(int(s) + state.s_pick_tol, 0, 255),
            clamp_int(int(v) + state.v_pick_tol, 0, 255),
        )
        self.update_state(config=replace(state.config, lower_hsv=lower_hsv, upper_hsv=upper_hsv))
        self._event_queue.put(
            {
                "type": "picked",
                "xy": (x, y),
                "hsv": (int(h), int(s), int(v)),
                "lower_hsv": lower_hsv,
                "upper_hsv": upper_hsv,
            }
        )

    def run(self) -> None:
        try:
            self._run_loop()
        except Exception as exc:  # pragma: no cover - defensive thread boundary
            self._event_queue.put(
                {
                    "type": "error",
                    "message": f"后台视频线程异常：{exc}",
                    "detail": traceback.format_exc(),
                }
            )

    def _run_loop(self) -> None:
        cap = None
        image_frame = None
        opened_signature = None
        last_print_time = 0.0
        last_frame_time = time.monotonic()
        fps = 0.0

        while self._running.is_set():
            state = self.snapshot()
            signature = self._source_signature(state)

            if signature != opened_signature:
                if cap is not None:
                    cap.release()
                    cap = None
                image_frame = None
                opened_signature = signature

                if state.source_mode == "camera":
                    cap = cv2.VideoCapture(state.camera_index)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not cap.isOpened():
                        self._event_queue.put({"type": "error", "message": f"无法打开摄像头 {state.camera_index}"})
                        time.sleep(0.3)
                        continue
                    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    if cam_w > 0 and cam_h > 0:
                        message = f"摄像头 {state.camera_index} 已打开，设备分辨率 {cam_w}x{cam_h}"
                    else:
                        message = f"摄像头 {state.camera_index} 已打开，等待首帧确认分辨率"
                    self._event_queue.put({"type": "status", "message": message})
                elif state.image_path is not None:
                    image_frame = cv2.imread(str(state.image_path))
                    if image_frame is None:
                        self._event_queue.put({"type": "error", "message": f"无法读取图片：{state.image_path}"})
                        time.sleep(0.3)
                        continue
                    self._event_queue.put({"type": "status", "message": f"图片已载入：{state.image_path.name}"})
                elif state.source_mode == "self_test":
                    image_frame = create_self_test_frame(
                        state.width if state.width > 0 else 640,
                        state.height if state.height > 0 else 480,
                    )
                    self._event_queue.put({"type": "status", "message": "已生成自测画面"})
                elif state.source_mode == "idle":
                    self._event_queue.put({"type": "status", "message": "摄像头已停止"})

            if state.source_mode == "camera":
                if cap is None:
                    time.sleep(0.02)
                    continue
                ok, frame = cap.read()
                if not ok:
                    self._event_queue.put({"type": "error", "message": "摄像头读取失败"})
                    time.sleep(0.03)
                    continue
            else:
                if image_frame is None:
                    time.sleep(0.03)
                    continue
                frame = image_frame.copy()
                time.sleep(1 / 30)

            with self._latest_raw_frame_lock:
                self._latest_raw_frame = frame.copy()

            now = time.monotonic()
            elapsed = max(now - last_frame_time, 1e-6)
            last_frame_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed

            try:
                result = detect_objects(frame, state.config)
                display = render_frame(frame, result, state)
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as exc:  # pragma: no cover - defensive GUI boundary
                self._event_queue.put({"type": "error", "message": f"识别处理失败：{exc}"})
                time.sleep(0.1)
                continue

            self._frame_queue.put(
                {
                    "rgb": rgb,
                    "raw_rgb": raw_rgb,
                    "fps": fps,
                    "shape": frame.shape[:2],
                    "all_rects": result.all_rect_details,
                    "filtered_rects": result.filtered_rect_details,
                    "center_rects": result.center_rect_details,
                    "closest_rect": result.closest_rect_details,
                    "has_hsv_target": bool(result.all_rects),
                    "has_final_target": result.closest_rect is not None,
                    "hsv_low": state.config.lower_hsv,
                    "hsv_high": state.config.upper_hsv,
                }
            )

            if state.print_interval > 0 and now - last_print_time >= state.print_interval:
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
                    bool(result.all_rects),
                    flush=True,
                )
                last_print_time = now

        if cap is not None:
            cap.release()

    @staticmethod
    def _source_signature(state: AppState) -> tuple:
        if state.source_mode == "camera":
            return (state.source_mode, state.camera_index)
        if state.source_mode == "self_test":
            return (state.source_mode, state.width, state.height)
        if state.source_mode == "idle":
            return (state.source_mode,)
        return (state.source_mode, str(state.image_path) if state.image_path else None)


def render_frame(frame, result, state: AppState):
    if state.view_mode == "mask":
        display = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    else:
        display = frame.copy()

    if state.show_center_region:
        region = center_region_bounds(frame.shape, state.config.center_region_ratio)
        cv2.rectangle(display, (region.x, region.y), (region.x2, region.y2), (255, 160, 0), 2)

    if state.show_reticle:
        frame_h, frame_w = frame.shape[:2]
        draw_center_reticle(display, (frame_w // 2, frame_h // 2))

    if state.box_mode == "all":
        boxes = result.all_rects
        box_color = (90, 220, 90)
    elif state.box_mode == "center":
        boxes = result.center_rects
        box_color = (0, 230, 255)
    else:
        boxes = []
        box_color = (90, 220, 90)

    for box in boxes:
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
        cv2.rectangle(display, (box.x, box.y), (box.x2, box.y2), (0, 230, 255), 3)
        cv2.circle(display, (int(box.center[0]), int(box.center[1])), 4, (0, 230, 255), -1)
        cv2.putText(
            display,
            f"{box.xywh} A={box.area}",
            (box.x, min(frame.shape[0] - 8, box.y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

    return display


class SliderRow(ttk.Frame):
    def __init__(self, parent, text: str, from_: int, to: int, value: int, command) -> None:
        super().__init__(parent)
        self.from_ = int(from_)
        self.to = int(to)
        self.value_var = tk.IntVar(value=value)
        ttk.Label(self, text=text, width=14).pack(side=tk.LEFT)
        self.slider = ttk.Scale(
            self,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
        )
        self.slider.set(value)
        self.slider.configure(command=lambda _value: self._changed(command))
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        self.input = ttk.Spinbox(
            self,
            from_=from_,
            to=to,
            textvariable=self.value_var,
            width=9,
            command=lambda: self._entry_changed(command),
        )
        self.input.pack(side=tk.LEFT)
        self.input.bind("<Return>", lambda _event: self._entry_changed(command))
        self.input.bind("<FocusOut>", lambda _event: self._entry_changed(command))

    def value(self) -> int:
        return int(round(float(self.slider.get())))

    def set_value(self, value: int) -> None:
        value = clamp_int(value, self.from_, self.to)
        self.slider.set(value)
        self.value_var.set(value)

    def _changed(self, command) -> None:
        self.value_var.set(self.value())
        command()

    def _entry_changed(self, command) -> None:
        try:
            value = int(self.value_var.get())
        except (tk.TclError, ValueError):
            value = self.value()
        value = clamp_int(value, self.from_, self.to)
        self.value_var.set(value)
        self.slider.set(value)
        command()


class App(tk.Tk):
    def __init__(self, initial_state: AppState) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1240x760")
        self.minsize(980, 620)

        self.frame_queue = LatestQueue()
        self.event_queue = LatestQueue()
        self.worker = VideoWorker(initial_state, self.frame_queue, self.event_queue)

        self.photo = None
        self.last_rgb = None
        self.last_raw_rgb = None
        self.last_image_size = (0, 0)
        self.last_display_size = (0, 0)
        self.last_display_origin = (0, 0)
        self.last_error_detail = ""
        self.video_message = "等待视频画面"
        self.sliders: dict[str, SliderRow] = {}
        self.resize_after_id = None
        self.picking_active = False
        self.preview_hsv: Optional[tuple[int, int, int]] = None
        self.preview_rgb: Optional[tuple[int, int, int]] = None
        self.preview_xy: Optional[tuple[int, int]] = None

        self.source_mode = tk.StringVar(value=initial_state.source_mode)
        self.camera_var = tk.IntVar(value=initial_state.camera_index)
        self.camera_devices = [CameraDevice(initial_state.camera_index, f"摄像头 {initial_state.camera_index}")]
        self.camera_device_var = tk.StringVar(value=self.camera_devices[0].label)
        self.width_var = tk.IntVar(value=initial_state.width)
        self.height_var = tk.IntVar(value=initial_state.height)
        self.view_mode = tk.StringVar(value=initial_state.view_mode)
        self.box_mode = tk.StringVar(value=initial_state.box_mode)
        self.show_center_var = tk.BooleanVar(value=initial_state.show_center_region)
        self.show_reticle_var = tk.BooleanVar(value=initial_state.show_reticle)
        self.status_var = tk.StringVar(value="就绪")

        self._build_ui(initial_state)
        self.refresh_cameras(show_status=False)
        if initial_state.source_mode == "camera" and not self.camera_devices:
            self.source_mode.set("idle")
            self.worker.update_state(source_mode="idle")
        self._push_state()
        self.worker.start()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(15, self._poll_queues)

    def _build_ui(self, state: AppState) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f4f6f8")
        style.configure("Panel.TFrame", background="#ffffff", relief=tk.FLAT)
        style.configure("TLabel", background="#ffffff", foreground="#253041", font=("Arial", 11))
        style.configure("Title.TLabel", font=("Arial", 12, "bold"))
        style.configure("TButton", padding=(10, 6))
        style.configure("TCheckbutton", background="#ffffff", font=("Arial", 11))
        style.configure("TRadiobutton", background="#ffffff", font=("Arial", 11))

        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        panel = self._create_scrollable_panel(root)

        video_outer = tk.Frame(root, bg="#111827")
        video_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_canvas = tk.Canvas(video_outer, bg="#111827", highlightthickness=0, bd=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Motion>", self._on_video_motion)
        self.video_canvas.bind("<Button-1>", self._on_video_click)
        self.video_canvas.bind("<Configure>", self._on_video_area_resize)
        self._show_video_message("等待视频画面")

        self._source_controls(panel)
        self._display_controls(panel)
        self._hsv_controls(panel, state.config)
        self._filter_controls(panel, state.config)
        self._processing_controls(panel, state.config)
        self._pick_controls(panel, state)
        self._result_labels(panel)
        ttk.Label(panel, textvariable=self.status_var, wraplength=330).pack(fill=tk.X, pady=(10, 0))

        self.bind("<plus>", lambda _event: self._zoom(10))
        self.bind("<equal>", lambda _event: self._zoom(10))
        self.bind("<minus>", lambda _event: self._zoom(-10))

    def _create_scrollable_panel(self, root):
        outer = ttk.Frame(root, style="Panel.TFrame", width=380)
        outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        outer.pack_propagate(False)

        panel_canvas = tk.Canvas(outer, bg="#ffffff", highlightthickness=0, bd=0, width=360)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=panel_canvas.yview)
        panel_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        panel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = ttk.Frame(panel_canvas, style="Panel.TFrame", padding=12)
        window_id = panel_canvas.create_window((0, 0), window=panel, anchor=tk.NW)

        def on_panel_configure(_event) -> None:
            panel_canvas.configure(scrollregion=panel_canvas.bbox("all"))

        def on_canvas_configure(event) -> None:
            panel_canvas.itemconfigure(window_id, width=event.width)

        def on_mousewheel(event) -> None:
            if event.delta:
                if abs(event.delta) >= 120:
                    units = int(-event.delta / 120)
                else:
                    units = -1 if event.delta > 0 else 1
                panel_canvas.yview_scroll(units, "units")

        def bind_mousewheel(_event) -> None:
            panel_canvas.bind_all("<MouseWheel>", on_mousewheel)
            panel_canvas.bind_all("<Button-4>", lambda _evt: panel_canvas.yview_scroll(-1, "units"))
            panel_canvas.bind_all("<Button-5>", lambda _evt: panel_canvas.yview_scroll(1, "units"))

        def unbind_mousewheel(_event) -> None:
            panel_canvas.unbind_all("<MouseWheel>")
            panel_canvas.unbind_all("<Button-4>")
            panel_canvas.unbind_all("<Button-5>")

        panel.bind("<Configure>", on_panel_configure)
        panel_canvas.bind("<Configure>", on_canvas_configure)
        panel_canvas.bind("<Enter>", bind_mousewheel)
        panel_canvas.bind("<Leave>", unbind_mousewheel)
        return panel

    def _group(self, parent, title: str):
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=(0, 0, 0, 10))
        frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(frame, text=title, style="Title.TLabel").pack(anchor=tk.W, pady=(0, 6))
        return frame

    def refresh_cameras(self, show_status: bool = True) -> None:
        if show_status:
            self.status_var.set("正在刷新摄像头列表...")
            self.update_idletasks()

        selected_index = int(self.camera_var.get())
        devices = discover_camera_devices()
        self.camera_devices = devices

        if not devices:
            self.camera_combo["values"] = ["未发现可用摄像头"]
            self.camera_device_var.set("未发现可用摄像头")
            if show_status:
                self.status_var.set("未发现可用摄像头，请检查连接或权限")
            return

        labels = [device.label for device in devices]
        self.camera_combo["values"] = labels
        selected_pos = 0
        for pos, device in enumerate(devices):
            if device.index == selected_index:
                selected_pos = pos
                break
        self.camera_combo.current(selected_pos)
        self.camera_device_var.set(labels[selected_pos])
        self.camera_var.set(devices[selected_pos].index)
        if show_status:
            self.status_var.set(f"已发现 {len(devices)} 个摄像头")

    def _on_camera_selected(self) -> None:
        index = self.camera_combo.current()
        if 0 <= index < len(self.camera_devices):
            device = self.camera_devices[index]
            self.camera_var.set(device.index)
            self.status_var.set(f"已选择：{device.label}")

    def _source_controls(self, parent) -> None:
        frame = self._group(parent, "输入源")
        row = ttk.Frame(frame, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="摄像头").pack(anchor=tk.W)
        self.camera_combo = ttk.Combobox(
            row,
            textvariable=self.camera_device_var,
            state="readonly",
        )
        self.camera_combo["values"] = [device.label for device in self.camera_devices]
        self.camera_combo.current(0)
        self.camera_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_camera_selected())
        self.camera_combo.pack(fill=tk.X, pady=(2, 0))

        row = ttk.Frame(frame, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)
        ttk.Button(row, text="刷新设备", command=self.refresh_cameras).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="打开摄像头", command=self.use_camera).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="停止摄像头", command=self.stop_camera).pack(side=tk.LEFT)

        row = ttk.Frame(frame, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="自测宽").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0, to=4096, textvariable=self.width_var, width=7, command=self._push_state).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="自测高").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0, to=2160, textvariable=self.height_var, width=7, command=self._push_state).pack(side=tk.LEFT, padx=6)

        row = ttk.Frame(frame, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)
        ttk.Button(row, text="打开图片", command=self.open_image).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="自测画面", command=self.use_self_test).pack(side=tk.LEFT)

    def _display_controls(self, parent) -> None:
        frame = self._group(parent, "显示")
        row = ttk.Frame(frame, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)
        for text, value in [("原画面", "original"), ("HSV掩膜", "mask")]:
            ttk.Radiobutton(row, text=text, value=value, variable=self.view_mode, command=self._push_state).pack(side=tk.LEFT)

        row = ttk.Frame(frame, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)
        self.box_combo = ttk.Combobox(row, state="readonly", width=18)
        self.box_combo["values"] = ("全部矩形框", "中心候选框", "只显示最终最近框")
        self.box_combo.current(0)
        self.box_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_box_combo())
        self.box_combo.pack(fill=tk.X)

        row = ttk.Frame(frame, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(row, text="显示中心框", variable=self.show_center_var, command=self._push_state).pack(side=tk.LEFT)
        ttk.Checkbutton(row, text="显示准星", variable=self.show_reticle_var, command=self._push_state).pack(side=tk.LEFT, padx=(10, 0))
        self.sliders["display_scale"] = SliderRow(frame, "显示缩放%", 25, 100, 100, self._on_zoom_slider)
        self.sliders["display_scale"].pack(fill=tk.X, pady=2)

    def _hsv_controls(self, parent, config: DetectionConfig) -> None:
        frame = self._group(parent, "HSV阈值")
        specs = [
            ("h_low", "H下限", 0, 179, config.lower_hsv[0]),
            ("s_low", "S下限", 0, 255, config.lower_hsv[1]),
            ("v_low", "V下限", 0, 255, config.lower_hsv[2]),
            ("h_high", "H上限", 0, 179, config.upper_hsv[0]),
            ("s_high", "S上限", 0, 255, config.upper_hsv[1]),
            ("v_high", "V上限", 0, 255, config.upper_hsv[2]),
        ]
        for key, text, low, high, value in specs:
            self.sliders[key] = SliderRow(frame, text, low, high, value, self._push_state)
            self.sliders[key].pack(fill=tk.X, pady=1)

    def _filter_controls(self, parent, config: DetectionConfig) -> None:
        frame = self._group(parent, "过滤")
        specs = [
            ("min_area_px", "最小像素面积", 0, 10_000_000, int(config.min_area_px)),
            ("max_area_px", "最大像素面积", 0, 10_000_000, int(config.max_area_px)),
            ("center_ratio", "中心区域%", 1, 100, int(config.center_region_ratio * 100)),
        ]
        for key, text, low, high, value in specs:
            self.sliders[key] = SliderRow(frame, text, low, high, value, self._push_state)
            self.sliders[key].pack(fill=tk.X, pady=1)

    def _processing_controls(self, parent, config: DetectionConfig) -> None:
        frame = self._group(parent, "性能与滤波")
        specs = [
            ("process_scale", "处理缩放%", 10, 100, int(config.process_scale * 100)),
            ("frame_blur", "原图滤波核", 0, 31, config.frame_blur_kernel),
            ("mask_median", "掩膜中值核", 0, 31, config.mask_median_kernel),
            ("morph", "形态学核", 0, 31, config.morph_kernel),
            ("open_iter", "开运算", 0, 5, config.open_iterations),
            ("close_iter", "闭运算", 0, 5, config.close_iterations),
        ]
        for key, text, low, high, value in specs:
            self.sliders[key] = SliderRow(frame, text, low, high, value, self._push_state)
            self.sliders[key].pack(fill=tk.X, pady=1)

    def _pick_controls(self, parent, state: AppState) -> None:
        frame = self._group(parent, "点击取色")
        row = ttk.Frame(frame, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=(0, 6))
        self.pick_button = ttk.Button(row, text="点击取色", command=self._toggle_pick_mode)
        self.pick_button.pack(side=tk.LEFT)
        ttk.Label(row, text="当前颜色").pack(side=tk.LEFT, padx=(12, 6))
        self.color_swatch = tk.Canvas(
            row,
            width=36,
            height=24,
            bg="#000000",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        self.color_swatch.pack(side=tk.LEFT)
        specs = [
            ("pick_radius", "取色半径(0精确)", 0, 50, state.pick_radius),
            ("h_pick_tol", "H容差", 0, 90, state.h_pick_tol),
            ("s_pick_tol", "S容差", 0, 255, state.s_pick_tol),
            ("v_pick_tol", "V容差", 0, 255, state.v_pick_tol),
        ]
        for key, text, low, high, value in specs:
            self.sliders[key] = SliderRow(frame, text, low, high, value, self._push_state)
            self.sliders[key].pack(fill=tk.X, pady=1)
        self.pick_text = tk.StringVar(value="取色：尚未点击")
        ttk.Label(frame, textvariable=self.pick_text, wraplength=330).pack(fill=tk.X, pady=(5, 0))

    def _result_labels(self, parent) -> None:
        frame = self._group(parent, "识别结果")
        self.result_vars = {
            "fps": tk.StringVar(value="FPS：-"),
            "hsv": tk.StringVar(value="HSV：-"),
            "all": tk.StringVar(value="全部框：-"),
            "filtered": tk.StringVar(value="像素面积过滤后：-"),
            "center": tk.StringVar(value="中心候选框：-"),
            "closest": tk.StringVar(value="最终最近框：-"),
        }
        for var in self.result_vars.values():
            ttk.Label(frame, textvariable=var, wraplength=330).pack(anchor=tk.W, fill=tk.X)

    def current_config(self) -> DetectionConfig:
        min_area_px = self.sliders["min_area_px"].value()
        max_area_px = self.sliders["max_area_px"].value()
        if max_area_px <= 0:
            max_area_px = 10_000_000
        if min_area_px > max_area_px:
            min_area_px = max_area_px

        return DetectionConfig(
            lower_hsv=(
                self.sliders["h_low"].value(),
                self.sliders["s_low"].value(),
                self.sliders["v_low"].value(),
            ),
            upper_hsv=(
                self.sliders["h_high"].value(),
                self.sliders["s_high"].value(),
                self.sliders["v_high"].value(),
            ),
            min_area_px=min_area_px,
            max_area_px=max_area_px,
            frame_blur_kernel=self.sliders["frame_blur"].value(),
            mask_median_kernel=self.sliders["mask_median"].value(),
            morph_kernel=self.sliders["morph"].value(),
            open_iterations=self.sliders["open_iter"].value(),
            close_iterations=self.sliders["close_iter"].value(),
            process_scale=clamp_process_scale(self.sliders["process_scale"].value() / 100.0),
            center_region_ratio=clamp_center_region_ratio(self.sliders["center_ratio"].value() / 100.0),
        )

    def _push_state(self) -> None:
        box_map = {
            "全部矩形框": "all",
            "中心候选框": "center",
            "只显示最终最近框": "closest",
        }
        self.worker.update_state(
            camera_index=int(self.camera_var.get()),
            width=int(self.width_var.get()),
            height=int(self.height_var.get()),
            config=self.current_config(),
            view_mode=self.view_mode.get(),
            box_mode=box_map.get(self.box_combo.get(), "all"),
            show_center_region=bool(self.show_center_var.get()),
            show_reticle=bool(self.show_reticle_var.get()),
            display_scale=self.sliders["display_scale"].value() / 100.0,
            pick_radius=self.sliders["pick_radius"].value(),
            h_pick_tol=self.sliders["h_pick_tol"].value(),
            s_pick_tol=self.sliders["s_pick_tol"].value(),
            v_pick_tol=self.sliders["v_pick_tol"].value(),
        )

    def _on_box_combo(self) -> None:
        self._push_state()

    def _on_zoom_slider(self) -> None:
        self._push_state()
        self._render_last_frame()

    def _on_video_area_resize(self, _event) -> None:
        if self.last_rgb is None:
            if self.video_message:
                self._show_video_message(self.video_message)
            return
        if self.resize_after_id is not None:
            self.after_cancel(self.resize_after_id)
        self.resize_after_id = self.after(40, self._render_after_resize)

    def _render_after_resize(self) -> None:
        self.resize_after_id = None
        self._render_last_frame()

    def _zoom(self, delta: int) -> None:
        slider = self.sliders["display_scale"]
        slider.set_value(max(25, min(100, slider.value() + delta)))
        self._on_zoom_slider()

    def use_camera(self) -> None:
        selected_pos = self.camera_combo.current()
        if 0 <= selected_pos < len(self.camera_devices):
            self.camera_var.set(self.camera_devices[selected_pos].index)
        elif not self.camera_devices:
            self.status_var.set("没有可打开的摄像头，请先刷新设备列表")
            self._show_video_message("没有可打开的摄像头，请先点击“刷新”")
            return

        self.source_mode.set("camera")
        self.worker.update_state(
            source_mode="camera",
            camera_index=int(self.camera_var.get()),
            width=int(self.width_var.get()),
            height=int(self.height_var.get()),
        )
        self.status_var.set(f"已切换到摄像头 {self.camera_var.get()}")
        self._show_video_message(f"正在打开摄像头 {self.camera_var.get()}，分辨率由设备决定...")

    def stop_camera(self) -> None:
        self.source_mode.set("idle")
        self.worker.update_state(source_mode="idle", image_path=None)
        self.last_rgb = None
        self.last_raw_rgb = None
        self.last_image_size = (0, 0)
        self.last_display_size = (0, 0)
        self.last_display_origin = (0, 0)
        self.photo = None
        self._show_video_message("摄像头已停止")
        self.status_var.set("摄像头已停止，可刷新或切换其它设备")
        self.result_vars["fps"].set("FPS：-")
        self.result_vars["hsv"].set("HSV：-")
        self.result_vars["all"].set("全部框：-")
        self.result_vars["filtered"].set("像素面积过滤后：-")
        self.result_vars["center"].set("中心候选框：-")
        self.result_vars["closest"].set("最终最近框：-")

    def open_image(self) -> None:
        file_name = filedialog.askopenfilename(
            title="选择测试图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if not file_name:
            return
        preview = cv2.imread(file_name)
        if preview is None:
            messagebox.showerror("图片读取失败", f"OpenCV 无法读取这张图片：\n{file_name}")
            self._show_error(f"无法读取图片：{file_name}")
            return
        self.source_mode.set("image")
        self.worker.update_state(source_mode="image", image_path=Path(file_name))
        self.status_var.set(f"已打开图片：{file_name}")
        self._show_video_message("图片已读取，正在运行识别...")

    def use_self_test(self) -> None:
        self.source_mode.set("self_test")
        self.worker.update_state(
            source_mode="self_test",
            image_path=None,
            width=int(self.width_var.get()),
            height=int(self.height_var.get()),
        )
        self.status_var.set("正在生成自测画面")
        self._show_video_message("正在生成自测画面...")

    def _on_video_click(self, event) -> None:
        if not self.picking_active:
            return
        sample = self._sample_pick_from_event(event)
        if sample is None:
            return
        x, y, hsv, rgb = sample
        self._set_color_swatch(rgb)
        self.preview_xy = (x, y)
        self.preview_hsv = hsv
        self.preview_rgb = rgb
        lower_hsv, upper_hsv = self._hsv_range_from_pick(hsv)
        self._apply_picked_hsv(
            {
                "xy": (x, y),
                "hsv": hsv,
                "lower_hsv": lower_hsv,
                "upper_hsv": upper_hsv,
            }
        )
        self._set_pick_mode(False)

    def _on_video_motion(self, event) -> None:
        if not self.picking_active:
            return
        sample = self._sample_pick_from_event(event)
        if sample is None:
            return
        x, y, hsv, rgb = sample
        self.preview_xy = (x, y)
        self.preview_hsv = hsv
        self.preview_rgb = rgb
        self._set_color_swatch(rgb)
        self.pick_text.set(f"预览坐标：{(x, y)} | 预览HSV：{hsv} | 左键确认")

    def _toggle_pick_mode(self) -> None:
        self._set_pick_mode(not self.picking_active)

    def _set_pick_mode(self, active: bool) -> None:
        self.picking_active = active
        if active:
            if self.last_raw_rgb is None:
                self.pick_text.set("取色：请先打开摄像头、图片或自测画面")
                self.picking_active = False
                return
            self.pick_button.configure(text="取消取色")
            self.video_canvas.configure(cursor="crosshair")
            self.pick_text.set("取色：移动鼠标预览颜色，左键确认")
        else:
            self.pick_button.configure(text="点击取色")
            self.video_canvas.configure(cursor="")

    def _sample_pick_from_event(self, event) -> Optional[tuple[int, int, tuple[int, int, int], tuple[int, int, int]]]:
        if self.last_raw_rgb is None or self.last_image_size == (0, 0):
            return None
        width, height = self.last_image_size
        display_w, display_h = self.last_display_size
        if display_w <= 0 or display_h <= 0:
            return None

        origin_x, origin_y = self.last_display_origin
        x_float = (event.x - origin_x) * width / display_w
        y_float = (event.y - origin_y) * height / display_h
        if not (0 <= x_float < width and 0 <= y_float < height):
            return None
        x = int(x_float)
        y = int(y_float)

        radius = max(0, self.sliders["pick_radius"].value())
        if radius <= 0:
            rgb = self.last_raw_rgb[y, x].astype(int)
            hsv = cv2.cvtColor(self.last_raw_rgb[y : y + 1, x : x + 1], cv2.COLOR_RGB2HSV)[0, 0].astype(int)
        else:
            x1 = clamp_int(x - radius, 0, width - 1)
            x2 = clamp_int(x + radius + 1, 1, width)
            y1 = clamp_int(y - radius, 0, height - 1)
            y2 = clamp_int(y + radius + 1, 1, height)
            patch = self.last_raw_rgb[y1:y2, x1:x2]
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            rgb = np.median(patch.reshape(-1, 3), axis=0).astype(int)
            hsv = np.median(hsv_patch.reshape(-1, 3), axis=0).astype(int)
        return (
            x,
            y,
            (int(hsv[0]), int(hsv[1]), int(hsv[2])),
            (int(rgb[0]), int(rgb[1]), int(rgb[2])),
        )

    def _hsv_range_from_pick(self, hsv: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        h, s, v = hsv
        h_tol = self.sliders["h_pick_tol"].value()
        if h_tol >= 90:
            low_h, high_h = 0, 179
        else:
            low_h = (h - h_tol) % 180
            high_h = (h + h_tol) % 180
        lower_hsv = (
            low_h,
            clamp_int(s - self.sliders["s_pick_tol"].value(), 0, 255),
            clamp_int(v - self.sliders["v_pick_tol"].value(), 0, 255),
        )
        upper_hsv = (
            high_h,
            clamp_int(s + self.sliders["s_pick_tol"].value(), 0, 255),
            clamp_int(v + self.sliders["v_pick_tol"].value(), 0, 255),
        )
        return lower_hsv, upper_hsv

    def _set_color_swatch(self, rgb: tuple[int, int, int]) -> None:
        color = "#{:02x}{:02x}{:02x}".format(*rgb)
        self.color_swatch.configure(bg=color)

    def _poll_queues(self) -> None:
        frame_item = self.frame_queue.get_nowait()
        if frame_item is not None:
            self.last_rgb = frame_item["rgb"]
            self.last_raw_rgb = frame_item.get("raw_rgb")
            self.last_image_size = (self.last_rgb.shape[1], self.last_rgb.shape[0])
            self._render_last_frame()
            self.result_vars["fps"].set(
                f"FPS：{frame_item['fps']:.1f} | 画面：{frame_item['shape'][1]}x{frame_item['shape'][0]}"
            )
            self.result_vars["hsv"].set(
                f"HSV：下限 {frame_item['hsv_low']} | 上限 {frame_item['hsv_high']}"
            )
            if frame_item["has_hsv_target"]:
                self.result_vars["all"].set(f"全部框：{frame_item['all_rects']}")
                self.result_vars["filtered"].set(f"像素面积过滤后：{frame_item['filtered_rects']}")
                self.result_vars["center"].set(f"中心候选框：{frame_item['center_rects']}")
            else:
                self.result_vars["all"].set("全部框：未检测到 HSV 范围内目标")
                self.result_vars["filtered"].set("像素面积过滤后：无")
                self.result_vars["center"].set("中心候选框：无")

            if frame_item["has_final_target"]:
                self.result_vars["closest"].set(f"最终最近框：{frame_item['closest_rect']}")
            else:
                self.result_vars["closest"].set("最终最近框：无")

        event_item = self.event_queue.get_nowait()
        if event_item is not None:
            if event_item["type"] == "error":
                self._show_error(event_item["message"], event_item.get("detail", ""))
            elif event_item["type"] == "status":
                self.status_var.set(event_item["message"])
            elif event_item["type"] == "picked":
                self._apply_picked_hsv(event_item)

        self.after(15, self._poll_queues)

    def _render_last_frame(self) -> None:
        if self.last_rgb is None:
            return
        try:
            image = Image.fromarray(self.last_rgb)
            size = self._fit_video_size(image.width, image.height)
            if size != (image.width, image.height):
                image = image.resize(size, pil_bilinear_resample())
            self.last_display_size = (image.width, image.height)
            self.photo = ImageTk.PhotoImage(image)
            canvas_w = max(1, self.video_canvas.winfo_width())
            canvas_h = max(1, self.video_canvas.winfo_height())
            origin_x = max(0, (canvas_w - image.width) // 2)
            origin_y = max(0, (canvas_h - image.height) // 2)
            self.last_display_origin = (origin_x, origin_y)
            self.video_message = ""
            self.video_canvas.delete("all")
            self.video_canvas.create_image(origin_x, origin_y, image=self.photo, anchor=tk.NW, tags=("frame",))
            self.status_var.set("画面更新正常")
        except Exception as exc:
            self._show_error(f"GUI显示帧失败：{exc}", traceback.format_exc())

    def _fit_video_size(self, image_w: int, image_h: int) -> tuple[int, int]:
        canvas_w = max(1, self.video_canvas.winfo_width())
        canvas_h = max(1, self.video_canvas.winfo_height())
        if canvas_w <= 1 or canvas_h <= 1 or image_w <= 0 or image_h <= 0:
            return (image_w, image_h)

        fill_scale = min(canvas_w / image_w, canvas_h / image_h)
        user_scale = max(0.25, min(1.0, self.sliders["display_scale"].value() / 100.0))
        final_scale = max(0.01, fill_scale * user_scale)
        return (
            max(1, int(round(image_w * final_scale))),
            max(1, int(round(image_h * final_scale))),
        )

    def _show_video_message(self, message: str) -> None:
        self.photo = None
        self.video_message = message
        self.last_display_size = (0, 0)
        self.last_display_origin = (0, 0)
        self.video_canvas.delete("all")
        canvas_w = max(1, self.video_canvas.winfo_width())
        canvas_h = max(1, self.video_canvas.winfo_height())
        wrapped = textwrap.fill(message, width=54)
        self.video_canvas.create_text(
            canvas_w // 2,
            canvas_h // 2,
            text=wrapped,
            fill="#e5e7eb",
            font=("Arial", 16, "bold"),
            justify=tk.CENTER,
            tags=("message",),
        )

    def _show_error(self, message: str, detail: str = "") -> None:
        self.last_error_detail = detail
        wrapped_message = textwrap.fill(message, width=72)
        self.status_var.set(message)
        self._show_video_message(f"{wrapped_message}\n\n可先点击“自测画面”验证显示和算法链路，或换摄像头编号。")

    def _apply_picked_hsv(self, payload: dict) -> None:
        lower_hsv = payload["lower_hsv"]
        upper_hsv = payload["upper_hsv"]
        for key, value in [
            ("h_low", lower_hsv[0]),
            ("s_low", lower_hsv[1]),
            ("v_low", lower_hsv[2]),
            ("h_high", upper_hsv[0]),
            ("s_high", upper_hsv[1]),
            ("v_high", upper_hsv[2]),
        ]:
            self.sliders[key].set_value(value)
        self.pick_text.set(f"取色坐标：{payload['xy']} | 取色HSV：{payload['hsv']}")
        self._push_state()

    def on_close(self) -> None:
        self.worker.stop()
        self.worker.join(timeout=1.2)
        self.destroy()


def require_runtime() -> None:
    if _CV_IMPORT_ERROR is not None:
        raise SystemExit("OpenCV 未安装，请先执行：python3 -m pip install -r requirements.txt") from _CV_IMPORT_ERROR
    if _PIL_IMPORT_ERROR is not None:
        raise SystemExit("Pillow 未安装，请先执行：python3 -m pip install -r requirements.txt") from _PIL_IMPORT_ERROR


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight Tkinter GUI for HSV detection.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--width", type=int, default=640, help="Self-test frame width; camera resolution is device-defined.")
    parser.add_argument("--height", type=int, default=480, help="Self-test frame height; camera resolution is device-defined.")
    parser.add_argument("--process-scale", type=float, default=0.75)
    parser.add_argument("--center-region-ratio", type=float, default=0.5)
    parser.add_argument("--min-area-px", type=int, default=0)
    parser.add_argument("--max-area-px", type=int, default=10_000_000)
    parser.add_argument("--hsv-low", type=parse_hsv_triplet, default=DEFAULT_LOWER_HSV)
    parser.add_argument("--hsv-high", type=parse_hsv_triplet, default=DEFAULT_UPPER_HSV)
    parser.add_argument("--print-interval", type=float, default=0.5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    require_runtime()
    cv2.setUseOptimized(True)

    state = AppState(
        source_mode="image" if args.image else "camera",
        camera_index=args.camera,
        image_path=args.image,
        width=args.width,
        height=args.height,
        config=DetectionConfig(
            lower_hsv=args.hsv_low,
            upper_hsv=args.hsv_high,
            min_area_px=max(0, args.min_area_px),
            max_area_px=max(max(0, args.min_area_px), args.max_area_px),
            process_scale=clamp_process_scale(args.process_scale),
            center_region_ratio=clamp_center_region_ratio(args.center_region_ratio),
        ),
        print_interval=args.print_interval,
    )

    app = App(state)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
