# 项目开发交接文档

## 1. 项目目标

本项目是一个 Python + OpenCV 的接近实时 HSV 目标识别程序。核心流程是：

```text
摄像头/图片输入
-> 可插拔原图处理
-> HSV 阈值生成二值掩膜
-> 可插拔掩膜处理
-> 轮廓提取
-> 全部矩形框
-> 像素面积过滤
-> 中心区域过滤
-> 选择最接近画面中心点的矩形框
-> GUI 显示、可选中心准星、终端坐标输出
```

主要代码文件：

- `hsv_detector.py`：检测核心、OpenCV HighGUI 调试版、命令行入口。
- `tk_hsv_detector.py`：推荐 GUI，使用标准库 Tkinter/ttk；GUI 主线程和视频识别线程分离，更适合轻量算法测试。
- `qt_hsv_detector.py`：可选实验 GUI，使用 PySide6；功能较完整但依赖较重。
- `image_system_test.py`：图片系统测试脚本，用于不打开 GUI 的回归测试。
- `test_detector_core.py`：核心 smoke test，覆盖默认红橙 HSV 跨 Hue 零点范围。
- `requirements.txt`：Python 依赖。
- `README.md`：用户运行说明。
- `DEVELOPMENT.md`：交接给下一位程序员的开发说明。

## 2. 运行方式

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

默认依赖面向轻量 Tkinter GUI。Qt 实验版额外依赖 PySide6：

```bash
python3 -m pip install -r requirements-qt.txt
```

摄像头实时运行：

```bash
python3 tk_hsv_detector.py --camera 0 --process-scale 0.75 --center-region-ratio 0.5
```

OpenCV HighGUI 调试版：

```bash
python3 hsv_detector.py --camera 0 --process-scale 0.5 --center-region-ratio 0.5
```

图片测试：

```bash
python3 hsv_detector.py --image /path/to/test.png --headless
```

图片系统测试：

```bash
python3 image_system_test.py /path/to/test.png --require-closest --save-dir test-output
```

保存图片识别结果：

```bash
python3 hsv_detector.py --image /path/to/test.png --headless --save-output output.png
```

## 3. 核心数据结构

`BoundingBox` 表示一个矩形框，坐标格式是 `(x, y, w, h)`：

- `x, y`：左上角坐标。
- `w, h`：宽和高。
- `x2, y2`：右下角坐标，由 `x + w` 和 `y + h` 得到。
- `center`：矩形框中心点。
- `area_ratio(frame_shape)`：矩形框面积占整幅画面的比例。
- `area`：矩形框像素面积，等于 `w * h`。

`DetectionConfig` 是检测参数：

- `lower_hsv` / `upper_hsv`：HSV 阈值，OpenCV 的 H 范围是 `0..179`。默认值是 `lower_hsv=(170, 90, 80)`、`upper_hsv=(18, 255, 255)`，偏向红橙色，并使用 Hue 跨零点范围。
- `min_area_px` / `max_area_px`：矩形框像素面积过滤范围，是 GUI 推荐使用的面积过滤方式。
- `min_area_ratio` / `max_area_ratio`：保留给旧命令行参数兼容；默认 `0..1`，通常不会额外过滤。
- `frame_blur_kernel`：原图高斯滤波核。
- `mask_median_kernel`：掩膜中值滤波核。
- `morph_kernel`：形态学操作核。
- `open_iterations` / `close_iterations`：开闭运算次数。
- `process_scale`：处理分辨率比例，`0.5` 表示宽高各缩小到 50% 后检测，再把坐标映射回原图。
- `center_region_ratio`：中心判定区域比例，`0.5` 表示用整幅画面宽高各 50% 的居中矩形作为中心区域。

`DetectionResult` 是检测结果：

- `mask`：最终二值掩膜，尺寸与输入 frame 一致。
- `all_rects` / `all_rect_details`：从轮廓提取出的全部矩形框，详情格式为 `(x, y, w, h, area)`。
- `filtered_rects` / `filtered_rect_details`：经过像素面积过滤后的矩形框。
- `center_rects` / `center_rect_details`：矩形框中心点落在中心判定区域内的候选框。
- `closest_rect` / `closest_rect_details`：从 `center_rects` 中选出的、最接近画面中心点的矩形框。没有候选时为 `None`。
- `has_hsv_target`：是否检测到 HSV 范围内的任何原始矩形框。
- `has_final_target`：是否存在最终最近目标。

没有检测到 HSV 范围内目标是正常状态，不是异常。GUI 应继续显示原画面或掩膜，只在结果面板显示“未检测到 HSV 范围内目标”，不要硬画目标框。

## 4. 中心区域判定

中心区域不是一个点，而是整幅画面等比例缩小后的居中矩形。

例如画面尺寸为 `1280x720`，`center_region_ratio=0.5`：

- 中心区域宽度是 `1280 * 0.5 = 640`
- 中心区域高度是 `720 * 0.5 = 360`
- 区域左上角是 `((1280 - 640) / 2, (720 - 360) / 2) = (320, 180)`
- 区域矩形是 `(320, 180, 640, 360)`

代码入口是：

- `center_region_bounds(frame_shape, center_region_ratio)`
- `filter_boxes_by_center_region(boxes, frame_shape, config)`
- `choose_closest_to_center(boxes, frame_shape)`

当前规则：目标矩形框的中心点落在中心区域内，才进入 `center_rects`。如果后续需要“矩形框与中心区域有交集即可算入”，应只修改 `filter_boxes_by_center_region()`。

GUI 里中心显示分为两种：

- `draw_center_reticle()`：画面正中央十字准星，用于观察目标离中心点多远；Tkinter GUI 通过 `显示准星` 复选框控制是否绘制。
- `显示中心框`：控制是否显示中心判定矩形边框。该开关只影响显示，不影响中心区域过滤逻辑。
- `显示准星`：控制是否显示中心点十字准星。该开关只影响显示，不影响检测、面积过滤或中心区域过滤逻辑。

## 5. 可插拔图像处理流程

原图处理函数类型：

```python
FrameProcessor = Callable[[Any, DetectionConfig], Any]
```

掩膜处理函数类型：

```python
MaskProcessor = Callable[[Any, DetectionConfig], Any]
```

当前处理列表：

```python
FRAME_PROCESSORS = [
    gaussian_blur_frame,
]

MASK_PROCESSORS = [
    median_blur_mask,
    morph_open_mask,
    morph_close_mask,
]
```

新增算法时，写一个同签名函数并加入列表即可。建议：

- 作用在 BGR 原图上的增强、降噪、白平衡类算法放入 `FRAME_PROCESSORS`。
- 作用在二值图上的去噪、膨胀、腐蚀、连通区域修复放入 `MASK_PROCESSORS`。
- 新增参数优先加到 `DetectionConfig`，再在 GUI 的 `create_controls()` 和 `read_controls()` 里加滑条。

## 6. 轻量 Tkinter GUI 线程模型

推荐入口是 `tk_hsv_detector.py`。它使用 Python 标准库 Tkinter/ttk，依赖比 Qt 轻，适合做算法测试工具。

线程职责：

- GUI 主线程：`App(tk.Tk)`、所有 Tk/ttk 控件、`ImageTk.PhotoImage` 创建和视频显示。
- 视频识别线程：`VideoWorker(threading.Thread)`，负责摄像头/图片读取、HSV 检测、画矩形框、按显示开关画中心区域和十字准星。

数据流：

```text
App 控件变化
-> VideoWorker.update_state(...)
-> VideoWorker 后台线程读取 AppState 快照
-> detect_objects(frame, config)
-> render_frame(...)
-> LatestQueue(maxsize=1)
-> App.after(...) 轮询队列
-> GUI 主线程创建 ImageTk.PhotoImage 并显示
```

这个设计有两个关键点：

- 后台线程不直接操作 Tk 控件，也不创建 `ImageTk.PhotoImage`，避免 Tk 线程安全问题。
- `LatestQueue(maxsize=1)` 只保留最新处理帧，避免 GUI 处理慢时积压旧帧。
- 摄像头输入分辨率由设备/驱动自行决定。`VideoWorker` 不主动设置 `CAP_PROP_FRAME_WIDTH` 或 `CAP_PROP_FRAME_HEIGHT`，只尝试读取设备报告的尺寸；读不到时以首帧 `frame.shape` 为准。
- 左侧视频区域使用 `tk.Canvas` 绘制缩放后的 `PhotoImage`，不要用 `tk.Label(image=...)` 直接承载大图，否则图像请求尺寸会把窗口撑大并挤压右侧面板。
- 右侧控制面板放在滚动 Canvas 中，固定保留在右侧；窗口高度不足时通过滚动访问全部控件。
- 摄像头下拉框由 `discover_camera_devices()` 填充。OpenCV 跨平台不可靠暴露摄像头名称，因此 macOS 上会尽量用 `system_profiler SPCameraDataType` 读取名称，并按扫描到的 OpenCV 设备顺序配对；读不到名称时用 `摄像头 N` 加设备编号和分辨率兜底。
- `停止摄像头` 会把 `source_mode` 切到 `idle`，触发 `VideoWorker` 释放当前 `VideoCapture`。用户刷新或切换设备前建议先停止，避免当前占用的摄像头在扫描时打不开。
- 错误事件会直接显示到左侧视频区域，不再只放在右侧底部状态栏，避免用户看不到失败原因。
- `自测画面` 输入源使用 `create_self_test_frame()` 生成内置橙红测试图，不依赖摄像头和外部图片，用来验证 GUI 显示和识别链路。

点击取色数据流：

```text
右侧“点击取色”按钮
-> App 进入 picking_active 状态
-> 鼠标在左侧 tk.Canvas 视频区域移动
-> 按显示缩放和图像在 Canvas 中的居中偏移映射回原图坐标
-> 从 frame_queue 附带的 latest raw_rgb 中取鼠标指针下像素，或按取色半径采样
-> 右侧当前颜色小方块和预览 HSV 实时更新
-> 鼠标左键确认
-> App 按 H/S/V 容差更新 HSV 滑条并退出取色状态
```

Tkinter 版的取色预览不要走后台线程事件队列。鼠标移动频率很高，GUI 主线程直接从最新原始 RGB 帧采样更轻，也能避免把预览事件堆进识别线程。`VideoWorker.pick_hsv()` 仍保留为备用接口，但当前 GUI 流程优先使用 `App._sample_pick_from_event()` 和 `App._hsv_range_from_pick()`。

## 7. Qt GUI 线程模型

`qt_hsv_detector.py` 保留为可选实验版。它使用 PySide6，界面能力更强，但依赖更重；当前项目作为算法测试 GUI，优先维护 Tkinter 版。

线程职责：

- GUI 主线程：`MainWindow`、所有 Qt 控件、视频缩放显示、按钮和滑条交互。
- 视频识别线程：`VideoWorker(QThread)`，负责摄像头/图片读取、HSV 检测、画矩形框、画中心区域和十字瞄准框。

数据流：

```text
MainWindow 控件变化
-> VideoWorker.update_state(...)
-> VideoWorker 后台线程读取 WorkerState 快照
-> detect_objects(frame, config)
-> render_frame(...)
-> frame_ready(QImage, meta)
-> MainWindow 更新视频和识别结果
```

点击取色数据流：

```text
VideoLabel 鼠标点击
-> 原图坐标 x/y
-> VideoWorker.pick_hsv_at(x, y)
-> 后台线程保存的最新 frame 中取 HSV
-> hsv_picked(payload)
-> MainWindow 更新 HSV 滑条
```

Qt 版 GUI 的布局缩放在 GUI 线程完成：`VideoLabel.set_image(image, scale)` 根据缩放比例调整 `QPixmap` 和 QLabel 固定尺寸，外层 `QScrollArea` 自动处理滚动和居中。检测结果坐标始终保持原图坐标。

## 8. 实时性能设计

当前优化点：

- 摄像头模式默认使用 `CameraSource` 后台线程持续取帧。
- 主线程只处理“最新帧”，避免摄像头缓冲旧帧造成延迟。
- `cv2.CAP_PROP_BUFFERSIZE` 设置为 `1`，尽量降低后端缓冲。
- `process_scale` 支持低分辨率检测，高分辨率显示。
- GUI 坐标输出用 `--print-interval` 限频，避免 stdout 拖慢循环。
- `cv2.setUseOptimized(True)` 开启 OpenCV 优化路径。

性能调优优先级：

1. 降低 `--process-scale`，例如 `0.5` 或 `0.33`，优先降低处理分辨率而不是强改摄像头输入分辨率。
2. 减小滤波核和形态学迭代次数。
3. 关闭终端频繁打印：`--print-interval 0`。
4. 如后续目标很多，可考虑只在 ROI 内做 HSV 检测，或用 `connectedComponentsWithStats` 替代轮廓。

如需对比单线程取帧，可加：

```bash
python3 hsv_detector.py --sync-camera
```

## 9. GUI 控件和快捷键

`HSV控制面板` 窗口中的控件：

- `H下限`, `S下限`, `V下限`
- `H上限`, `S上限`, `V上限`
- `最小像素面积`
- `最大像素面积`
- `原图滤波核`
- `掩膜中值核`
- `形态学核`
- `开运算次数`
- `闭运算次数`
- `处理缩放%`
- `中心区域%`
- `显示中心框`：`1` 显示中心判定矩形，`0` 隐藏。
- `显示准星`：显示或隐藏画面中心点十字准星。
- `取色半径(0精确)`
- `H取色容差`
- `S取色容差`
- `V取色容差`
- `显示缩放%`

快捷键：

- `q` / `Esc`：退出。
- `m`：切换原画面和 HSV 掩膜。
- `b`：循环切换 `all`、`center`、`closest` 三种框显示模式。
- `+` / `-`：放大或缩小 GUI 显示画面。
- 鼠标左键：仅在右侧 `点击取色` 模式中确认当前预览颜色，按容差自动设置 HSV 阈值。
- `c`：下一个摄像头。
- `p`：上一个摄像头。
- `0` 到 `9`：切换到对应摄像头编号。

## 10. GUI 中文显示和缩放

OpenCV 自带 `cv2.putText()` 对中文支持有限，因此项目使用 Pillow 绘制画面上的中文状态文字。依赖在 `requirements.txt` 中：

```text
pillow>=10.0.0
```

相关代码：

- `get_chinese_font(size)`：优先查找 macOS 和常见 Linux 中文字体。
- `draw_text_lines(image, lines)`：用 Pillow 在 OpenCV BGR 图像上绘制中文文本；如果 Pillow 不可用，会退回 ASCII 文本。
- `App._fit_video_size(image_w, image_h)`：Tkinter GUI 按左侧 Canvas 视频区域计算等比适配尺寸，保持摄像头/图片原始宽高比。
- `resize_for_display(image, display_scale)`：OpenCV HighGUI 调试版按 `显示缩放%` 对最终显示画面缩放。
- `read_display_scale()` / `adjust_display_scale(delta)`：读取滑条缩放值，并支持 `+/-` 快捷键。

缩放策略：

1. 检测、坐标过滤、中心区域计算始终在原图坐标系完成。
2. 先在原图尺寸上绘制矩形框、中心区域和十字准星，其中中心区域和准星分别受显示开关控制。
3. Tkinter GUI 把整张显示画面按左侧 Canvas 视频框做等比最大适配，`显示缩放%=100` 时尽可能显示完整并占满可用区域；低于 `100` 时按比例缩小并留黑边。
4. OpenCV HighGUI 调试版把整张显示画面按 `display_scale` 缩放。
5. 最后绘制中文状态文本。

这样放大或缩小时，图像、矩形框、中心区域和准星会同步缩放，检测输出坐标仍保持原图坐标。Tkinter GUI 不裁切、不拉伸画面；左侧视频框比例和输入比例不一致时，会出现上下黑边或左右黑边。高分辨率视频只会缩放绘制到 Canvas 中，不应该改变窗口整体请求尺寸。

Tkinter GUI 取色时，`App._sample_pick_from_event()` 会读取当前 `display_scale`，同时使用 `last_display_origin` 计算图像在 Canvas 中的居中偏移，把窗口鼠标坐标映射回原图坐标再取色。不要只用 `event.x / scale`，否则当视频画面没有填满左侧区域时，取色点会发生漂移。

## 11. HSV 点击取色

OpenCV HighGUI 没有真正的颜色选择器组件，所以保留鼠标点击取色来提升调参直观性。推荐的 Tkinter GUI 已经有右侧取色按钮和当前颜色预览小方块。

Tkinter GUI 取色流程：

1. 用户点击右侧 `点击取色` 按钮。
2. `App._set_pick_mode(True)` 设置 `picking_active=True`，视频区域鼠标指针改为十字。
3. 鼠标在视频区域移动时，`App._on_video_motion()` 调用 `App._sample_pick_from_event()`。
4. `_sample_pick_from_event()` 使用 `last_display_origin` 和 `last_display_size` 把 Canvas 坐标映射回原图坐标，并从 `last_raw_rgb` 采样。这里使用的是原始 RGB 帧，不是已经画框或转成掩膜的显示帧。
5. `取色半径=0` 时取鼠标指针下单个像素；半径大于 0 时才取周围区域 RGB 和 HSV 中位数。RGB 用于更新右侧颜色小方块，HSV 用于预览文本。
6. 用户按鼠标左键时，`App._on_video_click()` 确认当前颜色，调用 `_hsv_range_from_pick()` 按容差生成 HSV 上下限，更新 HSV 滑条，并退出取色状态。

OpenCV HighGUI 取色流程：

相关代码：

- `HsvPickerState`
- `set_hsv_trackbars_from_pick(hsv)`
- `cv2.setMouseCallback(VIEW_WINDOW, hsv_picker.on_mouse)`

交互逻辑：

1. 主循环用 `hsv_picker.set_frame(frame)` 保存最新原图。
2. 用户在 `HSV检测器` 窗口左键点击。
3. `HsvPickerState.on_mouse()` 在点击点周围按 `取色半径px` 取采样区域。
4. 采样区域转换到 HSV，取每个通道的中位数作为 `取色HSV`。
5. 使用 `H/S/V取色容差` 生成新的 HSV 上下限。
6. 用 `cv2.setTrackbarPos()` 更新 HSV 滑条。

H 通道支持跨零点范围。例如点击到 `H=2`，`H取色容差=10` 时，低阈值会变成 `172`，高阈值会变成 `12`。`create_hsv_mask()` 已经处理了 `lower_h > upper_h` 的 Hue 环绕情况。注意跨零点时传给 `cv2.inRange()` 的两个边界数组必须都是 `np.uint8`，因此代码使用 `hsv_bound()` 统一创建边界数组。

## 12. 坐标和缩放注意事项

即使启用了 `process_scale`，所有对外输出坐标仍然是原图坐标系。

实现方式：

1. 原图按 `process_scale` 缩小。
2. 在缩小图上生成掩膜、找轮廓、算矩形框。
3. 用 `BoundingBox.scale(scale_x, scale_y)` 把矩形框映射回原图。
4. 掩膜用 `INTER_NEAREST` 放大回原图尺寸，方便 GUI 显示。

像素面积过滤和中心区域过滤都在原图坐标系中执行。GUI 中的 `最小像素面积` 和 `最大像素面积` 同时提供滑条和输入框；滑条用于粗调，输入框用于精确输入。画面上的矩形框标签显示 `(x, y, w, h) A=area`，中心候选框在中心候选模式中使用黄色标注。

## 13. 图片系统测试

`image_system_test.py` 用于把“手动打开图片看看效果”变成可重复的系统测试。

常用命令：

```bash
python3 image_system_test.py /path/to/test.png --require-closest
python3 image_system_test.py image1.png image2.png --min-filtered 1 --save-dir test-output
```

输出格式是每张图片一行 JSON：

```json
{
  "image": "/path/to/test.png",
  "width": 1280,
  "height": 720,
  "all_rects": [],
  "filtered_rects": [],
  "center_rects": [],
  "closest_rect": null,
  "passed": false,
  "failures": ["closest_rect 为空"]
}
```

可用断言：

- `--min-all N`：要求至少 N 个原始矩形框。
- `--min-filtered N`：要求至少 N 个面积过滤后的矩形框。
- `--min-center N`：要求至少 N 个中心区域候选框。
- `--require-closest`：要求必须找到最终最近框。

可视化输出：

- `--save-dir test-output` 会保存带框图片。
- `--display mask` 可保存 HSV 掩膜视图。
- `--boxes all|center|closest` 可切换保存结果里的框显示模式。

该脚本复用 `detect_objects()`，因此它测试的是和 Qt GUI 一致的核心识别链路。建议后续每次调整 HSV、滤波流程、中心区域规则、面积过滤规则后，都用固定图片集跑一遍。

## 14. 建议测试清单

基础测试：

```bash
python3 -m py_compile hsv_detector.py tk_hsv_detector.py qt_hsv_detector.py image_system_test.py test_detector_core.py
python3 test_detector_core.py
```

图片识别测试：

```bash
python3 image_system_test.py /path/to/test.png --require-closest --hsv-low 170,90,80 --hsv-high 18,255,255
python3 hsv_detector.py --image /path/to/test.png --headless --hsv-low 170,90,80 --hsv-high 18,255,255
```

GUI 测试：

```bash
python3 tk_hsv_detector.py --image /path/to/test.png
python3 tk_hsv_detector.py --camera 0 --process-scale 0.75
python3 hsv_detector.py --image /path/to/test.png
python3 hsv_detector.py --camera 0 --process-scale 0.5
```

重点确认：

- 点击 `自测画面` 后左侧能显示内置测试图。
- HSV 掩膜能稳定覆盖目标颜色。
- `all_rects`、`filtered_rects`、`center_rects`、`closest_rect` 输出符合预期。
- 调小 `中心区域%` 后，边缘目标会从 `center_rects` 中消失。
- `显示中心框` 为 `0` 时 GUI 不显示中心矩形边框，但检测结果仍按中心区域过滤。
- `显示准星` 关闭后 GUI 不显示中心十字准星，但检测结果不变。
- 点击右侧 `点击取色` 后，鼠标移动会更新当前颜色小方块；左键确认后 `取色HSV` 显示更新，HSV 滑条随之变化。
- 拖动 `显示缩放%` 或按 `+/-` 后，画面、矩形框、中心区域和准星同步缩放。
- 摄像头切换失败时能退回原摄像头。

## 15. 后续可扩展方向

- 增加配置文件，例如 `config.yaml`，保存 HSV 和过滤参数。
- 增加 FPS 统计窗口，区分取帧 FPS、检测 FPS、显示 FPS。
- 增加 ROI 检测，只处理画面中部或用户指定区域。
- 对不同颜色目标维护多组 HSV 范围。
- 如需更完整的 GUI，可迁移到 PyQt 或 Tkinter，把滑条、复选框、颜色预览、参数保存做成独立面板。
- 把检测核心拆成包，例如 `detector/core.py`、`detector/gui.py`、`detector/sources.py`，便于大型项目继续维护。
