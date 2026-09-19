# HSV Mask OpenCV 实时识别项目

这个项目用 Python + OpenCV 做接近实时的 HSV 颜色分割识别：

- 给定 HSV 范围生成掩膜
- 对原画面和掩膜执行可插拔图像处理流程
- 从掩膜中找轮廓和矩形框
- 按矩形框面积占整张图的比例过滤
- 用“整幅画面等比例缩小后的居中矩形”作为可调中心区域
- 从面积过滤后、且落入中心区域的矩形框中选择最接近画面中心点的目标
- 以图传中心点为 `(0, 0)` 计算最终最近框中心点的中央坐标系偏移，右/上为正，左/下为负
- GUI 默认显示画面正中央十字准星，并可用开关隐藏，便于判断目标离中心点的距离
- Tkinter GUI 会在右侧工具栏顶部显示中央坐标系偏移，并通过 FastAPI 暴露 `/get-offset-xy`
- Tkinter GUI 支持保存当前右侧参数为 JSON、加载 JSON 配置、恢复程序默认配置，并可自动加载 `default.json`
- 支持右侧按钮进入取色模式，鼠标移动实时预览颜色，左键确认后自动设置 HSV 阈值
- 摄像头模式默认使用后台线程持续取最新帧，降低取帧阻塞造成的延迟
- 支持降低处理分辨率来提升 FPS，并把矩形框坐标映射回原图尺寸
- GUI 中可切换原画面/HSV 掩膜、全部矩形框/中心最近矩形框、摄像头编号，并支持同步缩放显示
- 支持只打开一张图片进行测试

## 安装

```bash
python3 -m pip install -r requirements.txt
```

默认依赖只包含 OpenCV、NumPy、Pillow。Tkinter 通常随 Python 自带。Qt 实验版需要额外安装：

```bash
python3 -m pip install -r requirements-qt.txt
```

## 摄像头实时运行

推荐使用轻量 Tkinter GUI：

```bash
python3 tk_hsv_detector.py
```

Tkinter GUI 使用 Python 标准库 `tkinter/ttk`，比 Qt 更轻，适合作为测试算法的工具窗口。界面是左侧视频画面、右侧中文控制面板，包含复选框、按钮、下拉框、文件选择器和可缩放视频区域。右侧控制面板固定保留并可滚动，左侧视频区域跟随窗口剩余空间变化。GUI 主线程只负责界面，摄像头读取、图片读取、HSV 检测和画框在 `VideoWorker(threading.Thread)` 后台线程中完成。

摄像头输入分辨率默认由设备和驱动自行决定，程序只尝试读取实际返回的帧尺寸并显示在结果面板中。左侧视频框使用 Canvas 绘制视频，不会被高分辨率画面撑大窗口；画面会保持输入宽高比，在不裁切、不拉伸的前提下尽可能显示完整，比例不一致时允许出现上下黑边或左右黑边。

摄像头选择使用下拉框。点击 `刷新` 会扫描可打开的摄像头，并尽量显示设备名称、设备编号和当前报告的分辨率；如果系统或 OpenCV 读不到设备名称，会退回显示 `摄像头 N`。点击 `停止摄像头` 会释放当前摄像头并清空视频画面，方便刷新列表或切换到其它设备。

轻量 GUI 的实时运行示例：

```bash
python3 tk_hsv_detector.py --camera 0 --process-scale 0.75 --center-region-ratio 0.5
```

启动后会同时启动 FastAPI 接口，默认监听 `0.0.0.0:8005`，允许同一局域网内的设备访问。右侧界面会显示本机当前局域网 IP，例如：

```text
http://<本机局域网IP>:8005/get-offset-xy
http://<本机局域网IP>:8005/if-obj-stable
```

接口返回当前“最终最近框”在中央坐标系下的偏移量和偏移比例，例如：

```json
{"x-offset":50,"x-offset-rate":0.16,"y-offset":-50,"y-offset-rate":-0.21}
```

如果当前没有最终最近框，接口返回：

```json
{"x-offset":null,"x-offset-rate":null,"y-offset":null,"y-offset-rate":null}
```

稳定识别接口 `/if-obj-stable` 返回当前最终最近框是否能在最近帧窗口内稳定出现：

```json
{"if-obj-stable":1}
```

或：

```json
{"if-obj-stable":0}
```

稳定判定使用最近 `10` 帧窗口：有框帧数至少达到 `80%`，中心点相邻帧位移变化平滑，面积变化比例保持在 `0.5..2.0`，宽高比变化比例保持在 `0.6..1.6`。窗口连续 `3` 帧判定稳定后接口返回 `1`，连续 `3` 帧判定不稳定后返回 `0`。这里的“框”使用和偏移量一致的最终最近框，因此会经过 HSV、面积过滤和中心区域筛选。

可以用 `--api-host` 和 `--api-port` 修改接口监听地址；如果 `--api-host` 使用默认的 `0.0.0.0`，界面展示时会自动换成本机局域网 IP。这个偏移量使用原始图像像素宽高计算，不使用 GUI 缩放后的显示尺寸，因此不同电脑窗口大小、屏幕分辨率不同，也不会改变同一输入画面下的偏移结果。

偏移比例同样基于原始图像尺寸：x 轴边界是 `-画面宽度/2..画面宽度/2`，y 轴边界是 `-画面高度/2..画面高度/2`。例如 640x480 输入下，x 轴边界为 `-320..320`，y 轴边界为 `-240..240`；若中央坐标系偏移为 `(50, -50)`，比例为 `50/320=0.16` 和 `-50/240=-0.21`，保留两位小数。

右侧 `配置` 区有三个按钮：

- `保存当前配置`：保存 HSV、过滤、性能与滤波、显示、取色、自测尺寸和摄像头编号等右侧参数。点击后会询问是否保存为默认配置；选择“是”会写入当前运行目录下的 `default.json`，选择“否”会弹出文件选择器另存为其它 JSON 文件。
- `加载配置`：从 JSON 文件读取并立即应用到右侧参数和后台识别状态。
- `恢复默认`：恢复为程序启动时的默认参数。

程序启动时会自动搜索 `default.json`：优先查找当前运行目录，其次查找脚本所在目录。找到后会自动加载；找不到则不自动加载。

轻量 GUI 打开图片测试：

```bash
python3 tk_hsv_detector.py --image /path/to/test.png
```

启动轻量 GUI 后，也可以点击右侧 `打开图片` 按钮直接用图片调试。若左侧一直显示“等待视频画面”，先点击 `自测画面`：它会生成一张内置橙红色测试图，不依赖摄像头和外部图片，用来确认 GUI 显示链路是否正常。Qt 版本 `qt_hsv_detector.py` 保留为可选实验版，如需运行可安装 `requirements-qt.txt`。

保留的 OpenCV HighGUI 调试版：

```bash
python3 hsv_detector.py
```

默认 HSV 是偏红的橙色范围：`--hsv-low 170,90,80 --hsv-high 18,255,255`。这里 `H下限 > H上限` 是有意设计，用来覆盖 OpenCV HSV 中红色跨越 `0/179` 的情况。

常用参数：

```bash
python3 hsv_detector.py --camera 0 --max-camera-index 4 --hsv-low 170,90,80 --hsv-high 18,255,255
```

更偏实时性的运行方式：

```bash
python3 hsv_detector.py --camera 0 --process-scale 0.5 --center-region-ratio 0.5
```

`--process-scale 0.5` 表示用宽高各 50% 的图做 HSV、滤波、轮廓检测，然后把输出矩形框坐标还原到原始画面坐标系。摄像头模式默认使用多线程取帧：后台线程尽快从摄像头读取最新画面，主线程只处理最新帧，减少旧帧堆积导致的延迟。如果需要对比单线程行为，可以加 `--sync-camera`。

`--center-region-ratio 0.5` 表示中心判定区域是整幅画面宽高各 50% 的居中矩形。目标矩形中心点落在该区域内，才会进入 `center_rects` 候选集合。

轻量 GUI 和 OpenCV HighGUI 都可以调整这些核心参数。OpenCV HighGUI 版本会在 `HSV控制面板` 窗口中实时调整：

- `H下限/S下限/V下限` 和 `H上限/S上限/V上限`：HSV 阈值
- `最小像素面积`：矩形框最小像素面积，`0` 表示允许非常小的框进入候选
- `最大像素面积`：矩形框最大像素面积，用于过滤过大的误检区域
- `原图滤波核`：原画面高斯滤波核大小
- `掩膜中值核`：掩膜中值滤波核大小
- `形态学核`：形态学处理核大小
- `开运算次数`：开运算次数，用于去小噪点
- `闭运算次数`：闭运算次数，用于补目标内部空洞
- `处理缩放%`：处理分辨率比例，降低后通常能明显提升 FPS
- `中心区域%`：中心判定矩形的宽高比例
- `显示中心框`：`1` 显示中心判定矩形边框，`0` 隐藏
- `显示准星`：显示或隐藏画面正中央十字准星，只影响辅助显示
- `取色半径(0精确)`：`0` 表示取鼠标指针正下方单个像素；调大后使用周围区域 HSV 中位数抗噪
- `H取色容差`：点击取色后，自动生成 HSV 范围时的 H 容差
- `S取色容差`：点击取色后，自动生成 HSV 范围时的 S 容差
- `V取色容差`：点击取色后，自动生成 HSV 范围时的 V 容差
- `显示缩放%`：Tkinter GUI 中 `100%` 表示等比最大填充左侧视频框，调低后画面、矩形框、中心框和准星同步缩小

更直观地调整 HSV：

1. 按 `m` 切到原画面。
2. 点击右侧 `点击取色` 按钮，进入取色模式。
3. 鼠标在视频画面上移动时，右侧 `当前颜色` 小方块会实时显示鼠标指针正下方的颜色，并显示预览 HSV。
4. 在目标颜色上按鼠标左键确认，例如交通锥中部的橙色区域。
5. 程序会显示 `取色坐标` 和 `取色HSV`，并按 `H/S/V取色容差` 自动设置 HSV 上下限。
6. 再微调 HSV 上下限或容差滑条，让掩膜覆盖目标、避开背景。

取色预览始终基于原始画面颜色，即使当前显示模式切到了 `HSV掩膜`，也不会从黑白掩膜图上取色。

实时性能建议：

- 摄像头输入分辨率由设备决定，优先用 `处理缩放%` 控制算法负载。
- `处理缩放%` 可以先试 `50`，小目标较多时再调高。
- 滤波核和形态学迭代不要过大，`5` 和 `1~2` 通常够用。
- GUI 模式默认每 `0.5` 秒打印一次坐标；如果只看画面，可用 `--print-interval 0` 减少终端输出开销。

## GUI 快捷键

| 按键 | 功能 |
| --- | --- |
| `q` / `Esc` | 退出 |
| `m` | 切换原画面 / HSV 掩膜 |
| `b` | 切换全部矩形框 / 中心区域候选框 / 只显示最终最近框 |
| `+` / `-` | 放大 / 缩小 GUI 显示画面 |
| 鼠标左键 | 在 `点击取色` 模式中确认当前预览颜色 |
| `c` | 切换到下一个可打开摄像头 |
| `p` | 切换到上一个可打开摄像头 |
| `0`-`9` | 直接切换到对应摄像头编号 |

## 图片测试

轻量 GUI 直接打开图片：

```bash
python3 tk_hsv_detector.py --image /path/to/test.png
```

启动轻量 GUI 后，也可以点击右侧 `打开图片` 按钮直接打开图片调试。

轻量 GUI 内置自测画面：

```bash
python3 tk_hsv_detector.py
```

启动后点击右侧 `自测画面`。如果自测画面能显示，但摄像头没有画面，通常是摄像头编号或 macOS 摄像头权限问题；如果自测画面能显示，但某张图片不显示，通常是该图片路径或格式不能被 OpenCV 读取。

使用图片做系统测试，不打开 GUI：

```bash
python3 image_system_test.py /path/to/test.png --require-closest --save-dir test-output
```

核心 smoke test：

```bash
python3 test_detector_core.py
```

批量测试多张图片：

```bash
python3 image_system_test.py image1.png image2.png image3.png --min-filtered 1 --save-dir test-output
```

脚本会输出 JSON，包含 `all_rects`、`filtered_rects`、`center_rects`、`closest_rect` 和 `passed`。`--save-dir` 会保存带框结果图，方便人工检查。

OpenCV HighGUI 调试版打开图片：

```bash
python3 hsv_detector.py --image /path/to/test.png
```

只跑一次识别并打印坐标，不打开 GUI：

```bash
python3 hsv_detector.py --image /path/to/test.png --headless
```

保存带框结果：

```bash
python3 hsv_detector.py --image /path/to/test.png --headless --save-output output.png
```

## 在代码中获取矩形框坐标

`detect_objects(frame, config)` 会返回 `DetectionResult`：

```python
import cv2
from hsv_detector import DetectionConfig, detect_objects

frame = cv2.imread("test.png")
config = DetectionConfig(
    lower_hsv=(170, 90, 80),
    upper_hsv=(18, 255, 255),
    min_area_px=0,
    max_area_px=10000000,
    process_scale=0.5,
    center_region_ratio=0.5,
)

result = detect_objects(frame, config)

print("全部矩形框:", result.all_rect_details)
print("像素面积过滤后的矩形框:", result.filtered_rect_details)
print("中心区域内的候选矩形框:", result.center_rect_details)
print("最接近画面中心的矩形框:", result.closest_rect_details)
print("中央坐标系偏移:", result.closest_offset_xy)
print("中央坐标系偏移比例:", result.closest_offset_rate_xy)
```

矩形框详情格式是 `(x, y, w, h, area)`，其中 `area = w * h`，单位是像素。
中央坐标系偏移格式是 `(x, y)`，单位同样是原始图像像素；右和上为正，左和下为负。
中央坐标系偏移比例格式是 `(x_rate, y_rate)`，按原始图像半宽和半高归一化后保留两位小数。

## 新增或修改图像处理流程

主程序里有两个列表：

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

新增处理算法时，只要写一个函数并加入对应列表即可：

```python
def dilate_mask(mask, config):
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)

MASK_PROCESSORS.append(dilate_mask)
```

如果算法作用在原图上，加入 `FRAME_PROCESSORS`；如果作用在二值掩膜上，加入 `MASK_PROCESSORS`。
