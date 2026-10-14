# SFRreg 四边缘自动提取

从图像中自动检测 Imatest **SFRreg** 注册标（圆内两个亮象限、两个暗象限），在四条倾斜半边上生成轴对齐 ROI，并输出标注图、局部放大图和 JSON 坐标。

ROI 几何参考 Imatest 23.2.6 SFRreg 默认设置：`fillf = 0.6`、`roilth = 0.85`、ROI width = Normal。边缘顺序同 Imatest：垂直边 **T、B**，然后是水平边 **L、R**。

> 本工具负责注册标检测、ROI 提取与可视化，不计算 MTF/SFR。

## 功能

- 多尺度检测单个或多个 SFRreg 注册标
- 亚像素拟合水平、垂直交叉边并精修中心
- 针对强鱼眼畸变提供中心带回退拟合
- 检查每个 ROI 的边缘极性、连续性和两侧平台均匀性
- 自动修剪触及圆周或不满足连续边缘条件的 ROI
- 同时输出 NumPy 0-based 坐标和 Imatest/MATLAB 1-based 坐标
- 支持 PGM、PNG、TIFF、JPEG 等 Pillow 可读取的静态图像

## 环境要求

- Python 3.10 或更高版本
- NumPy
- OpenCV
- Pillow

安装依赖：

```bash
python -m pip install numpy opencv-python pillow
```

克隆仓库：

```bash
git clone https://github.com/linhangyu-xreal/sfrreg-extract-edges.git
cd sfrreg-extract-edges
```

## 快速开始

```bash
python sfrreg_extract_edges.py path/to/image.pgm
```

默认在输入图像所在目录生成三份结果：

| 文件 | 内容 |
|---|---|
| `<stem>_sfrreg_annotated.png` | 全图标注：四色 ROI、圆、交叉边和局部放大插图 |
| `<stem>_sfrreg_zoom.png` | 第一个注册标的像素级放大图 |
| `<stem>_sfrreg_rois.json` | 图像信息、标记中心、半径、倾斜角和四个 ROI 坐标 |

指定输出位置：

```bash
python sfrreg_extract_edges.py image.tif \
  --output results/annotated.png \
  --zoom results/zoom.png \
  --json results/rois.json
```

Windows PowerShell 示例：

```powershell
python .\sfrreg_extract_edges.py C:\images\sample.pgm `
  --roi-width wider `
  --expected-marks 1
```

## 命令行参数

```text
sfrreg_extract_edges.py IMAGE [选项]
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `IMAGE` | 必填 | 输入图像：PGM / PNG / TIFF / JPEG 等 |
| `-o`, `--output` | `<stem>_sfrreg_annotated.png` | 全图标注路径 |
| `--zoom` | `<stem>_sfrreg_zoom.png` | 放大图路径 |
| `--json` | `<stem>_sfrreg_rois.json` | JSON 路径 |
| `--fill-factor` | `0.6` | Imatest `fillf`；控制沿边方向的 ROI 长度 |
| `--roi-length-pct` | `0.85` | Imatest `roilth`；相对于默认值进一步缩放长度 |
| `--roi-width` | `normal` | `normal` / `wider` / `widest` |
| `--min-diameter-frac` | `0.04` | 最小标记直径 / 图像高度 |
| `--max-diameter-frac` | `0.45` | 最大标记直径 / 图像短边 |
| `--expected-marks` | `0` | 最多保留的标记数；`0` 表示保留全部合格目标 |
| `--max-slant` | `20` | 边缘相对水平/垂直方向允许的最大倾斜角（度） |
| `--min-rim-ratio` | `1.8` | 圆内/圆外环对比度比的最小值 |
| `--peak-threshold-frac` | `0.18` | 鞍点候选阈值相对于全图最大响应的比例 |
| `--quadrant-mismatch-ratio` | `0.65` | 象限极性失配相对于对比度的最大比例 |
| `--center-prior-scale` | 不启用 | 中心先验尺度 / 图像短边；用于降低强外围干扰的排序 |
| `--no-inset` | 关闭 | 不在全图标注中嵌入局部放大图 |

查看当前版本的完整帮助：

```bash
python sfrreg_extract_edges.py --help
```

## 常用调参

只保留质量最高的一个标记，并优先考虑画面中心区域：

```bash
python sfrreg_extract_edges.py image.png \
  --expected-marks 1 \
  --center-prior-scale 0.35
```

检测更小、对比度更弱的标记：

```bash
python sfrreg_extract_edges.py image.png \
  --min-diameter-frac 0.02 \
  --peak-threshold-frac 0.12 \
  --min-rim-ratio 1.5
```

放宽大倾角和象限极性条件：

```bash
python sfrreg_extract_edges.py image.png \
  --max-slant 35 \
  --quadrant-mismatch-ratio 0.75
```

阈值降低会增加误检概率。建议先小幅调整单个参数，再检查标注图和 JSON 结果。

## 坐标约定

JSON 同时给出两套坐标。两者都以图像左上角为原点，x 向右、y 向下。

- `x, y, w, h` 与 `xyxy`：Python / NumPy **0-based、右下开区间**。切片为 `img[y:y+h, x:x+w]`，也等价于 `img[y1:y2, x1:x2]`。
- `imatest_ltrb_1based`：`[L, T, R, B]`，MATLAB / Imatest **1-based、闭区间**，对应 `img(T:B, L:R)`。

终端也会同时打印两套坐标，便于与 Imatest 的 `roi_mult` 对照。

## 算法概要

1. **多尺度鞍点检测**：利用四象限对角差 `|TL + BR - TR - BL|` 产生中心候选。
2. **二次角谐波验证**：在候选中心采样圆环，确认能量集中于每圈两个周期的四象限结构。
3. **半径和圆周验证**：结合圆环灰度标准差衰减、谐波有效范围和圆内外对比度估计标记半径。
4. **象限极性验证**：要求相对象限同极性、相邻象限反极性。
5. **交叉边拟合**：对梯度峰进行亚像素直线拟合；强畸变时改用中心带拟合，避免圆周吸引线拟合。
6. **ROI 生成**：沿 T、B、L、R 四条半边创建轴对齐矩形，避开中心交叉和外圆周。
7. **连续边缘校验**：检查边缘是否贯穿 ROI、极性是否一致以及两侧平台是否均匀；必要时向内修剪 ROI。

标注颜色为 T 红、B 绿、L 蓝、R 橙；圆为青色，交叉边为黄色。

## 检测失败时

- **标记太小**：降低 `--min-diameter-frac`，例如 `0.02`。
- **标记较弱或模糊**：小幅降低 `--peak-threshold-frac` 或 `--min-rim-ratio`。
- **边缘倾角较大**：提高 `--max-slant`，例如 `35`。
- **存在类似四分圆的干扰**：设置 `--expected-marks 1`，并尝试 `--center-prior-scale 0.35`。
- **鱼眼畸变很强**：适当放宽 `--quadrant-mismatch-ratio`；如果圆被压成极扁椭圆，建议先做去畸变或使用预畸变标记。
- **ROI 连续性校验失败**：检查圆周、反光、遮挡或其他结构是否进入矩形；可尝试减小 `--fill-factor` 或 `--roi-length-pct`。
- **过曝或对比度不足**：改善曝光和照明。亮暗差过低时，检测和边缘定位都会不稳定。

## 文件结构

```text
.
├── sfrreg_extract_edges.py  # 主程序
└── README.md                # 使用说明
```

## 说明

该项目以 Imatest SFRreg 的几何和工作流程为参考，用于自动化提取 slanted-edge ROI。它不是 Imatest 官方项目，也不替代 Imatest 的 MTF/SFR 计算与结果判定。
