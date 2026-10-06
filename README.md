# SFRreg 四边缘自动提取

从图像里自动检测 Imatest **SFRreg** 注册标（圆内两个亮象限、两个暗象限），在四条倾斜半边上各放一个轴对齐 ROI，并把结果画回原图。

ROI 几何对齐 Imatest 23.2.6 SFRreg 默认：`fillf = 0.6`、`roilth = 0.85`、ROI width = Normal。边缘顺序与 Imatest 相同：先垂直 **T、B**，再水平 **L、R**。

本脚本只做检测、ROI 提取和标注，不计算 MTF/SFR。

## 环境

使用已有的 `imaging-tools` conda 环境：

```text
C:\Users\lin\anaconda3\envs\imaging-tools\python.exe
```

依赖：Python 3.10、NumPy、OpenCV、Pillow。

## 快速开始

```powershell
C:\Users\lin\anaconda3\envs\imaging-tools\python.exe `
  C:\Users\lin\Desktop\mtf\scripts\sfrreg_extract_edges.py `
  C:\Users\lin\Desktop\mtf\m0000100.pgm
```

默认把三份结果写到输入图像所在目录：

| 文件 | 内容 |
|------|------|
| `<stem>_sfrreg_annotated.png` | 原图标注：对比度拉伸、四色 ROI、圆与交叉边、角落放大插图 |
| `<stem>_sfrreg_zoom.png` | 第一个注册标的像素级放大 |
| `<stem>_sfrreg_rois.json` | 中心、半径、倾斜角、四个 ROI 坐标 |

`m0000100.pgm` 的实测结果：中心 `(321.1, 176.3)`，半径 `24.3 px`，垂直边约 −8.6°，水平边约 +5.4°。

## 命令行

```text
sfrreg_extract_edges.py IMAGE [选项]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `IMAGE` | （必填） | PGM / PNG / TIFF / JPEG |
| `-o`, `--output` | `<stem>_sfrreg_annotated.png` | 全图标注路径 |
| `--zoom` | `<stem>_sfrreg_zoom.png` | 放大图路径 |
| `--json` | `<stem>_sfrreg_rois.json` | JSON 路径 |
| `--fill-factor` | `0.6` | 对应 Imatest `fillf`。沿边长度 ≈ `fillf × R` |
| `--roi-length-pct` | `0.85` | 对应 Imatest `roilth`。相对 0.85 再缩放长度 |
| `--roi-width` | `normal` | `normal` / `wider` / `widest`（垂直于边的宽度比例 0.85 / 1.15 / 1.50） |
| `--min-diameter-frac` | `0.04` | 最小直径 / 图像高度（Imatest 建议约 4%） |
| `--max-diameter-frac` | `0.45` | 最大直径 / min(宽, 高) |
| `--expected-marks` | `0` | 保留前 N 个注册标；`0` 表示全部通过筛选的目标 |
| `--max-slant` | `20` | 相对水平/垂直的最大倾斜角（度） |
| `--no-inset` | 关 | 全图标注不嵌入放大插图 |

例：只分析更大、更糊的边缘，并指定输出路径：

```powershell
C:\Users\lin\anaconda3\envs\imaging-tools\python.exe `
  C:\Users\lin\Desktop\mtf\scripts\sfrreg_extract_edges.py `
  C:\Users\lin\Desktop\mtf\m0000100.pgm `
  --roi-width wider `
  -o C:\Users\lin\Desktop\mtf\scripts\ann.png
```

## 坐标约定

JSON 里同时给出两套坐标，原点都在图像左上，x 向右、y 向下。

- `x, y, w, h` 与 `xyxy`：Python / NumPy **0-based、右下开区间**。切片为 `img[y:y+h, x:x+w]`，等价于 `img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]`。
- `imatest_ltrb_1based`：`[L, T, R, B]`，MATLAB / Imatest **1-based、闭区间**。对应 `img(T:B, L:R)`。

终端表格两列都打印，便于直接对照 Imatest `roi_mult`。

## 算法概要

1. **多尺度鞍点**：四象限对角差 `|TL+BR − TR+BL|` 在注册标中心取峰值。
2. **2 次角谐波**：在候选中心采圆环，四象限图案的功率集中在每圈 2 周期分量。
3. **半径**：圆环灰度标准差在圆边处跌落；同时要求圆外对比度明显低于圆内（与棋盘格角点区分）。
4. **象限极性**：相对两象限同亮暗、相邻象限相反。
5. **两条交叉边**：亚像素梯度峰拟合近垂直线 `x = ay+b` 与近水平线 `y = ax+b`，交点作为精修中心。
6. **四个 ROI**：沿每条半边，内侧离开中心交叉约 `0.1 R`，长度按 `fillf` / `roilth`，宽度按 `--roi-width`；矩形轴对齐（斜边从框中穿过，这是 slanted-edge SFR 的用法）。框被限制在约 `0.92 R` 以内，避开圆周。

标注颜色：T 红、B 绿、L 蓝、R 橙；青色圆、黄色交叉边。

## 与 Imatest 23.2.6 的对照

对本机 `m0000100.pgm`，Imatest 写入 `imatest-v2.ini` 的 `roi_mult` 为：

```text
319 160 331 174   # T
316 180 328 194   # B
305 172 319 184   # L
325 173 339 185   # R
```

本脚本默认结果（1-based LTRB）：

```text
317 160 329 174   # T  13×15
314 180 326 194   # B  13×15
305 170 319 182   # L  15×13
325 172 339 184   # R  15×13
```

框尺寸相同。竖直 ROI 的 y 范围、水平 ROI 的 x 范围与 Imatest 一致。垂直于边缘的方向大约差 1–2 像素：这里把框中心放在拟合斜边上，Imatest 的框略偏亮侧。

用 Imatest IT 样张 `sfrreg_example.jpg`（9 个注册标）可全部检出。

## 检测失败时

- 注册标太小：降低 `--min-diameter-frac`（例如 `0.02`）。
- 斜边超过约 15°：加大 `--max-slant`（例如 `35`）。
- 场景里有类似四分圆的干扰：设置 `--expected-marks 1`，或先裁掉棋盘、灯管等区域。
- 对比度很低或严重过曝：Imatest 要求亮暗差大约不低于动态范围的 20%。
- 鱼眼把圆压成很扁的椭圆时，圆拟合会变差；可把目标正对相机，或换用预畸变 SFRreg 标。

## 文件

```text
scripts/
  sfrreg_extract_edges.py   # 本工具
  README.md                 # 本说明
```
