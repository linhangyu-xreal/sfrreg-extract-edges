#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatically extract the four SFRreg slanted-edge ROIs (Imatest 23.2 style).

Detects Imatest SFRreg registration marks (circles with two light and two dark
quadrants), places one axis-aligned ROI on each of the four half-edges, and
annotates the result on the original image.

ROI geometry follows Imatest 23.2 SFRreg defaults:
  fill factor 0.6, ROI length 85%, Normal width, vertical edges first (T, B),
  then horizontal (L, R). Coordinates can be printed in Imatest 1-based LTRB.

Example:
  C:\\Users\\lin\\anaconda3\\envs\\imaging-tools\\python.exe ^
      sfrreg_extract_edges.py C:\\Users\\lin\\Desktop\\mtf\\m0000100.pgm
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Imatest 23.2.6 SFRreg defaults (imatest-v2.ini [sfrreg]).
DEFAULT_FILL_FACTOR = 0.60
DEFAULT_ROI_LENGTH_PCT = 0.85
DEFAULT_WIDTH_RATIO = {  # perpendicular / along-edge, by Imatest "ROI width"
    "normal": 0.85,
    "wider": 1.15,
    "widest": 1.50,
}
MIN_ROI = 8
INNER_GAP_FRAC = 0.10  # keep the 4-way intersection out of the ROI
RIM_KEEP_FRAC = 0.92  # keep ROI inside this fraction of the mark radius


def load_gray(path: Path) -> np.ndarray:
    """Load an 8-bit or 16-bit still as float64 grayscale."""
    with Image.open(path) as im:
        if im.mode not in ("L", "I;16", "I", "F", "RGB", "RGBA"):
            im = im.convert("L")
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float64) @ (0.2126, 0.7152, 0.0722)
    else:
        arr = arr.astype(np.float64)
        if arr.max() > 255.0:
            arr = arr * (255.0 / arr.max())
    return arr


def bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample img at (xs, ys). Out-of-bounds samples are NaN."""
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    dx = xs - x0
    dy = ys - y0
    valid = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    out = np.full(xs.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return out
    x0v, y0v = x0[valid], y0[valid]
    dxv, dyv = dx[valid], dy[valid]
    out[valid] = (
        img[y0v, x0v] * (1 - dxv) * (1 - dyv)
        + img[y0v, x0v + 1] * dxv * (1 - dyv)
        + img[y0v + 1, x0v] * (1 - dxv) * dyv
        + img[y0v + 1, x0v + 1] * dxv * dyv
    )
    return out


def sample_ring(img: np.ndarray, cx: float, cy: float, r: float, n: int = 180) -> np.ndarray | None:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    vals = bilinear(img, cx + r * np.cos(theta), cy + r * np.sin(theta))
    if np.isnan(vals).mean() > 0.2:
        return None
    return vals[np.isfinite(vals)]


def harmonic2_score(vals: np.ndarray) -> tuple[float, float, float]:
    """Return (power-ratio of 2nd angular harmonic, phase/2 in radians, std)."""
    if vals is None or len(vals) < 16:
        return 0.0, 0.0, 0.0
    v = vals - vals.mean()
    spec = np.fft.rfft(v)
    power = np.abs(spec[1:]) ** 2
    total = float(power.sum()) + 1e-12
    ratio = float(power[1] / total)  # spec[2] -> power index 1
    angle = 0.5 * float(np.angle(spec[2]))
    return ratio, angle, float(vals.std())


def integral_image(img: np.ndarray) -> np.ndarray:
    return cv2.integral(img.astype(np.float32), sdepth=cv2.CV_64F)


def box_mean(ii: np.ndarray, x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
    """Means of [x0, x1) x [y0, y1) from a (H+1, W+1) integral image."""
    area = np.maximum((x1 - x0) * (y1 - y0), 1)
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return s / area


def saddle_map(img: np.ndarray, scales: list[int]) -> np.ndarray:
    """Multi-scale checkerboard/saddle response (axis-aligned 4-quadrant).

    At the centre of an SFRreg mark, opposite quadrants share polarity and
    adjacent quadrants do not, so |TL+BR-TR-BL| peaks.
    """
    h, w = img.shape
    ii = integral_image(img)
    resp = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for s in scales:
        if s < 3:
            continue
        # Need a margin of s pixels so every quadrant box is inside the image.
        valid = (xx >= s) & (yy >= s) & (xx < w - s) & (yy < h - s)
        if not np.any(valid):
            continue
        x = xx[valid]
        y = yy[valid]
        tl = box_mean(ii, x - s, y - s, x, y)
        tr = box_mean(ii, x, y - s, x + s, y)
        bl = box_mean(ii, x - s, y, x, y + s)
        br = box_mean(ii, x, y, x + s, y + s)
        saddle = np.abs((tl + br) - (tr + bl))
        contrast = np.maximum.reduce([tl, tr, bl, br]) - np.minimum.reduce([tl, tr, bl, br])
        local = np.zeros((h, w), dtype=np.float64)
        local[valid] = saddle * np.clip(contrast / 20.0, 0.25, 4.0)
        resp = np.maximum(resp, local)
    return resp


def nms_peaks(resp: np.ndarray, min_dist: int, threshold: float) -> list[tuple[float, int, int]]:
    """Greedy NMS on 3x3 local maxima (float32), then suppress by min_dist."""
    if resp.size == 0 or resp.max() <= 0:
        return []
    r32 = resp.astype(np.float32)
    local = cv2.dilate(r32, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero((r32 >= local) & (resp >= threshold))
    if xs.size == 0:
        return []
    scores = resp[ys, xs]
    order = np.argsort(-scores)
    taken = np.zeros(xs.size, dtype=bool)
    kept: list[tuple[float, int, int]] = []
    min_dist2 = float(min_dist) ** 2
    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)
    for i in order:
        if taken[i]:
            continue
        x, y = int(xs[i]), int(ys[i])
        kept.append((float(scores[i]), x, y))
        taken |= (xs_f - x) ** 2 + (ys_f - y) ** 2 < min_dist2
        if len(kept) >= 40:
            break
    return kept


def refine_center(img: np.ndarray, cx: float, cy: float, r_hint: float) -> tuple[float, float, float, float]:
    """Two-stage grid search of the 2nd-harmonic score around (cx, cy)."""
    r = max(6.0, 0.45 * r_hint)

    def score_at(x: float, y: float) -> tuple[float, float]:
        vals = sample_ring(img, x, y, r, n=120)
        ratio, ang, std = harmonic2_score(vals)
        return ratio * min(std / 15.0, 3.0), ang

    best = (-1.0, cx, cy, 0.0)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            score, ang = score_at(cx + dx, cy + dy)
            if score > best[0]:
                best = (score, cx + dx, cy + dy, ang)
    cx, cy = best[1], best[2]
    for dy in np.arange(-1.0, 1.01, 0.25):
        for dx in np.arange(-1.0, 1.01, 0.25):
            score, ang = score_at(cx + dx, cy + dy)
            if score > best[0]:
                best = (score, cx + dx, cy + dy, ang)
    return best[1], best[2], best[0], best[3]


def quadrant_polarity(img: np.ndarray, cx: float, cy: float, radius: float, edge_angle: float) -> tuple[float, float]:
    """Return (checkerboard contrast, opposite-quadrant mismatch).

    A registration mark has two dark and two light opposite quadrants. Sample a
    small disk in each quadrant, slightly inside the circle.
    """
    r = 0.50 * radius
    means = []
    h, w = img.shape
    yy, xx = np.mgrid[-2:3, -2:3]
    for k in range(4):
        a = edge_angle + np.pi / 4.0 + k * np.pi / 2.0
        x = cx + r * np.cos(a) + xx
        y = cy + r * np.sin(a) + yy
        vals = bilinear(img, x.ravel(), y.ravel())
        finite = vals[np.isfinite(vals)]
        if finite.size < 6:
            return 0.0, 1e9
        means.append(float(finite.mean()))
    m = np.asarray(means)
    contrast = abs(0.5 * (m[0] + m[2]) - 0.5 * (m[1] + m[3]))
    mismatch = abs(m[0] - m[2]) + abs(m[1] - m[3])
    return contrast, mismatch


def rim_ratio(img: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """Inner-ring std / outer-ring std. Large when the 4-quadrant pattern is circular."""
    inner = sample_ring(img, cx, cy, 0.70 * radius, n=180)
    outer = sample_ring(img, cx, cy, min(radius * 1.22, radius + 8.0), n=180)
    if inner is None or outer is None or inner.std() < 4.0:
        return 0.0
    return float(inner.std() / (outer.std() + 1.0))


def n_zero_crossings(vals: np.ndarray) -> int:
    if vals is None or len(vals) < 16:
        return 0
    # Light circular smoothing so a single noisy sample does not split one edge.
    k = 5
    sm = vals.copy()
    for i in range(1, k // 2 + 1):
        sm = sm + np.roll(vals, i) + np.roll(vals, -i)
    sm = sm / k
    d = sm - sm.mean()
    s = np.signbit(d)
    return int(np.count_nonzero(s != np.roll(s, 1)))


def ring_std_profile(img: np.ndarray, cx: float, cy: float, r_max: float) -> tuple[np.ndarray, np.ndarray]:
    radii = np.arange(4.0, max(8.0, r_max) + 0.01, 1.0)
    stds = []
    for r in radii:
        vals = sample_ring(img, cx, cy, r, n=180)
        stds.append(np.nan if vals is None else float(vals.std()))
    return radii, np.array(stds, dtype=np.float64)


def radius_from_contrast_drop(radii: np.ndarray, stds: np.ndarray) -> float | None:
    """Radius where ring contrast falls to 50% of the inner plateau (the circle rim)."""
    finite = np.isfinite(stds)
    if finite.sum() < 6:
        return None
    # Inner plateau: radii that still sit on the four quadrants.
    inner = finite & (radii >= 6) & (radii <= max(10, 0.6 * radii[finite][-1]))
    if inner.sum() < 3:
        inner = finite
    plateau = float(np.median(stds[inner]))
    if plateau < 5.0:
        return None
    thr = 0.50 * plateau
    # First crossing after the plateau, while std is still falling.
    start = int(np.argmax(radii >= 8))
    for i in range(start, len(radii) - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        if stds[i] >= thr > stds[i + 1]:
            t = (stds[i] - thr) / (stds[i] - stds[i + 1] + 1e-9)
            return float(radii[i] + t * (radii[i + 1] - radii[i]))
        if stds[i] < thr and radii[i] >= 10:
            return float(radii[i])
    return None


def harmonic_ok_until(img: np.ndarray, cx: float, cy: float, r_guess: float) -> float:
    """Largest radius whose 2nd-harmonic power ratio stays high."""
    last = 0.0
    for r in np.arange(6.0, r_guess * 1.4 + 4.0, 1.0):
        vals = sample_ring(img, cx, cy, r, n=180)
        ratio, _, std = harmonic2_score(vals)
        if ratio >= 0.25 and std >= 8.0:
            last = r
        elif last > 0 and r > last + 2:
            break
    return last


def fit_edge_lines(img: np.ndarray, cx: float, cy: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit the two crossing slanted edges as x=ay+b (vertical-ish) and y=ax+b (horizontal-ish)."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    r_in, r_out = 4.0, max(8.0, 0.85 * radius)

    v_rows, v_x = [], []
    y0 = int(np.floor(cy - r_out))
    y1 = int(np.ceil(cy + r_out))
    x_lo = int(np.floor(cx - r_out))
    x_hi = int(np.ceil(cx + r_out))
    h, w = img.shape
    for y in range(max(1, y0), min(h - 1, y1 + 1)):
        dy = y - cy
        # Skip the very centre where four edges meet.
        if abs(dy) < 3:
            continue
        xa = max(1, x_lo)
        xb = min(w - 1, x_hi)
        g = np.abs(gx[y, xa:xb])
        if g.size < 6:
            continue
        j = int(np.argmax(g))
        if g[j] < 8:
            continue
        x = xa + j
        if x <= xa + 1 or x >= xb - 2:
            continue
        # Parabolic sub-pixel peak.
        gm, g0, gp = gx[y, x - 1], gx[y, x], gx[y, x + 1]
        # Use magnitude samples around the peak.
        mag = np.abs([gm, g0, gp])
        denom = mag[0] - 2 * mag[1] + mag[2]
        off = 0.0 if abs(denom) < 1e-9 else np.clip(0.5 * (mag[0] - mag[2]) / denom, -0.5, 0.5)
        xs = x + off
        if (xs - cx) ** 2 + dy ** 2 < r_in ** 2:
            continue
        if (xs - cx) ** 2 + dy ** 2 > r_out ** 2:
            continue
        v_rows.append(y)
        v_x.append(xs)

    h_cols, h_y = [], []
    for x in range(max(1, x_lo), min(w - 1, x_hi + 1)):
        dx = x - cx
        if abs(dx) < 3:
            continue
        ya = max(1, y0)
        yb = min(h - 1, y1)
        g = np.abs(gy[ya:yb, x])
        if g.size < 6:
            continue
        j = int(np.argmax(g))
        if g[j] < 8:
            continue
        y = ya + j
        if y <= ya + 1 or y >= yb - 2:
            continue
        mag = np.abs([gy[y - 1, x], gy[y, x], gy[y + 1, x]])
        denom = mag[0] - 2 * mag[1] + mag[2]
        off = 0.0 if abs(denom) < 1e-9 else np.clip(0.5 * (mag[0] - mag[2]) / denom, -0.5, 0.5)
        ys = y + off
        if dx ** 2 + (ys - cy) ** 2 < r_in ** 2:
            continue
        if dx ** 2 + (ys - cy) ** 2 > r_out ** 2:
            continue
        h_cols.append(x)
        h_y.append(ys)

    if len(v_rows) < 8 or len(h_cols) < 8:
        raise RuntimeError("could not localise the two crossing edges")

    # x = a*y + b
    vert = np.polyfit(np.asarray(v_rows, dtype=np.float64), np.asarray(v_x, dtype=np.float64), 1)
    # y = a*x + b
    horz = np.polyfit(np.asarray(h_cols, dtype=np.float64), np.asarray(h_y, dtype=np.float64), 1)
    return vert, horz


def fit_centered_edge_lines(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the crossing edges in the central band of a validated mark.

    Strong fisheye distortion can make the circular rim sharper than an
    internal half-edge.  Restricting this fallback fit to the middle of the
    mark prevents the rim from attracting the line fit.
    """
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    h, w = img.shape
    span = max(8, int(round(0.50 * radius)))
    band = max(6, int(round(0.42 * radius)))

    def subpixel_peak(values: np.ndarray, index: int) -> float:
        if index <= 0 or index >= len(values) - 1:
            return 0.0
        left, middle, right = values[index - 1 : index + 2]
        denominator = left - 2.0 * middle + right
        if abs(denominator) < 1e-9:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))

    vertical_rows: list[float] = []
    vertical_x: list[float] = []
    for row in range(max(1, int(cy) - span), min(h - 1, int(cy) + span + 1)):
        if abs(row - cy) < 3:
            continue
        start = max(1, int(math.floor(cx - band)))
        stop = min(w - 1, int(math.ceil(cx + band)) + 1)
        magnitudes = np.abs(gx[row, start:stop])
        if magnitudes.size < 6:
            continue
        index = int(np.argmax(magnitudes))
        if magnitudes[index] < 8:
            continue
        vertical_rows.append(float(row))
        vertical_x.append(start + index + subpixel_peak(magnitudes, index))

    horizontal_cols: list[float] = []
    horizontal_y: list[float] = []
    for column in range(max(1, int(cx) - span), min(w - 1, int(cx) + span + 1)):
        if abs(column - cx) < 3:
            continue
        start = max(1, int(math.floor(cy - band)))
        stop = min(h - 1, int(math.ceil(cy + band)) + 1)
        magnitudes = np.abs(gy[start:stop, column])
        if magnitudes.size < 6:
            continue
        index = int(np.argmax(magnitudes))
        if magnitudes[index] < 8:
            continue
        horizontal_cols.append(float(column))
        horizontal_y.append(start + index + subpixel_peak(magnitudes, index))

    def robust_fit(independent: list[float], dependent: list[float]) -> np.ndarray:
        if len(independent) < 8:
            raise RuntimeError("could not localise a centred crossing edge")
        x = np.asarray(independent, dtype=np.float64)
        y = np.asarray(dependent, dtype=np.float64)
        coefficients = np.polyfit(x, y, 1)
        residuals = y - np.polyval(coefficients, x)
        median_residual = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median_residual)))
        cutoff = max(1.5, 3.0 * 1.4826 * mad)
        keep = np.abs(residuals - median_residual) <= cutoff
        if int(keep.sum()) >= 8:
            coefficients = np.polyfit(x[keep], y[keep], 1)
        return coefficients

    # x = a*y+b and y = a*x+b, matching fit_edge_lines().
    return (
        robust_fit(vertical_rows, vertical_x),
        robust_fit(horizontal_cols, horizontal_y),
    )


def line_intersection(vert: np.ndarray, horz: np.ndarray) -> tuple[float, float]:
    """Intersection of x = av*y + bv and y = ah*x + bh."""
    av, bv = float(vert[0]), float(vert[1])
    ah, bh = float(horz[0]), float(horz[1])
    # x = av*(ah*x + bh) + bv
    denom = 1.0 - av * ah
    if abs(denom) < 1e-9:
        raise RuntimeError("edge lines are parallel")
    x = (av * bh + bv) / denom
    y = ah * x + bh
    return x, y


def edge_x_at_y(vert: np.ndarray, y: np.ndarray | float) -> np.ndarray | float:
    return vert[0] * y + vert[1]


def edge_y_at_x(horz: np.ndarray, x: np.ndarray | float) -> np.ndarray | float:
    return horz[0] * x + horz[1]


def clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int] | None:
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < MIN_ROI or y2 - y1 < MIN_ROI:
        return None
    return x1, y1, x2, y2


def place_four_rois(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    vert: np.ndarray,
    horz: np.ndarray,
    fill_factor: float,
    roi_length_pct: float,
    width_ratio: float,
) -> list[dict]:
    """Place Imatest-style axis-aligned ROIs on the four half-edges.

    Length along the edge is fill_factor * R, scaled so the INI default
    roi_length_pct=0.85 reproduces that length. Width is width_ratio * length
    (Normal ≈ 0.85). The inner end of each ROI sits INNER_GAP_FRAC*R away
    from the mark centre so the four-way junction is excluded.
    """
    h, w = img.shape
    length = max(MIN_ROI, int(round(radius * fill_factor * (roi_length_pct / DEFAULT_ROI_LENGTH_PCT))))
    width = max(MIN_ROI, int(round(length * width_ratio)))
    gap = max(2.0, INNER_GAP_FRAC * radius)

    def v_roi(sign: int, name: str, index: int) -> dict:
        # sign -1 = top (smaller y), +1 = bottom.
        y_inner = cy + sign * gap
        y_outer = y_inner + sign * length
        y1, y2 = (y_outer, y_inner) if sign < 0 else (y_inner, y_outer)
        ys = np.linspace(y1, y2, num=max(5, length))
        xs = edge_x_at_y(vert, ys)
        x_mid = float(np.median(xs))
        # Integer 0-based half-open box.
        yi1 = int(round(y1))
        yi2 = int(round(y2))
        xi1 = int(round(x_mid - width / 2.0))
        xi2 = xi1 + width
        box = clip_box(xi1, yi1, xi2, yi2, w, h)
        if box is None:
            raise RuntimeError(f"{name} ROI is outside the image")
        # Keep the far corners inside the mark.
        box = shrink_to_mark(box, cx, cy, radius * RIM_KEEP_FRAC, axis="v")
        return roi_dict(index, name, "vertical", box, x_mid, 0.5 * (y1 + y2), vert, img)

    def h_roi(sign: int, name: str, index: int) -> dict:
        # sign -1 = left (smaller x), +1 = right.
        x_inner = cx + sign * gap
        x_outer = x_inner + sign * length
        x1, x2 = (x_outer, x_inner) if sign < 0 else (x_inner, x_outer)
        xs = np.linspace(x1, x2, num=max(5, length))
        ys = edge_y_at_x(horz, xs)
        y_mid = float(np.median(ys))
        xi1 = int(round(x1))
        xi2 = int(round(x2))
        yi1 = int(round(y_mid - width / 2.0))
        yi2 = yi1 + width
        box = clip_box(xi1, yi1, xi2, yi2, w, h)
        if box is None:
            raise RuntimeError(f"{name} ROI is outside the image")
        box = shrink_to_mark(box, cx, cy, radius * RIM_KEEP_FRAC, axis="h")
        return roi_dict(index, name, "horizontal", box, 0.5 * (x1 + x2), y_mid, horz, img)

    rois = [
        v_roi(-1, "T", 1),
        v_roi(+1, "B", 2),
        h_roi(-1, "L", 3),
        h_roi(+1, "R", 4),
    ]
    return rois


def shrink_to_mark(
    box: tuple[int, int, int, int],
    cx: float,
    cy: float,
    r_max: float,
    axis: str,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2 - 1, y1), (x1, y2 - 1), (x2 - 1, y2 - 1)]
    if all((x - cx) ** 2 + (y - cy) ** 2 <= r_max ** 2 for x, y in corners):
        return box
    # Pull the outer side in until all corners sit inside the circle.
    for _ in range(16):
        x1, y1, x2, y2 = box
        if axis == "v":
            # Outer side is the one farther from cy.
            if abs(y1 - cy) > abs(y2 - 1 - cy):
                y1 += 1
            else:
                y2 -= 1
        else:
            if abs(x1 - cx) > abs(x2 - 1 - cx):
                x1 += 1
            else:
                x2 -= 1
        if x2 - x1 < MIN_ROI or y2 - y1 < MIN_ROI:
            break
        box = (x1, y1, x2, y2)
        corners = [(x1, y1), (x2 - 1, y1), (x1, y2 - 1), (x2 - 1, y2 - 1)]
        if all((x - cx) ** 2 + (y - cy) ** 2 <= r_max ** 2 for x, y in corners):
            return box
    return box


def roi_dict(
    index: int,
    name: str,
    orientation: str,
    box: tuple[int, int, int, int],
    edge_x: float,
    edge_y: float,
    coeffs: np.ndarray,
    img: np.ndarray,
) -> dict:
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    if orientation == "vertical":
        angle = float(np.degrees(np.arctan(coeffs[0])))  # dx/dy from vertical
    else:
        angle = float(np.degrees(np.arctan(coeffs[0])))  # dy/dx from horizontal
    return {
        "index": index,
        "name": name,
        "orientation": orientation,
        "x": x1,
        "y": y1,
        "w": x2 - x1,
        "h": y2 - y1,
        "xyxy": [x1, y1, x2, y2],
        "imatest_ltrb_1based": [x1 + 1, y1 + 1, x2, y2],
        "edge_center": [float(edge_x), float(edge_y)],
        "slant_deg": angle,
        "mean": float(crop.mean()) if crop.size else None,
        "min": float(crop.min()) if crop.size else None,
        "max": float(crop.max()) if crop.size else None,
    }


def enforce_continuous_edge_support(
    img: np.ndarray,
    mark: dict,
    roi: dict,
    minimum_support: float = 0.90,
    maximum_platform_uniformity: float = 0.13,
) -> dict:
    """Trim a half-edge ROI until the complete rectangle stays in the mark.

    A valid edge must have the same polarity on at least ``minimum_support``
    of its scan lines and must reach both ROI end borders.  The black and
    white platforms on either side of the edge must also remain uniform;
    this rejects scan lines where the curved disk boundary enters an outer
    corner of the rectangular ROI.
    """
    smooth = cv2.GaussianBlur(img.astype(np.float64), (3, 3), 0)
    gradient_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    x1, y1, x2, y2 = (int(value) for value in roi["xyxy"])
    vertical = roi["orientation"] == "vertical"
    coefficients = mark["vert"] if vertical else mark["horz"]
    along = list(range(y1, y2)) if vertical else list(range(x1, x2))
    if roi["name"] in ("T", "L"):
        along.reverse()  # Always inspect from the junction towards the rim.

    evidence: list[tuple[int, float, float, float]] = []
    exclusion = 3.0
    for position in along:
        predicted = float(coefficients[0] * position + coefficients[1])
        if vertical:
            normal = int(round(predicted))
            lo = max(x1, normal - 2)
            hi = min(x2, normal + 3)
            first_stop = min(x2, int(math.floor(predicted - exclusion)) + 1)
            second_start = max(x1, int(math.ceil(predicted + exclusion)))
            first = smooth[position, x1:first_stop]
            second = smooth[position, second_start:x2]
            gradient = gradient_x[position, lo:hi]
        else:
            normal = int(round(predicted))
            lo = max(y1, normal - 2)
            hi = min(y2, normal + 3)
            first_stop = min(y2, int(math.floor(predicted - exclusion)) + 1)
            second_start = max(y1, int(math.ceil(predicted + exclusion)))
            first = smooth[y1:first_stop, position]
            second = smooth[second_start:y2, position]
            gradient = gradient_y[lo:hi, position]
        if first.size < 2 or second.size < 2 or gradient.size == 0:
            evidence.append((position, 0.0, 0.0, float("inf")))
            continue
        signed_contrast = float(second.mean() - first.mean())
        edge_strength = float(np.max(np.abs(gradient)))
        platform_uniformity = max(float(first.std()), float(second.std())) / (
            abs(signed_contrast) + 1e-9
        )
        evidence.append(
            (position, signed_contrast, edge_strength, platform_uniformity)
        )

    usable = [
        (contrast, strength)
        for _, contrast, strength, _ in evidence
        if strength >= 8.0 and abs(contrast) >= 5.0
    ]
    if not usable:
        raise RuntimeError(f"{roi['name']} ROI has no continuous edge evidence")
    polarity = 1.0 if float(np.median([value[0] for value in usable])) >= 0 else -1.0
    reference_contrast = float(np.median([abs(value[0]) for value in usable]))
    minimum_contrast = max(5.0, 0.55 * reference_contrast)
    valid = [
        strength >= 8.0
        and contrast * polarity >= minimum_contrast
        and uniformity <= maximum_platform_uniformity
        for _, contrast, strength, uniformity in evidence
    ]

    minimum_length = 16
    best: tuple[int, int, int] | None = None
    for start in range(0, len(valid) - minimum_length + 1):
        if not valid[start]:
            continue
        for end in range(len(valid) - 1, start + minimum_length - 2, -1):
            if not valid[end]:
                continue
            segment = valid[start : end + 1]
            support_count = sum(segment)
            if support_count / len(segment) < minimum_support:
                continue
            candidate = (len(segment), -start, end)
            if best is None or candidate > best:
                best = candidate
            break
    if best is None:
        raise RuntimeError(
            f"{roi['name']} ROI edge does not reach both endpoints "
            f"({minimum_support:.0%} support required)"
        )

    length, negative_start, end = best
    start = -negative_start
    # Keep one extra scan line between the ROI border and a detected rim so
    # both outer corners stay inside the mark.
    boundary_was_detected = end < len(valid) - 1
    if boundary_was_detected and end - start + 1 > minimum_length:
        safe_end = end - 1
        while safe_end > start + minimum_length - 1 and not valid[safe_end]:
            safe_end -= 1
        if valid[safe_end] and safe_end - start + 1 >= minimum_length:
            end = safe_end
            length = end - start + 1

    selected_positions = [entry[0] for entry in evidence[start : end + 1]]
    along1 = min(selected_positions)
    along2 = max(selected_positions) + 1
    original_box = [x1, y1, x2, y2]
    if vertical:
        y1, y2 = along1, along2
        midpoint = 0.5 * (y1 + y2 - 1)
        edge_x = float(coefficients[0] * midpoint + coefficients[1])
        edge_y = midpoint
    else:
        x1, x2 = along1, along2
        midpoint = 0.5 * (x1 + x2 - 1)
        edge_x = midpoint
        edge_y = float(coefficients[0] * midpoint + coefficients[1])
    crop = img[y1:y2, x1:x2]
    support = sum(valid[start : end + 1]) / length
    maximum_uniformity = max(entry[3] for entry in evidence[start : end + 1])
    roi.update(
        {
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "xyxy": [x1, y1, x2, y2],
            "imatest_ltrb_1based": [x1 + 1, y1 + 1, x2, y2],
            "edge_center": [edge_x, edge_y],
            "mean": float(crop.mean()),
            "min": float(crop.min()),
            "max": float(crop.max()),
            "edge_support": support,
            "edge_platform_uniformity_max": maximum_uniformity,
            "edge_endpoint_evidence": True,
            "edge_support_original_xyxy": original_box,
            "edge_support_trimmed": original_box != [x1, y1, x2, y2],
        }
    )
    return roi


def detect_marks(
    img: np.ndarray,
    min_diameter_frac: float,
    max_diameter_frac: float,
    expected: int,
    max_slant_deg: float = 20.0,
    min_rim_ratio: float = 1.8,
    peak_threshold_frac: float = 0.18,
    quadrant_mismatch_ratio: float = 0.65,
    center_prior_scale: float | None = None,
) -> list[dict]:
    """Detect and validate SFRreg marks.

    The optional thresholds allow a caller to tune candidate generation and
    validation for a capture set without reimplementing the detector.  Set
    ``center_prior_scale`` to a fraction of the shorter image dimension to
    prefer central marks when strong peripheral structures create candidates.
    """
    h, w = img.shape
    min_r = max(8, int(round(min_diameter_frac * h / 2.0)))
    max_r = max(min_r + 4, int(round(max_diameter_frac * min(h, w) / 2.0)))
    scales = sorted(set(max(3, int(round(0.40 * r))) for r in range(min_r, max_r + 1, 2)))
    resp = saddle_map(img, scales)
    thr = max(8.0, peak_threshold_frac * float(resp.max()))
    peaks = nms_peaks(resp, min_dist=max(10, min_r), threshold=thr)
    marks: list[dict] = []
    for score, x, y in peaks:
        try:
            cx, cy, hscore, phase = refine_center(img, float(x), float(y), r_hint=2.0 * min_r)
            radii, stds = ring_std_profile(img, cx, cy, r_max=max_r * 1.2)
            r_drop = radius_from_contrast_drop(radii, stds)
            search_hint = min(
                max_r / 1.4,
                max(2.0 * min_r, r_drop or 0.0),
            )
            r_harm = harmonic_ok_until(img, cx, cy, r_guess=search_hint)
            cands = [v for v in (r_drop, r_harm) if v and min_r <= v <= max_r]
            if not cands:
                continue
            radius = float(np.median(cands))
            # Re-centre at the estimated mid-radius, then re-estimate R.
            cx, cy, hscore, phase = refine_center(img, cx, cy, r_hint=radius)
            radii, stds = ring_std_profile(img, cx, cy, r_max=max_r * 1.2)
            r_drop = radius_from_contrast_drop(radii, stds)
            if r_drop and min_r <= r_drop <= max_r:
                radius = float(r_drop)
            vals = sample_ring(img, cx, cy, 0.50 * radius, n=240)
            ratio, ang, std = harmonic2_score(vals)
            if ratio < 0.35 or std < 8.0:
                continue
            if n_zero_crossings(vals) not in (4, 6):  # 4 ideal; 6 if a sample hits the exact edge
                continue
            contrast, mismatch = quadrant_polarity(img, cx, cy, radius, ang)
            if contrast < 12.0 or mismatch > quadrant_mismatch_ratio * contrast:
                continue
            measured_rim_ratio = rim_ratio(img, cx, cy, radius)
            if measured_rim_ratio < min_rim_ratio:
                continue

            try:
                vert, horz = fit_edge_lines(img, cx, cy, radius)
                ix, iy = line_intersection(vert, horz)
                original_offset = math.hypot(ix - cx, iy - cy)
            except RuntimeError:
                vert, horz = fit_centered_edge_lines(img, cx, cy, radius)
                ix, iy = line_intersection(vert, horz)
                original_offset = float("inf")

            # Prefer the centre-band fit when the full-mark fit has locked on
            # to the outer ellipse rather than an internal crossing edge.
            if original_offset > max(5.0, 0.10 * radius):
                try:
                    centered_vert, centered_horz = fit_centered_edge_lines(
                        img, cx, cy, radius
                    )
                    centered_x, centered_y = line_intersection(
                        centered_vert, centered_horz
                    )
                    centered_offset = math.hypot(
                        centered_x - cx, centered_y - cy
                    )
                    if centered_offset < original_offset:
                        vert, horz = centered_vert, centered_horz
                        ix, iy = centered_x, centered_y
                except RuntimeError:
                    pass
            if (
                abs(ix - cx) < max(5.0, 0.15 * radius)
                and abs(iy - cy) < max(5.0, 0.15 * radius)
            ):
                cx, cy = float(ix), float(iy)
            v_ang = math.degrees(math.atan(float(vert[0])))
            h_ang = math.degrees(math.atan(float(horz[0])))
            # Imatest default favours roughly H/V edges (about 2–15° of slant).
            if abs(v_ang) > max_slant_deg or abs(h_ang) > max_slant_deg:
                continue
            quality = float(ratio) * float(std) * float(contrast) / (1.0 + mismatch)
            selection_score = quality
            if center_prior_scale is not None and center_prior_scale > 0:
                center_distance = math.hypot(cx - w / 2.0, cy - h / 2.0)
                center_scale = center_prior_scale * min(w, h)
                selection_score = quality / (
                    1.0 + (center_distance / center_scale) ** 2
                )
            marks.append(
                {
                    "center": [float(cx), float(cy)],
                    "radius": float(radius),
                    "saddle": float(score),
                    "harmonic2": float(ratio),
                    "ring_std": float(std),
                    "phase_deg": math.degrees(float(ang)),
                    "quadrant_contrast": float(contrast),
                    "rim_ratio": float(measured_rim_ratio),
                    "quality": quality,
                    "selection_score": selection_score,
                    "vert_coeffs": [float(vert[0]), float(vert[1])],
                    "horz_coeffs": [float(horz[0]), float(horz[1])],
                    "vert_slant_deg": v_ang,
                    "horz_slant_deg": h_ang,
                    "vert": vert,
                    "horz": horz,
                }
            )
        except (RuntimeError, ValueError, ArithmeticError):
            continue

    # Keep the better of overlapping detections.
    marks.sort(key=lambda m: m["selection_score"], reverse=True)
    filtered: list[dict] = []
    for m in marks:
        cx, cy = m["center"]
        r = m["radius"]
        overlap = False
        for q in filtered:
            dx = cx - q["center"][0]
            dy = cy - q["center"][1]
            if dx * dx + dy * dy < (0.80 * (r + q["radius"])) ** 2:
                overlap = True
                break
        if not overlap:
            filtered.append(m)
    marks = filtered
    if expected > 0:
        marks = marks[:expected]
    marks.sort(key=lambda m: (m["center"][0] - w / 2.0) ** 2 + (m["center"][1] - h / 2.0) ** 2)
    return marks


def stretch_display(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, (0.5, 99.8))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    vis = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (vis * 255.0).astype(np.uint8)


def try_font(size: int) -> ImageFont.ImageFont:
    for name in ("consola.ttf", "arial.ttf", "calibri.ttf", "C:/Windows/Fonts/consola.ttf",
                 "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyh.ttc"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill, stroke=(0, 0, 0)):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font, stroke_width=1)
    pad = 2
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill, stroke_width=1, stroke_fill=stroke)


ROI_COLORS = {
    "T": (255, 64, 64),
    "B": (64, 220, 64),
    "L": (64, 160, 255),
    "R": (255, 200, 32),
}


def annotate_full(
    gray: np.ndarray,
    marks: list[dict],
    rois_per_mark: list[list[dict]],
) -> Image.Image:
    """Imatest-like: muted image, full-contrast ROIs, coloured rectangles."""
    vis = stretch_display(gray)
    rgb = np.stack([vis, vis, vis], axis=-1).astype(np.float32)
    muted = 0.40 * rgb + 0.10 * 40.0
    out = muted.copy()
    h, w = vis.shape
    for rois in rois_per_mark:
        for roi in rois:
            x, y, ww, hh = roi["x"], roi["y"], roi["w"], roi["h"]
            out[y : y + hh, x : x + ww] = rgb[y : y + hh, x : x + ww]
    canvas = np.clip(out, 0, 255).astype(np.uint8)
    im = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(im)
    font = try_font(14)
    font_s = try_font(12)
    for mark, rois in zip(marks, rois_per_mark):
        cx, cy = mark["center"]
        r = mark["radius"]
        # Ellipse approximating the circle.
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 220, 255), width=1)
        # Two edge lines, clipped to 0.9 R.
        vert, horz = mark["vert"], mark["horz"]
        y0, y1 = cy - 0.9 * r, cy + 0.9 * r
        draw.line([(edge_x_at_y(vert, y0), y0), (edge_x_at_y(vert, y1), y1)], fill=(255, 255, 0), width=1)
        x0, x1 = cx - 0.9 * r, cx + 0.9 * r
        draw.line([(x0, edge_y_at_x(horz, x0)), (x1, edge_y_at_x(horz, x1))], fill=(255, 255, 0), width=1)
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(0, 220, 255))
        box_w = 2 if r >= 40 else 1
        for roi in rois:
            x1, y1, x2, y2 = roi["xyxy"]
            color = ROI_COLORS[roi["name"]]
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=color, width=box_w)
            if r >= 40:
                label = f"{roi['index']} {roi['name']}"
                lx, ly = x1, y1 - 16
                if ly < 0:
                    ly = y2 + 2
                if lx + 36 > w:
                    lx = max(0, x2 - 40)
                draw_label(draw, (int(lx), int(ly)), label, font_s, color)
        if r >= 40:
            draw_label(draw, (int(cx + r + 4), int(cy - 8)), f"R={r:.1f}px", font, (0, 220, 255))
    # Colour key (avoids stacking four labels on a ~50 px mark).
    legend = [("1 T", "T"), ("2 B", "B"), ("3 L", "L"), ("4 R", "R")]
    lx, ly = 8, 8
    for text, name in legend:
        draw_label(draw, (lx, ly), text, font, ROI_COLORS[name])
        ly += 18
    return im


def annotate_zoom(
    gray: np.ndarray,
    mark: dict,
    rois: list[dict],
    scale: int = 8,
) -> Image.Image:
    cx, cy = mark["center"]
    r = mark["radius"]
    margin = int(round(r * 1.6 + 8))
    x0 = max(0, int(np.floor(cx - margin)))
    y0 = max(0, int(np.floor(cy - margin)))
    x1 = min(gray.shape[1], int(np.ceil(cx + margin)))
    y1 = min(gray.shape[0], int(np.ceil(cy + margin)))
    crop = gray[y0:y1, x0:x1]
    vis = stretch_display(crop)
    vis = np.stack([vis, vis, vis], axis=-1)
    vis = cv2.resize(vis, (vis.shape[1] * scale, vis.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    im = Image.fromarray(vis, mode="RGB")
    draw = ImageDraw.Draw(im)
    font = try_font(18)

    def mxy(x, y):
        return (x - x0) * scale, (y - y0) * scale

    cxm, cym = mark["center"]
    rr = mark["radius"]
    box = [*mxy(cxm - rr, cym - rr), *mxy(cxm + rr, cym + rr)]
    draw.ellipse(box, outline=(0, 220, 255), width=2)
    vert, horz = mark["vert"], mark["horz"]
    y_a, y_b = cym - 0.95 * rr, cym + 0.95 * rr
    draw.line([mxy(edge_x_at_y(vert, y_a), y_a), mxy(edge_x_at_y(vert, y_b), y_b)], fill=(255, 255, 0), width=2)
    x_a, x_b = cxm - 0.95 * rr, cxm + 0.95 * rr
    draw.line([mxy(x_a, edge_y_at_x(horz, x_a)), mxy(x_b, edge_y_at_x(horz, x_b))], fill=(255, 255, 0), width=2)
    cc = mxy(cxm, cym)
    draw.ellipse([cc[0] - 3, cc[1] - 3, cc[0] + 3, cc[1] + 3], fill=(0, 220, 255))
    for roi in rois:
        x1r, y1r, x2r, y2r = roi["xyxy"]
        p1 = mxy(x1r, y1r)
        p2 = mxy(x2r, y2r)
        color = ROI_COLORS[roi["name"]]
        draw.rectangle([p1[0], p1[1], p2[0] - 1, p2[1] - 1], outline=color, width=3)
        draw_label(draw, (int(p1[0] + 4), int(p1[1] + 4)), f"{roi['index']} {roi['name']}", font, color)
    return im


def paste_inset(full: Image.Image, zoom: Image.Image, marks: list[dict], pad: int = 8) -> Image.Image:
    """Paste a zoomed callout into the corner that covers the fewest marks."""
    w, h = full.size
    zw, zh = zoom.size
    target = int(min(w, h) * 0.52)
    scale = min(target / zw, target / zh, 1.0)
    if scale < 1.0:
        zoom = zoom.resize((max(1, int(zw * scale)), max(1, int(zh * scale))), Image.Resampling.BILINEAR)
        zw, zh = zoom.size
    corners = [
        (pad, pad),
        (w - zw - pad, pad),
        (pad, h - zh - pad),
        (w - zw - pad, h - zh - pad),
    ]

    def covered(xy: tuple[int, int]) -> int:
        x, y = xy
        n = 0
        for m in marks:
            cx, cy = m["center"]
            if x - 4 <= cx <= x + zw + 4 and y - 4 <= cy <= y + zh + 4:
                n += 1
        return n

    x, y = min(corners, key=lambda c: (covered(c), -(c[0] + c[1])))
    out = full.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([x - 2, y - 2, x + zw + 1, y + zh + 1], outline=(0, 220, 255), width=2)
    out.paste(zoom, (x, y))
    return out


def save_json(path: Path, image_path: Path, gray: np.ndarray, marks: list[dict], rois_per_mark: list[list[dict]], args) -> None:
    payload = {
        "image": str(image_path),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "imatest_like": {
            "fill_factor": args.fill_factor,
            "roi_length_pct": args.roi_length_pct,
            "roi_width": args.roi_width,
            "order": "vertical T,B then horizontal L,R",
            "xyxy_is_0based_half_open": True,
            "imatest_ltrb_1based_is_inclusive": True,
        },
        "marks": [],
    }
    for mark, rois in zip(marks, rois_per_mark):
        payload["marks"].append(
            {
                "center_xy": mark["center"],
                "radius_px": mark["radius"],
                "harmonic2": mark["harmonic2"],
                "vert_slant_deg": mark["vert_slant_deg"],
                "horz_slant_deg": mark["horz_slant_deg"],
                "rois": rois,
            }
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="自动提取 Imatest SFRreg 四分圆的四条倾斜边缘 ROI，并标注在原图上。"
    )
    p.add_argument("image", type=Path, help="输入图像（PGM/PNG/TIFF/JPEG）")
    p.add_argument("-o", "--output", type=Path, default=None, help="标注图路径（默认：<stem>_sfrreg_annotated.png）")
    p.add_argument("--json", dest="json_path", type=Path, default=None, help="ROI JSON 路径")
    p.add_argument("--zoom", type=Path, default=None, help="放大标注图路径")
    p.add_argument("--fill-factor", type=float, default=DEFAULT_FILL_FACTOR, help="Imatest fillf，默认 0.6")
    p.add_argument("--roi-length-pct", type=float, default=DEFAULT_ROI_LENGTH_PCT, help="Imatest roilth，默认 0.85")
    p.add_argument("--roi-width", choices=list(DEFAULT_WIDTH_RATIO), default="normal", help="Imatest ROI width")
    p.add_argument("--min-diameter-frac", type=float, default=0.04, help="最小直径 / 图像高度，Imatest 建议约 4%")
    p.add_argument("--max-diameter-frac", type=float, default=0.45, help="最大直径 / min(宽,高)")
    p.add_argument("--expected-marks", type=int, default=0, help="期望的注册标数量，0 表示全部")
    p.add_argument("--max-slant", type=float, default=20.0, help="相对水平/垂直的最大倾斜角（度），Imatest 默认约 ±15°")
    p.add_argument("--min-rim-ratio", type=float, default=1.8, help="圆内/圆外环对比度比最小值（默认：1.8）")
    p.add_argument("--peak-threshold-frac", type=float, default=0.18, help="鞍点候选阈值/全图最大响应（默认：0.18）")
    p.add_argument("--quadrant-mismatch-ratio", type=float, default=0.65, help="象限极性失配/对比度最大值（默认：0.65）")
    p.add_argument("--center-prior-scale", type=float, default=None, help="中心先验尺度/短边；不指定则只按检测质量排序")
    p.add_argument("--no-inset", action="store_true", help="原图标注不嵌入放大插图")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 2
    gray = load_gray(image_path)
    print(f"image: {image_path.name}  {gray.shape[1]}x{gray.shape[0]}  min={gray.min():.0f} max={gray.max():.0f}")

    marks = detect_marks(
        gray,
        min_diameter_frac=args.min_diameter_frac,
        max_diameter_frac=args.max_diameter_frac,
        expected=args.expected_marks,
        max_slant_deg=args.max_slant,
        min_rim_ratio=args.min_rim_ratio,
        peak_threshold_frac=args.peak_threshold_frac,
        quadrant_mismatch_ratio=args.quadrant_mismatch_ratio,
        center_prior_scale=args.center_prior_scale,
    )
    if not marks:
        print("error: no SFRreg registration mark found", file=sys.stderr)
        return 1

    width_ratio = DEFAULT_WIDTH_RATIO[args.roi_width]
    rois_per_mark: list[list[dict]] = []
    for i, mark in enumerate(marks, 1):
        rois = place_four_rois(
            gray,
            mark["center"][0],
            mark["center"][1],
            mark["radius"],
            mark["vert"],
            mark["horz"],
            fill_factor=args.fill_factor,
            roi_length_pct=args.roi_length_pct,
            width_ratio=width_ratio,
        )
        for roi in rois:
            enforce_continuous_edge_support(gray, mark, roi)
        rois_per_mark.append(rois)
        cx, cy = mark["center"]
        print(
            f"mark {i}: center=({cx:.2f}, {cy:.2f})  R={mark['radius']:.1f}px  "
            f"H2={mark['harmonic2']:.2f}  slant V={mark['vert_slant_deg']:+.1f}° H={mark['horz_slant_deg']:+.1f}°"
        )
        print(f"  {'#':<4}{'edge':<6}{'x':>6}{'y':>6}{'w':>6}{'h':>6}  Imatest LTRB (1-based)  slant")
        for roi in rois:
            L, T, R, B = roi["imatest_ltrb_1based"]
            print(
                f"  {roi['index']:<4}{roi['name']:<6}{roi['x']:6d}{roi['y']:6d}"
                f"{roi['w']:6d}{roi['h']:6d}  [{L:4d} {T:4d} {R:4d} {B:4d}]  {roi['slant_deg']:+.1f}°"
            )

    stem_dir = image_path.parent
    stem = image_path.stem
    out_path = args.output or (stem_dir / f"{stem}_sfrreg_annotated.png")
    zoom_path = args.zoom or (stem_dir / f"{stem}_sfrreg_zoom.png")
    json_path = args.json_path or (stem_dir / f"{stem}_sfrreg_rois.json")

    full = annotate_full(gray, marks, rois_per_mark)
    zoom = annotate_zoom(gray, marks[0], rois_per_mark[0])
    if not args.no_inset:
        full = paste_inset(full, zoom, marks)
    full.save(out_path)
    zoom.save(zoom_path)
    save_json(json_path, image_path, gray, marks, rois_per_mark, args)
    print(f"wrote {out_path}")
    print(f"wrote {zoom_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
