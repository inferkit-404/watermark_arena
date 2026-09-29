"""主办方公开基线攻击 A0（赛题 §11.2）。

固定组合攻击：
  JPEG quality = 75
  -> 缩放至原尺寸的 90%
  -> 双线性恢复到原尺寸
  -> 轻度 Gaussian Blur, sigma = 0.4

JPEG 步骤需要 Pillow；缺失时降级为"缩放+模糊"（不阻塞，仅削弱攻击）。
全部在内存中处理（BytesIO），不读写图片文件。
"""
from __future__ import annotations

import io
import numpy as np


def attack(sample: dict) -> dict:
    image = np.asarray(sample["image"])
    h, w = image.shape[:2]

    # 1. JPEG q=75（需要 Pillow）
    out = _jpeg(image, quality=75)

    # 2. 缩放至 90% -> 双线性还原
    out = _resize_bilinear(out, int(round(w * 0.9)), int(round(h * 0.9)))
    out = _resize_bilinear(out, w, h)

    # 3. 轻度高斯模糊 sigma=0.4
    out = _gaussian_blur(out, sigma=0.4)
    return {"image": np.clip(out, 0, 255).astype(np.uint8)}


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception:
        return image.astype(np.float32)  # 降级：跳过 JPEG
    img = Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    return np.asarray(out, dtype=np.float32)


def _resize_bilinear(image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """双线性缩放。优先用 Pillow，否则用 numpy 实现。"""
    try:
        from PIL import Image
        img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        out = img.resize((new_w, new_h), Image.BILINEAR)
        return np.asarray(out, dtype=np.float32)
    except Exception:
        return _bilinear_np(image, new_w, new_h)


def _bilinear_np(image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    src = image.astype(np.float32)
    h, w = src.shape[:2]
    ys = np.clip(np.linspace(0, h - 1, new_h), 0, h - 1)
    xs = np.clip(np.linspace(0, w - 1, new_w), 0, w - 1)
    y0, y1 = np.floor(ys).astype(int), np.ceil(ys).astype(int)
    x0, x1 = np.floor(xs).astype(int), np.ceil(xs).astype(int)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[:, None]
    if src.ndim == 3:
        wy = wy[..., None]
        wx = wx[None, ..., None]
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bot = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image.astype(np.float32), sigma=sigma, axes=(0, 1))
    except Exception:
        # 退化：3×3 均值近似
        k = 3
        pad = k // 2
        p = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        out = np.zeros_like(image.astype(np.float32))
        for dy in range(k):
            for dx in range(k):
                out += p[dy:dy + image.shape[0], dx:dx + image.shape[1]]
        return out / (k * k)
