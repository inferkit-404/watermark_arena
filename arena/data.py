"""测试图像来源。

支持两种来源（由 config.evaluation.data_source 决定）：
1. data_source = null  -> 程序化合成默认集（开箱即用，全程 numpy，不读盘）
2. data_source = <目录> -> 用 PIL 读取目录内真实图片，统一缩放到 512×512 RGB uint8

合成器覆盖赛题 §5.1 所述多种视觉内容：自然场景/人像/建筑商品/图表文档/
生成式/不同纹理亮度色彩，防止水印算法只适用于单一内容类型。

输出统一为：{"sample_id": str, "image": np.ndarray uint8 H×W×3}
"""
from __future__ import annotations

import os
import glob
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def load_dataset(source=None, n: int = 64, size: int = 512, seed: int = 0) -> list[dict]:
    """加载/生成 n 张 size×size RGB uint8 图像。"""
    if source:
        return _load_from_dir(source, n, size, seed)
    return _synthesize(n, size, seed)


# --------------------------------------------------------------------------- #
# 目录读取
# --------------------------------------------------------------------------- #
def _load_from_dir(directory: str, n: int, size: int, seed: int) -> list[dict]:
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(
            "从目录读取图片需要 Pillow（pip install Pillow）") from e

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")
    files: list[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(directory, e)))
        files.extend(glob.glob(os.path.join(directory, "**", e), recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"目录 {directory} 中未找到图片")

    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    out: list[dict] = []
    for i, fp in enumerate(files[:n]):
        img = Image.open(fp).convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        out.append({"sample_id": f"real_{i:04d}", "image": arr})
    return out


# --------------------------------------------------------------------------- #
# 程序化合成
# --------------------------------------------------------------------------- #
_GENERATORS = []  # 在下面注册


def _synthesize(n: int, size: int, seed: int) -> list[dict]:
    rng = np.random.RandomState(seed)
    gens = list(_GENERATORS)
    out: list[dict] = []
    for i in range(n):
        gen = gens[i % len(gens)]
        img = gen(size, rng)
        img = np.clip(img, 0, 255).astype(np.uint8)
        out.append({"sample_id": f"syn_{i:04d}", "image": img})
    return out


def _register(fn):
    _GENERATORS.append(fn)
    return fn


@_register
def _gen_gradient(size, rng):
    """平滑色彩梯度（自然场景感的低频内容）。"""
    t = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(t, t)
    h = rng.uniform(0, 1)
    r = 0.5 + 0.5 * np.sin(2 * np.pi * (xx + h))
    g = 0.5 + 0.5 * np.sin(2 * np.pi * (yy + 0.33 + h))
    b = 0.5 + 0.5 * np.sin(2 * np.pi * ((xx + yy) * 0.5 + 0.66 + h))
    img = np.stack([r, g, b], axis=-1) * 0.6 + 0.2
    return img * 255


@_register
def _gen_natural(size, rng):
    """高斯滤波噪声 -> 自然纹理感。"""
    noise = rng.randn(size, size, 3)
    sigma = rng.uniform(1.0, 4.0) if _HAS_SCIPY else 0
    if _HAS_SCIPY:
        smooth = gaussian_filter(noise, sigma=sigma)
    else:
        smooth = noise
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    # 随机色调偏移
    tint = rng.uniform(0.7, 1.3, size=3)
    return smooth * tint * 255


@_register
def _gen_sinusoid(size, rng):
    """正弦条纹纹理（高频细节）。"""
    freq = rng.uniform(2, 10)
    phase = rng.uniform(0, 2 * np.pi)
    x = np.linspace(0, freq * np.pi, size)
    xx, _ = np.meshgrid(x, x)
    pat = 0.5 + 0.5 * np.sin(xx + phase)
    base = rng.uniform(0.2, 0.8, size=3)
    img = pat[..., None] * base + (1 - pat[..., None]) * (1 - base) * 0.3
    return img * 255


@_register
def _gen_shapes(size, rng):
    """随机几何形状（建筑/商品感的强边缘）。"""
    bg = rng.randint(40, 200, size=3)
    img = np.full((size, size, 3), bg, dtype=np.float64)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(rng.randint(12, 30)):
        cx, cy = rng.randint(0, size, 2)
        r = rng.randint(15, 90)
        col = rng.randint(0, 255, 3)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        img[mask] = col
    return img


@_register
def _gen_poster(size, rng):
    """海报/图表感：大色块网格 + 细网格线（高频边界但块内平滑，对 JPEG 友好）。"""
    cell = int(rng.choice([32, 64]))
    n = size // cell
    img = np.zeros((size, size, 3), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            img[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = rng.randint(40, 230, 3)
    # 细网格线提供高频细节
    img[::cell, :, :] *= 0.6
    img[:, ::cell, :] *= 0.6
    return np.clip(img, 0, 255)


@_register
def _gen_portrait(size, rng):
    """肤色平滑块（人像感的低纹理区域）。"""
    skin = np.array([rng.randint(190, 230), rng.randint(150, 190), rng.randint(130, 170)], dtype=np.float64)
    bg = np.array([rng.randint(40, 100), rng.randint(40, 100), rng.randint(60, 120)], dtype=np.float64)
    field = rng.rand(size, size)
    if _HAS_SCIPY:
        field = gaussian_filter(field, sigma=rng.uniform(8, 20))
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    img = field[..., None] * skin + (1 - field[..., None]) * bg
    return img


@_register
def _gen_document(size, rng):
    """文档/图表截图感：浅底 + 网格 + 文字状暗块（高频细节）。"""
    img = np.full((size, size, 3), 245, dtype=np.float64)
    step = int(rng.choice([16, 32, 64]))
    img[::step, :, :] = 180
    img[:, ::step, :] = 180
    for _ in range(rng.randint(20, 60)):
        y = rng.randint(0, size - 8)
        x = rng.randint(0, size - 40)
        w = rng.randint(20, 140)
        h = rng.randint(2, 7)
        img[y:y + h, x:x + w, :] = rng.randint(0, 80)
    return img
