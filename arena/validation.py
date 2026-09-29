"""图像合法性校验（赛题 §9.1）。

校验嵌入/攻击/检测各环节输出图像是否满足基础合法性：
尺寸/通道/dtype、无 NaN/Inf、取值范围、无大面积纯色覆盖、无异常透明等。
返回 (ok, reason) 元组，便于流水线据此触发 §9.6 替换逻辑。
"""
from __future__ import annotations

import numpy as np


def validate_image(img, expected_shape=(512, 512, 3)) -> tuple[bool, str]:
    """校验单张图像合法性。返回 (ok, reason)。"""
    if img is None:
        return False, "image is None"
    if not isinstance(img, np.ndarray):
        return False, f"image is not ndarray ({type(img).__name__})"
    if img.shape != tuple(expected_shape):
        return False, f"shape {tuple(img.shape)} != {tuple(expected_shape)}"
    if img.dtype != np.uint8:
        return False, f"dtype {img.dtype} != uint8"

    f = img.astype(np.float32)
    if not np.all(np.isfinite(f)):
        return False, "contains NaN/Inf"
    if img.min() < 0 or img.max() > 255:
        return False, f"value range [{int(img.min())},{int(img.max())}] outside [0,255]"

    # 大面积纯色覆盖：整体方差极低 或 单一像素占比过高
    if _near_solid(img):
        return False, "near-solid / large flat area"

    return True, "ok"


def _near_solid(img: np.ndarray) -> bool:
    """检测近乎纯色的图像（攻击方输出空图/纯色图的常见投机）。"""
    # 整体通道标准差极小 -> 近纯色
    per_ch_std = img.reshape(-1, img.shape[-1]).std(axis=0)
    if float(per_ch_std.max()) < 3.0:
        return True
    # 单一像素值占比过高 -> 近纯色
    flat = img.reshape(-1, img.shape[-1])
    # 用降采样加速众数估计
    idx = (flat[::97, 0].astype(np.int32) << 16) | (flat[::97, 1].astype(np.int32) << 8) | flat[::97, 2].astype(np.int32)
    if idx.size == 0:
        return True
    vals, counts = np.unique(idx, return_counts=True)
    if counts.max() / idx.size > 0.99:
        return True
    return False


def validate_output(result, required_keys) -> tuple[bool, str]:
    """校验返回结构是否包含所需键。"""
    if not isinstance(result, dict):
        return False, f"result is not dict ({type(result).__name__})"
    for k in required_keys:
        if k not in result:
            return False, f"missing key '{k}'"
    return True, "ok"


def extract_probability(result) -> tuple[float, bool]:
    """从检测结果提取 watermark_probability。

    返回 (prob, success)。异常/缺失时 prob=0.5（中性分数），success=False。
    """
    if not isinstance(result, dict):
        return 0.5, False
    p = result.get("watermark_probability", None)
    if p is None:
        return 0.5, False
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 0.5, False
    if not np.isfinite(p):
        return 0.5, False
    return float(min(1.0, max(0.0, p))), True
