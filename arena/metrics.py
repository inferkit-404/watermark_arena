"""评测指标：PSNR / SSIM / LPIPS(可选) / ROC-AUC + 质量判定。

全部基于 numpy + scipy 实现；LPIPS 在 torch/lpips 可用时启用，否则优雅降级。
ROC-AUC 采用 Mann-Whitney U 秩和法，处理并列秩，且**保留原始方向**——
不对小于 0.5 的结果做 max(AUC, 1-AUC) 修正（遵赛题 §10.4）。
"""
from __future__ import annotations

import numpy as np

try:  # SSIM 用 uniform_filter 做窗口统计
    from scipy.ndimage import uniform_filter
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy 是必需依赖，此处仅做防御
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# PSNR
# --------------------------------------------------------------------------- #
def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """峰值信噪比（dB）。完全一致时返回 +inf。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return -float("inf")
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


# --------------------------------------------------------------------------- #
# SSIM
# --------------------------------------------------------------------------- #
def _ssim_single(a: np.ndarray, b: np.ndarray, data_range: float, win: int) -> float:
    """单通道 SSIM。"""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_a = uniform_filter(a, size=win)
    mu_b = uniform_filter(b, size=win)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = uniform_filter(a * a, size=win) - mu_a2
    sigma_b2 = uniform_filter(b * b, size=win) - mu_b2
    sigma_ab = uniform_filter(a * b, size=win) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float(np.mean(num / den))


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0, win: int = 7) -> float:
    """结构相似性。多通道取平均。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return -1.0
    if not _HAS_SCIPY:
        # 退化实现：用均值方差近似（仅当 scipy 缺失，正常不会走到这里）
        return float(1.0 - min(1.0, np.mean((a - b) ** 2) / (data_range ** 2)))
    if a.ndim == 2:
        return _ssim_single(a, b, data_range, win)
    return float(np.mean([_ssim_single(a[..., c], b[..., c], data_range, win)
                          for c in range(a.shape[2])]))


# --------------------------------------------------------------------------- #
# LPIPS（可选）
# --------------------------------------------------------------------------- #
_LPIPS_NET = None
_LPIPS_AVAILABLE: bool | None = None


def lpips_available() -> bool:
    """探测 LPIPS 是否可用（仅探测一次）。"""
    global _LPIPS_AVAILABLE
    if _LPIPS_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            import lpips  # noqa: F401
            _LPIPS_AVAILABLE = True
        except Exception:
            _LPIPS_AVAILABLE = False
    return _LPIPS_AVAILABLE


def lpips_score(a: np.ndarray, b: np.ndarray) -> float | None:
    """感知距离，越大越不相似。torch/lpips 不可用或出错时返回 None。"""
    if not lpips_available():
        return None
    try:
        import torch
        global _LPIPS_NET
        if _LPIPS_NET is None:
            import lpips
            _LPIPS_NET = lpips.LPIPS(net="alex", verbose=False)
            _LPIPS_NET.eval()

        def prep(x: np.ndarray) -> "torch.Tensor":
            x = x.astype(np.float32) / 127.5 - 1.0  # -> [-1, 1]
            return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            d = _LPIPS_NET(prep(a), prep(b))
        return float(d.item())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# ROC-AUC（Mann-Whitney U，并列秩取平均，保留原始方向）
# --------------------------------------------------------------------------- #
def roc_auc(y_true, y_score) -> float:
    """二分类 ROC-AUC。

    - 完美分离（正样本分数全高于负样本）= 1.0
    - 完全反转 = 0.0
    - 随机 ≈ 0.5
    - 并列分数按平均秩处理；保留原始方向，不做 max(AUC,1-AUC) 修正。
    只有一类样本时返回 NaN。
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if len(y_true) != len(y_score):
        raise ValueError("y_true / y_score 长度不一致")

    pos_mask = (y_true == 1)
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # 平均秩（处理并列）
    n = len(y_score)
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 秩从 1 开始
        ranks_sorted[i:j + 1] = avg_rank
        i = j + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted

    sum_ranks_pos = ranks[pos_mask].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# --------------------------------------------------------------------------- #
# 质量判定
# --------------------------------------------------------------------------- #
def quality_ok(kind: str, ssim_v: float | None, psnr_v: float | None,
               lpips_v: float | None, cfg: dict) -> bool:
    """按 §9.2(embed)/§9.3(attack) 阈值判定质量是否达标。

    None 值（如 LPIPS 不可用）跳过该项检查，不影响其它项。
    """
    qc = cfg["quality"][kind]
    if ssim_v is not None and ssim_v < qc["ssim_min"]:
        return False
    if psnr_v is not None and psnr_v < qc["psnr_min"]:
        return False
    if lpips_v is not None and lpips_v > qc["lpips_max"]:
        return False
    return True


def mean_quality(records: list) -> dict:
    """对 (ssim, psnr, lpips) 三元组列表求均值，lpips 缺失时取有效项均值。"""
    if not records:
        return {"ssim": None, "psnr": None, "lpips": None, "n": 0}
    ssims = [r[0] for r in records if r[0] is not None]
    psnrs = [r[1] for r in records if r[1] is not None]
    lps = [r[2] for r in records if r[2] is not None]
    return {
        "ssim": float(np.mean(ssims)) if ssims else None,
        "psnr": float(np.mean(psnrs)) if psnrs else None,
        "lpips": float(np.mean(lps)) if lps else None,
        "n": len(records),
    }
