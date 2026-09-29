"""单攻防组合评测流水线（赛题 §10）。

对每个 (攻击 a, 防御 d) 组合，从同一批原图构造同源配对、对称攻击：

    正分支: I+ = E_d(I)  ─┐
                          ├──> A_a ──> D_d ──> s+
    负分支: I- = I        ─┘        A_a ──> D_d ──> s-

最后 y_true=[1..1,0..0], y_score=[s+..,s-..]，M[a,d]=ROC-AUC。

非法处理（§9.6）：
  - 嵌入输出非法 -> 正样本用原始无水印图替代（valid_embed=0，使防御 AUC 下降）
  - 攻击输出非法 -> 用攻击前输入进检测（valid_attack=0，使攻击方无收益）
  - 检测异常 -> 分数取中性 0.5（detect_success=0）
攻击/检测均不接收标签；同一攻击同参同时作用于正负分支。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from . import metrics as M
from . import validation as V
from .loader import timed_safe_call


# --------------------------------------------------------------------------- #
# 恒等攻击（用于 §11.3 资格对照 AUC(Id,D0) 与 Clean AUC）
# --------------------------------------------------------------------------- #
def identity_attack(sample: dict) -> dict:
    return {"image": np.asarray(sample["image"]).copy()}


@dataclass
class PairResult:
    attack_id: str
    defense_id: str
    n: int
    auc: float
    # 成功率（不抛异常）
    embed_success_rate: float = 0.0
    attack_success_rate: float = 0.0
    detect_success_rate: float = 0.0
    # 合法率（§9.1：尺寸/通道/dtype/NaN/纯色等；始终为硬门槛）
    embed_valid_rate: float = 0.0
    attack_valid_rate: float = 0.0
    # 质量合格率（§9.2/§9.3：SSIM/PSNR/LPIPS；strict 时质量不达标亦触发替换）
    embed_quality_rate: float = 0.0
    attack_quality_rate: float = 0.0
    # 质量均值
    embed_quality: dict = field(default_factory=dict)
    attack_quality: dict = field(default_factory=dict)
    joint_quality: dict = field(default_factory=dict)  # 攻击后图 vs 原图
    # 耗时（秒）
    time_embed: float = 0.0
    time_attack: float = 0.0
    time_detect: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "attack_id": self.attack_id, "defense_id": self.defense_id, "n": self.n,
            "auc": self.auc,
            "embed_success_rate": self.embed_success_rate,
            "attack_success_rate": self.attack_success_rate,
            "detect_success_rate": self.detect_success_rate,
            "embed_valid_rate": self.embed_valid_rate,
            "attack_valid_rate": self.attack_valid_rate,
            "embed_quality_rate": self.embed_quality_rate,
            "attack_quality_rate": self.attack_quality_rate,
            "embed_quality": self.embed_quality,
            "attack_quality": self.attack_quality,
            "joint_quality": self.joint_quality,
            "time_embed": self.time_embed,
            "time_attack": self.time_attack,
            "time_detect": self.time_detect,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# 单组合评测
# --------------------------------------------------------------------------- #
def evaluate_pair(attack_fn, embed_fn, detect_fn, images: list[dict],
                  cfg: dict, attack_id: str = "A", defense_id: str = "D") -> PairResult:
    """对单个 (攻击, 防御) 组合执行 §10 全流程。"""
    strict = cfg["quality"].get("strict", True)

    y_true: list[int] = []
    y_score: list[float] = []

    # 计数
    n = len(images)
    embed_succ = embed_legal = embed_qok = 0
    atk_succ = atk_legal = atk_qok = 0
    det_succ = 0
    embed_q: list[tuple] = []
    attack_q: list[tuple] = []
    joint_q: list[tuple] = []
    t_embed = t_attack = t_detect = 0.0

    for s in images:
        I = np.asarray(s["image"])
        expected = tuple(I.shape)  # §7.1：输出尺寸/通道/dtype 须与输入一致
        sid = s.get("sample_id", "")

        # ---- 1. 防御嵌入：正分支 ----
        ok, res, dt = timed_safe_call(embed_fn, {"sample_id": sid, "image": I.copy()})
        t_embed += dt
        embed_succ += int(ok)
        I_plus = I.copy()  # 默认回退为原图（无水印）
        if ok:
            legal, qok, img_used = _process_embed(res, I, expected, strict, cfg, embed_q)
            embed_legal += int(legal)
            embed_qok += int(qok)
            if img_used is not None:
                I_plus = img_used  # 合法且（质量达标或非 strict）时用嵌入图
        # 负分支 = 原始无水印图
        I_minus = I.copy()

        # ---- 2. 对称攻击：同一攻击同参作用于正负分支，不接收标签 ----
        ok_p, res_p, dt_p = timed_safe_call(attack_fn, {"sample_id": sid, "image": I_plus.copy()})
        ok_n, res_n, dt_n = timed_safe_call(attack_fn, {"sample_id": sid, "image": I_minus.copy()})
        t_attack += dt_p + dt_n
        atk_succ += int(ok_p and ok_n)

        legal_p, qok_p, atk_plus = _process_attack(res_p, ok_p, I_plus, expected, strict, cfg, attack_q)
        legal_n, qok_n, atk_minus = _process_attack(res_n, ok_n, I_minus, expected, strict, cfg, None)
        atk_legal += int(legal_p) + int(legal_n)
        atk_qok += int(qok_p) + int(qok_n)  # 正负分支各计一次，按 2n 归一

        # ---- 3. 防御检测 ----
        ok_dp, res_dp, dt_d = timed_safe_call(detect_fn, {"image": atk_plus.copy()})
        ok_dn, res_dn, _ = timed_safe_call(detect_fn, {"image": atk_minus.copy()})
        t_detect += dt_d
        sp, ssp = V.extract_probability(res_dp if ok_dp else None)
        sn, ssn = V.extract_probability(res_dn if ok_dn else None)
        det_succ += int(ssp and ssn)

        y_true.extend([1, 0])
        y_score.extend([sp, sn])

        # 联合质量（攻击后正样本 vs 原图），仅报告
        joint_q.append((M.ssim(I, atk_plus), M.psnr(I, atk_plus), M.lpips_score(I, atk_plus)))

    auc = M.roc_auc(y_true, y_score)

    return PairResult(
        attack_id=attack_id, defense_id=defense_id, n=n, auc=auc,
        embed_success_rate=embed_succ / n if n else 0.0,
        attack_success_rate=atk_succ / n if n else 0.0,
        detect_success_rate=det_succ / n if n else 0.0,
        embed_valid_rate=embed_legal / n if n else 0.0,
        attack_valid_rate=atk_legal / (2 * n) if n else 0.0,
        embed_quality_rate=embed_qok / n if n else 0.0,
        attack_quality_rate=atk_qok / (2 * n) if n else 0.0,
        embed_quality=M.mean_quality(embed_q),
        attack_quality=M.mean_quality(attack_q),
        joint_quality=M.mean_quality(joint_q),
        time_embed=t_embed, time_attack=t_attack, time_detect=t_detect,
    )


def _process_embed(res, pre_image, expected, strict, cfg, quality_log):
    """校验嵌入输出。返回 (legal, quality_ok, image_to_use)。

    image_to_use：合法且（质量达标或非 strict）时为嵌入图；否则 None（调用方回退为原图，§9.6）。
    """
    pre = np.asarray(pre_image)
    vok, _ = V.validate_output(res, ["image"])
    if not vok:
        return False, False, None
    out = np.asarray(res["image"])
    vok2, _ = V.validate_image(out, expected)
    if not vok2:
        return False, False, None
    ssim_v = M.ssim(pre, out)
    psnr_v = M.psnr(pre, out)
    lp_v = M.lpips_score(pre, out)
    if quality_log is not None:
        quality_log.append((ssim_v, psnr_v, lp_v))
    qok = M.quality_ok("embed", ssim_v, psnr_v, lp_v, cfg)
    if qok or not strict:
        return True, qok, out
    return True, qok, None  # 合法但质量不达标 + strict -> 回退原图


def _process_attack(res, ok, pre_image, expected, strict, cfg, quality_log):
    """校验攻击分支输出。返回 (legal, quality_ok, image_to_use)。

    image_to_use：合法且（质量达标或非 strict）时为输出图；否则回退为攻击前输入（§9.6）。
    quality_log 非 None 时记录该分支质量（仅正分支记录，作为代表）。
    """
    pre = np.asarray(pre_image)
    if not ok:
        return False, False, pre.copy()
    vok, _ = V.validate_output(res, ["image"])
    if not vok:
        return False, False, pre.copy()
    out = np.asarray(res["image"])
    vok2, _ = V.validate_image(out, expected)
    if not vok2:
        return False, False, pre.copy()
    ssim_v = M.ssim(pre, out)
    psnr_v = M.psnr(pre, out)
    lp_v = M.lpips_score(pre, out)
    if quality_log is not None:
        quality_log.append((ssim_v, psnr_v, lp_v))
    qok = M.quality_ok("attack", ssim_v, psnr_v, lp_v, cfg)
    if qok or not strict:
        return True, qok, out
    return True, qok, pre.copy()  # 合法但质量不达标 + strict -> 回退攻击前图


# --------------------------------------------------------------------------- #
# 无攻击 Clean AUC（§11.4）
# --------------------------------------------------------------------------- #
def evaluate_clean(embed_fn, detect_fn, images: list[dict], cfg: dict,
                   defense_id: str = "D") -> PairResult:
    """恒等攻击下的 AUC，即 Clean AUC。"""
    return evaluate_pair(identity_attack, embed_fn, detect_fn, images, cfg,
                         attack_id="Id", defense_id=defense_id)
