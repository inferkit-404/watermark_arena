"""攻防全对阵矩阵、排行与资格校验（赛题 §11/§12/§13/§14）。

- evaluate_matrix: 笛卡尔积 A_q × D_q，逐对调用 runner.evaluate_pair，形成 M[a,d] 矩阵
- 攻击指标 R_A_a = mean_d M[a,d]（越低越好）；tie-break: max_d M[a,d] 越低越好
- 防御指标 R_D_d = mean_a M[a,d]（越高越好）；tie-break: min_a M[a,d] 越高越好
- qualify_*: §11 第一阶段资格校验（成功率/AUC 门槛，仅筛选不计分）
同一批图像、相同样本数、相同配置用于全部组合（§12）。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from . import runner


@dataclass
class MatrixResult:
    attack_ids: list = field(default_factory=list)
    defense_ids: list = field(default_factory=list)
    matrix: dict = field(default_factory=dict)          # a -> {d -> auc}
    pair_results: dict = field(default_factory=dict)    # (a,d) -> PairResult
    attack_metrics: dict = field(default_factory=dict)  # a -> dict
    defense_metrics: dict = field(default_factory=dict) # d -> dict
    attack_ranking: list = field(default_factory=list)  # 攻击榜（升序）
    defense_ranking: list = field(default_factory=list) # 防御榜（降序）
    n_samples: int = 0

    def auc(self, a, d) -> float:
        return self.matrix[a][d]


# --------------------------------------------------------------------------- #
# 全对阵
# --------------------------------------------------------------------------- #
def evaluate_matrix(attacks: dict, defenses: dict, images: list[dict],
                    cfg: dict, progress=None) -> MatrixResult:
    """attacks: {attack_id: attack_fn}; defenses: {defense_id: (embed_fn, detect_fn)}.

    progress: 可选回调 (i, total, a_id, d_id, auc)，用于 CLI 打印进度。
    """
    res = MatrixResult(attack_ids=list(attacks.keys()),
                       defense_ids=list(defenses.keys()),
                       n_samples=len(images))

    # 预计算每个防御的 Clean AUC（恒等攻击），用于防御榜 tie-break（§14）
    clean_aucs: dict = {}

    total = len(attacks) * len(defenses)
    idx = 0
    for a_id, a_fn in attacks.items():
        res.matrix[a_id] = {}
        for d_id, (e_fn, d_fn) in defenses.items():
            pr = runner.evaluate_pair(a_fn, e_fn, d_fn, images, cfg,
                                      attack_id=a_id, defense_id=d_id)
            res.pair_results[(a_id, d_id)] = pr
            res.matrix[a_id][d_id] = pr.auc
            idx += 1
            if progress is not None:
                progress(idx, total, a_id, d_id, pr.auc)

    # 每个防御的 Clean AUC（单独跑一次恒等攻击）
    for d_id, (e_fn, d_fn) in defenses.items():
        cpr = runner.evaluate_clean(e_fn, d_fn, images, cfg, defense_id=d_id)
        clean_aucs[d_id] = cpr.auc

    res.attack_metrics = _attack_metrics(res, attacks, defenses)
    res.defense_metrics = _defense_metrics(res, attacks, defenses, clean_aucs)
    res.attack_ranking = _rank_attacks(res)
    res.defense_ranking = _rank_defenses(res)
    return res


def _attack_metrics(res: MatrixResult, attacks, defenses) -> dict:
    out = {}
    for a_id in attacks:
        aucs = [res.matrix[a_id][d_id] for d_id in defenses]
        prs = [res.pair_results[(a_id, d_id)] for d_id in defenses]
        out[a_id] = {
            "mean_auc": float(np.mean(aucs)),
            "max_auc": float(np.max(aucs)),   # 最坏防御（越低越好）
            "min_auc": float(np.min(aucs)),
            "strength": float(1.0 - np.mean(aucs)),
            "mean_attack_lpips": _mean_lpips(prs, "attack"),
            "mean_time": float(np.mean([p.time_attack for p in prs])),
            "mean_attack_valid_rate": float(np.mean([p.attack_valid_rate for p in prs])),
        }
    return out


def _defense_metrics(res: MatrixResult, attacks, defenses, clean_aucs) -> dict:
    out = {}
    for d_id in defenses:
        aucs = [res.matrix[a_id][d_id] for a_id in attacks]
        prs = [res.pair_results[(a_id, d_id)] for a_id in attacks]
        out[d_id] = {
            "mean_auc": float(np.mean(aucs)),
            "min_auc": float(np.min(aucs)),   # 最强攻击（越高越好）
            "max_auc": float(np.max(aucs)),
            "clean_auc": float(clean_aucs.get(d_id, float("nan"))),
            "mean_embed_lpips": _mean_lpips(prs, "embed"),
            "mean_time": float(np.mean([p.time_embed + p.time_detect for p in prs])),
            "mean_embed_valid_rate": float(np.mean([p.embed_valid_rate for p in prs])),
        }
    return out


def _mean_lpips(prs: list, key: str):
    vals = [p.as_dict()[f"{key}_quality"].get("lpips") for p in prs]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _rank_attacks(res: MatrixResult) -> list:
    """攻击榜：平均 ROC-AUC 升序；并列时 max_auc 升序（§13）。"""
    return sorted(
        res.attack_ids,
        key=lambda a: (res.attack_metrics[a]["mean_auc"],
                       res.attack_metrics[a]["max_auc"]),
    )


def _rank_defenses(res: MatrixResult) -> list:
    """防御榜：平均 ROC-AUC 降序；并列时 min_auc 降序、clean_auc 降序（§14）。"""
    return sorted(
        res.defense_ids,
        key=lambda d: (-res.defense_metrics[d]["mean_auc"],
                       -res.defense_metrics[d]["min_auc"],
                       -res.defense_metrics[d]["clean_auc"]),
    )


# --------------------------------------------------------------------------- #
# 资格校验（§11）
# --------------------------------------------------------------------------- #
def qualify_attack(attack_fn, baseline_d0_embed, baseline_d0_detect,
                   images: list[dict], cfg: dict) -> dict:
    """攻击资格：在公开基线 D0 上运行，检查成功率/合法率/AUC 下降量（§11.3）。"""
    qcfg = cfg["qualification"]["attack"]

    id_pr = runner.evaluate_pair(runner.identity_attack, baseline_d0_embed,
                                 baseline_d0_detect, images, cfg, "Id", "D0")
    a_pr = runner.evaluate_pair(attack_fn, baseline_d0_embed, baseline_d0_detect,
                                images, cfg, "A", "D0")
    delta = id_pr.auc - a_pr.auc

    # §11.3 攻击资格：成功率 / 合法率(§9.1) / AUC 下降量；质量(§9.3)由裁判在评测时处理，不计入资格门槛
    passed = {
        "success_rate": a_pr.attack_success_rate >= qcfg["success_rate_min"],
        "legal_rate": a_pr.attack_valid_rate >= qcfg["legal_rate_min"],
        "auc_drop": delta >= qcfg["delta_a"],
        "finite_auc": np.isfinite(a_pr.auc),
    }
    return {
        "passed": all(passed.values()),
        "checks": passed,
        "auc_id": id_pr.auc,
        "auc_attack": a_pr.auc,
        "delta": delta,
        "success_rate": a_pr.attack_success_rate,
        "legal_rate": a_pr.attack_valid_rate,
        "quality_rate": a_pr.attack_quality_rate,  # 仅报告
        "detail": a_pr.as_dict(),
    }


def qualify_defense(embed_fn, detect_fn, baseline_a0, images: list[dict],
                    cfg: dict) -> dict:
    """防御资格：无攻击 Clean AUC + 公开基线 A0 攻击后 Robust AUC（§11.4）。"""
    qcfg = cfg["qualification"]["defense"]

    clean_pr = runner.evaluate_clean(embed_fn, detect_fn, images, cfg, "D")
    robust_pr = runner.evaluate_pair(baseline_a0, embed_fn, detect_fn, images, cfg,
                                     "A0", "D")

    passed = {
        "embed_success": clean_pr.embed_success_rate >= qcfg["embed_success_min"],
        "embed_quality_rate": clean_pr.embed_quality_rate >= qcfg["embed_quality_rate_min"],
        "detect_success": clean_pr.detect_success_rate >= qcfg["detect_success_min"],
        "clean_auc": clean_pr.auc >= qcfg["clean_auc_min"],
        "robust_auc": robust_pr.auc >= qcfg["robust_auc_min"],
    }
    return {
        "passed": all(passed.values()),
        "checks": passed,
        "clean_auc": clean_pr.auc,
        "robust_auc": robust_pr.auc,
        "embed_quality": clean_pr.embed_quality,
        "detail_clean": clean_pr.as_dict(),
        "detail_robust": robust_pr.as_dict(),
    }
