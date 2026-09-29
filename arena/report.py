"""评测报告：攻防矩阵、排行榜、热力图、CSV/JSON 导出（赛题 §23）。"""
from __future__ import annotations

import csv
import io
import json
import math


# --------------------------------------------------------------------------- #
# 控制台矩阵表格
# --------------------------------------------------------------------------- #
def matrix_table(result) -> str:
    """对齐的 M[a,d] ROC-AUC 矩阵表格（行=攻击，列=防御）。"""
    attacks = result.attack_ids
    defenses = result.defense_ids
    col_w = 9
    head = "Attack\\Defense".ljust(16) + "".join(d[:col_w - 1].ljust(col_w) for d in defenses) + "  AttackMean"
    lines = [head, "-" * len(head)]
    for a in attacks:
        cells = "".join(_fmt_auc(result.matrix[a][d]).rjust(col_w) for d in defenses)
        mean = result.attack_metrics[a]["mean_auc"]
        lines.append(a.ljust(16) + cells + f"  {_fmt_auc(mean).rjust(col_w)}")
    lines.append("-" * len(head))
    cells = "".join(_fmt_auc(result.defense_metrics[d]["mean_auc"]).rjust(col_w) for d in defenses)
    lines.append("DefenseMean".ljust(16) + cells)
    return "\n".join(lines)


def _fmt_auc(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  N/A "
    return f"{v:.4f}"


# --------------------------------------------------------------------------- #
# 热力图（ANSI 256 色背景；攻击视角：低=绿=攻击强，高=红=防御强）
# --------------------------------------------------------------------------- #
def heatmap(result, color: bool = True) -> str:
    attacks = result.attack_ids
    defenses = result.defense_ids
    col_w = 9
    label_w = 16
    lines = ["Heatmap (green=low AUC=attack wins, red=high AUC=defense wins):", ""]
    head = "Attack\\Defense".ljust(label_w) + "".join(d[:col_w - 1].ljust(col_w) for d in defenses)
    lines.append(head)
    for a in attacks:
        cells = "".join(_heat_cell(result.matrix[a][d], col_w, color) for d in defenses)
        lines.append(a.ljust(label_w) + cells)
    return "\n".join(lines)


def _heat_cell(v, width, color):
    txt = _fmt_auc(v)
    if not color or v is None or (isinstance(v, float) and math.isnan(v)):
        return txt.rjust(width)
    # 0.4(绿) -> 0.5(黄) -> 1.0(红)
    t = max(0.0, min(1.0, (v - 0.4) / 0.6))
    r = int(40 + 215 * t)
    g = int(180 - 130 * t)
    b = int(60 + 40 * (1 - t))
    # 先把可见文本右对齐到 width，再包裹 ANSI 背景（ANSI 序列不占可见宽度）
    return f"\033[48;2;{r};{g};{b}m{txt.rjust(width)}\033[0m"


# --------------------------------------------------------------------------- #
# 排行榜
# --------------------------------------------------------------------------- #
def attack_leaderboard(result) -> str:
    lines = ["Attack Leaderboard (mean ROC-AUC ascending; lower = stronger elimination):",
             f"{'rank':<5}{'attack':<20}{'mean_AUC':<11}{'max_AUC':<11}{'strength':<11}{'mean_LPIPS':<12}"]
    for i, a in enumerate(result.attack_ranking, 1):
        m = result.attack_metrics[a]
        lines.append(f"{i:<5}{a:<20}{_fmt_auc(m['mean_auc']):<11}"
                     f"{_fmt_auc(m['max_auc']):<11}{_fmt_auc(m['strength']):<11}"
                     f"{_fmt(m['mean_attack_lpips']):<12}")
    return "\n".join(lines)


def defense_leaderboard(result) -> str:
    lines = ["Defense Leaderboard (mean ROC-AUC descending; higher = more robust):",
             f"{'rank':<5}{'defense':<20}{'mean_AUC':<11}{'min_AUC':<11}{'clean_AUC':<11}{'embed_LPIPS':<12}"]
    for i, d in enumerate(result.defense_ranking, 1):
        m = result.defense_metrics[d]
        lines.append(f"{i:<5}{d:<20}{_fmt_auc(m['mean_auc']):<11}"
                     f"{_fmt_auc(m['min_auc']):<11}{_fmt_auc(m['clean_auc']):<11}"
                     f"{_fmt(m['mean_embed_lpips']):<12}")
    return "\n".join(lines)


def _fmt(v):
    if v is None:
        return "N/A"
    return f"{v:.4f}"


# --------------------------------------------------------------------------- #
# 单组合明细
# --------------------------------------------------------------------------- #
def pair_detail(result, a_id, d_id) -> str:
    pr = result.pair_results[(a_id, d_id)]
    d = pr.as_dict()
    lines = [f"Pair detail: attack={a_id} defense={d_id}",
             f"  ROC-AUC        : {_fmt_auc(d['auc'])}",
             f"  embed success  : {d['embed_success_rate']:.3f}  legal: {d['embed_valid_rate']:.3f}  quality: {d['embed_quality_rate']:.3f}",
             f"  attack success : {d['attack_success_rate']:.3f}  legal: {d['attack_valid_rate']:.3f}  quality: {d['attack_quality_rate']:.3f}",
             f"  detect success : {d['detect_success_rate']:.3f}",
             f"  embed quality  : SSIM={_fmt(d['embed_quality'].get('ssim'))} "
             f"PSNR={_fmt(d['embed_quality'].get('psnr'))} LPIPS={_fmt(d['embed_quality'].get('lpips'))}",
             f"  attack quality : SSIM={_fmt(d['attack_quality'].get('ssim'))} "
             f"PSNR={_fmt(d['attack_quality'].get('psnr'))} LPIPS={_fmt(d['attack_quality'].get('lpips'))}",
             f"  joint quality  : SSIM={_fmt(d['joint_quality'].get('ssim'))} "
             f"PSNR={_fmt(d['joint_quality'].get('psnr'))} LPIPS={_fmt(d['joint_quality'].get('lpips'))}",
             f"  time(s)        : embed={d['time_embed']:.3f} attack={d['time_attack']:.3f} detect={d['time_detect']:.3f}"]
    if d["error"]:
        lines.append(f"  error          : {d['error']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CSV / JSON 导出
# --------------------------------------------------------------------------- #
def to_csv(result) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["attack", "defense", "auc",
                "embed_success_rate", "embed_valid_rate", "embed_quality_rate",
                "attack_success_rate", "attack_valid_rate", "attack_quality_rate",
                "detect_success_rate",
                "embed_ssim", "embed_psnr", "embed_lpips",
                "attack_ssim", "attack_psnr", "attack_lpips",
                "joint_ssim", "joint_psnr", "joint_lpips",
                "time_embed", "time_attack", "time_detect"])
    for a in result.attack_ids:
        for d in result.defense_ids:
            pr = result.pair_results[(a, d)].as_dict()
            w.writerow([a, d, f"{pr['auc']:.6f}",
                        f"{pr['embed_success_rate']:.4f}", f"{pr['embed_valid_rate']:.4f}",
                        f"{pr['embed_quality_rate']:.4f}",
                        f"{pr['attack_success_rate']:.4f}", f"{pr['attack_valid_rate']:.4f}",
                        f"{pr['attack_quality_rate']:.4f}",
                        f"{pr['detect_success_rate']:.4f}",
                        _n(pr['embed_quality'].get('ssim')), _n(pr['embed_quality'].get('psnr')),
                        _n(pr['embed_quality'].get('lpips')),
                        _n(pr['attack_quality'].get('ssim')), _n(pr['attack_quality'].get('psnr')),
                        _n(pr['attack_quality'].get('lpips')),
                        _n(pr['joint_quality'].get('ssim')), _n(pr['joint_quality'].get('psnr')),
                        _n(pr['joint_quality'].get('lpips')),
                        f"{pr['time_embed']:.4f}", f"{pr['time_attack']:.4f}", f"{pr['time_detect']:.4f}"])
    return buf.getvalue()


def to_json(result) -> str:
    return json.dumps({
        "n_samples": result.n_samples,
        "attack_ids": result.attack_ids,
        "defense_ids": result.defense_ids,
        "matrix": result.matrix,
        "pair_results": {f"{a}|{d}": result.pair_results[(a, d)].as_dict()
                         for a in result.attack_ids for d in result.defense_ids},
        "attack_metrics": result.attack_metrics,
        "defense_metrics": result.defense_metrics,
        "attack_ranking": result.attack_ranking,
        "defense_ranking": result.defense_ranking,
    }, indent=2, ensure_ascii=False)


def _n(v):
    return "" if v is None else f"{v:.6f}"


# --------------------------------------------------------------------------- #
# 完整控制台报告
# --------------------------------------------------------------------------- #
def full_report(result, color: bool = True) -> str:
    sections = [
        f"=== WatermarkGuard-Image 攻防矩阵 (N={result.n_samples}) ===",
        matrix_table(result),
        "",
        attack_leaderboard(result),
        "",
        defense_leaderboard(result),
        "",
        heatmap(result, color=color),
    ]
    return "\n".join(sections)
