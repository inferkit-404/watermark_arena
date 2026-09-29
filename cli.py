"""WatermarkGuard-Image 攻防对抗评测后端 命令行入口。

用法示例（在 watermark_arena/ 目录下运行）：

  # 全对阵矩阵：基线 + 示例提交
  python cli.py run-matrix \
      --attacks baseline:identity baseline:a0 submissions/attacks/jpeg_attack \
      --defenses baseline:d0 submissions/defenses/null_defense \
      --n 32 --out ./results

  # 单组合
  python cli.py run-pair --attack baseline:a0 --defense baseline:d0 --n 32

  # 资格校验
  python cli.py qualify-attack --attack submissions/attacks/jpeg_attack
  python cli.py qualify-defense --defense submissions/defenses/null_defense

  python cli.py list-baselines
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import yaml

# 支持以 `python cli.py` 或 `python -m cli` 运行：把本文件所在目录加入 sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from arena import data, loader, matrix, report  # noqa: E402
from baselines import attack_a0, defense_d0, identity_attack  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# 解析攻击/防御引用
# --------------------------------------------------------------------------- #
def resolve_attack(ref: str):
    """返回 (attack_id, attack_fn)。"""
    if ref.startswith("baseline:"):
        name = ref.split(":", 1)[1]
        if name in ("a0", "A0"):
            return "A0", attack_a0.attack
        if name in ("identity", "Id", "id"):
            return "Id", identity_attack.attack
        raise ValueError(f"未知基线攻击: {name}")
    mod = loader.load_attack(ref)
    aid = os.path.basename(os.path.abspath(ref.rstrip("/\\")))
    return aid, mod.attack


def resolve_defense(ref: str):
    """返回 (defense_id, embed_fn, detect_fn)。"""
    if ref.startswith("baseline:"):
        name = ref.split(":", 1)[1]
        if name in ("d0", "D0"):
            return "D0", defense_d0.embed, defense_d0.detect
        raise ValueError(f"未知基线防御: {name}")
    mod = loader.load_defense(ref)
    did = os.path.basename(os.path.abspath(ref.rstrip("/\\")))
    return did, mod.embed, mod.detect


def build_attacks(refs):
    out = {}
    for r in refs:
        aid, fn = resolve_attack(r)
        out[aid] = fn
    return out


def build_defenses(refs):
    out = {}
    for r in refs:
        did, e_fn, d_fn = resolve_defense(r)
        out[did] = (e_fn, d_fn)
    return out


def get_images(cfg, args):
    src = args.data_dir or cfg["evaluation"].get("data_source")
    n = args.n or cfg["evaluation"]["n_samples"]
    size = cfg["image"]["width"]
    seed = cfg["evaluation"]["seed"]
    return data.load_dataset(source=src, n=n, size=size, seed=seed)


# --------------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------------- #
def cmd_run_matrix(args):
    cfg = load_config(args.config)
    images = get_images(cfg, args)
    attacks = build_attacks(args.attacks)
    defenses = build_defenses(args.defenses)

    print(f"[run-matrix] attacks={list(attacks)} defenses={list(defenses)} N={len(images)}")

    def progress(i, total, a, d, auc):
        print(f"  [{i}/{total}] {a} x {d} -> AUC={auc:.4f}", flush=True)

    res = matrix.evaluate_matrix(attacks, defenses, images, cfg, progress=progress)
    color = cfg.get("report", {}).get("color", True)
    print("\n" + report.full_report(res, color=color) + "\n")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "matrix.csv"), "w", encoding="utf-8") as f:
            f.write(report.to_csv(res))
        with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
            f.write(report.to_json(res))
        with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
            f.write(report.full_report(res, color=False))
        print(f"[run-matrix] 结果已写入 {args.out}/ (matrix.csv, result.json, report.txt)")


def cmd_run_pair(args):
    cfg = load_config(args.config)
    images = get_images(cfg, args)
    a_id, a_fn = resolve_attack(args.attack)
    d_id, e_fn, d_fn = resolve_defense(args.defense)
    res = matrix.evaluate_matrix({a_id: a_fn}, {d_id: (e_fn, d_fn)}, images, cfg)
    print(report.pair_detail(res, a_id, d_id))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
            f.write(report.to_json(res))


def cmd_qualify_attack(args):
    cfg = load_config(args.config)
    images = get_images(cfg, args)
    a_id, a_fn = resolve_attack(args.attack)
    q = matrix.qualify_attack(a_fn, defense_d0.embed, defense_d0.detect, images, cfg)
    print(f"[qualify-attack] {a_id}")
    print(json.dumps(q, indent=2, ensure_ascii=False, default=_json_default))
    print(f"=> {'PASSED' if q['passed'] else 'FAILED'}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"qualify_attack_{a_id}.json"), "w", encoding="utf-8") as f:
            json.dump(q, f, indent=2, ensure_ascii=False, default=_json_default)


def cmd_qualify_defense(args):
    cfg = load_config(args.config)
    images = get_images(cfg, args)
    d_id, e_fn, d_fn = resolve_defense(args.defense)
    q = matrix.qualify_defense(e_fn, d_fn, attack_a0.attack, images, cfg)
    print(f"[qualify-defense] {d_id}")
    print(json.dumps(q, indent=2, ensure_ascii=False, default=_json_default))
    print(f"=> {'PASSED' if q['passed'] else 'FAILED'}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"qualify_defense_{d_id}.json"), "w", encoding="utf-8") as f:
            json.dump(q, f, indent=2, ensure_ascii=False, default=_json_default)


def cmd_list_baselines(args):
    print("可用基线引用（--attacks / --defenses 参数）：")
    print("  攻击: baseline:identity  恒等攻击（资格对照）")
    print("  攻击: baseline:a0        JPEG75->缩放90%->双线性->高斯模糊(§11.2)")
    print("  防御: baseline:d0        DCT 中频扩频零比特水印(§11.1)")
    print("\n示例提交目录：")
    print("  submissions/attacks/identity_attack")
    print("  submissions/attacks/jpeg_attack")
    print("  submissions/defenses/null_defense")


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(prog="watermark_arena", description="WatermarkGuard-Image 攻防对抗评测后端")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--config", default=os.path.join(_HERE, "config.yaml"))
        sp.add_argument("--n", type=int, default=None, help="采样图像数（覆盖配置）")
        sp.add_argument("--data-dir", default=None, help="图片目录（覆盖配置；默认程序化合成）")
        sp.add_argument("--out", default=None, help="结果输出目录")

    sp = sub.add_parser("run-matrix", help="全对阵矩阵 + 排行榜")
    sp.add_argument("--attacks", nargs="+", required=True)
    sp.add_argument("--defenses", nargs="+", required=True)
    add_common(sp)
    sp.set_defaults(func=cmd_run_matrix)

    sp = sub.add_parser("run-pair", help="单个攻防组合")
    sp.add_argument("--attack", required=True)
    sp.add_argument("--defense", required=True)
    add_common(sp)
    sp.set_defaults(func=cmd_run_pair)

    sp = sub.add_parser("qualify-attack", help="攻击资格校验(§11.3)")
    sp.add_argument("--attack", required=True)
    add_common(sp)
    sp.set_defaults(func=cmd_qualify_attack)

    sp = sub.add_parser("qualify-defense", help="防御资格校验(§11.4)")
    sp.add_argument("--defense", required=True)
    add_common(sp)
    sp.set_defaults(func=cmd_qualify_defense)

    sp = sub.add_parser("list-baselines", help="列出可用基线")
    sp.set_defaults(func=cmd_list_baselines)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Windows 控制台默认 GBK，重配置为 UTF-8 以正确显示中文
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args.func(args)


if __name__ == "__main__":
    main()
