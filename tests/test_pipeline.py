"""端到端 + 单元测试。

运行方式（在 watermark_arena/ 目录下）：
    python -m pytest tests/ -v
或：
    python tests/test_pipeline.py

注意：本测试不读取/查看任何图片文件，只检查数值与文本输出。
"""
import os
import sys
import numpy as np

# 把项目根目录（watermark_arena/）加入 sys.path，便于直接运行
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from arena import metrics as M, validation as V, data, runner, matrix  # noqa: E402
from baselines import attack_a0, defense_d0, identity_attack  # noqa: E402


def _cfg(n=24, strict=True):
    return {
        "image": {"width": 512, "height": 512, "channels": 3, "dtype": "uint8"},
        "evaluation": {"n_samples": n, "seed": 0, "data_source": None, "shuffle_pairs": True},
        "quality": {
            "strict": strict,
            "embed": {"ssim_min": 0.97, "psnr_min": 36.0, "lpips_max": 0.06},
            "attack": {"ssim_min": 0.92, "psnr_min": 28.0, "lpips_max": 0.15},
            "joint": {"ssim_min": 0.85},
        },
        "qualification": {
            "attack": {"success_rate_min": 0.99, "legal_rate_min": 0.95, "delta_a": 0.02},
            "defense": {"embed_success_min": 0.99, "embed_quality_rate_min": 0.95,
                        "detect_success_min": 0.99, "clean_auc_min": 0.90, "robust_auc_min": 0.60},
        },
    }


# --------------------------------------------------------------------------- #
# ROC-AUC 单元测试
# --------------------------------------------------------------------------- #
def test_roc_auc_perfect():
    assert abs(M.roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) - 1.0) < 1e-9


def test_roc_auc_reversed():
    # 完全反转：正样本分数全低于负样本 -> AUC=0（保留方向，不做 max 修正）
    assert abs(M.roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) - 0.0) < 1e-9


def test_roc_auc_ties():
    # 全部并列 -> AUC=0.5
    assert abs(M.roc_auc([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9


def test_roc_auc_known():
    # y_true=[1,0], scores=[0.7,0.3] -> 正样本秩2，负样本秩1 -> AUC=1.0
    assert abs(M.roc_auc([1, 0], [0.7, 0.3]) - 1.0) < 1e-9


def test_psnr_ssim_basic():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = a.copy()
    assert M.psnr(a, b) == float("inf")
    assert M.ssim(a, b) > 0.999
    c = (a + 10).astype(np.uint8)
    assert M.psnr(a, c) > 20


# --------------------------------------------------------------------------- #
# 合法性校验
# --------------------------------------------------------------------------- #
def test_validate_image_ok():
    img = data.load_dataset(n=1, size=128, seed=1)[0]["image"]
    ok, _ = V.validate_image(img, (128, 128, 3))
    assert ok


def test_validate_image_bad_shape():
    ok, _ = V.validate_image(np.zeros((64, 64, 3), dtype=np.uint8), (128, 128, 3))
    assert not ok


def test_validate_image_nan():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[0, 0, 0] = 0
    # uint8 无法直接存 NaN；构造非法范围
    bad = img.astype(np.float32)
    bad[0, 0, 0] = np.nan
    ok, _ = V.validate_image(bad.astype(np.float32), (128, 128, 3))
    assert not ok


def test_validate_image_solid():
    ok, _ = V.validate_image(np.full((128, 128, 3), 7, dtype=np.uint8), (128, 128, 3))
    assert not ok


# --------------------------------------------------------------------------- #
# 数据合成多样性
# --------------------------------------------------------------------------- #
def test_synthetic_diversity():
    imgs = data.load_dataset(n=21, size=128, seed=2)  # 覆盖 7 个生成器各 3 张
    assert len(imgs) == 21
    for s in imgs:
        assert s["image"].shape == (128, 128, 3)
        assert s["image"].dtype == np.uint8
        ok, _ = V.validate_image(s["image"], (128, 128, 3))
        assert ok, f"合成图不合法: {s['sample_id']}"


# --------------------------------------------------------------------------- #
# D0 基线：嵌入质量 + Clean AUC + A0 鲁棒 AUC
# --------------------------------------------------------------------------- #
def test_d0_clean_auc_high():
    cfg = _cfg(n=24)
    imgs = data.load_dataset(n=24, size=128, seed=3)
    pr = runner.evaluate_clean(defense_d0.embed, defense_d0.detect, imgs, cfg, "D0")
    print(f"\n[D0] Clean AUC={pr.auc:.4f} "
          f"embed SSIM={pr.embed_quality['ssim']:.4f} PSNR={pr.embed_quality['psnr']:.2f}")
    assert pr.auc > 0.90, f"Clean AUC 过低: {pr.auc}"


def test_d0_embed_quality():
    cfg = _cfg(n=16)
    imgs = data.load_dataset(n=16, size=128, seed=4)
    pr = runner.evaluate_clean(defense_d0.embed, defense_d0.detect, imgs, cfg, "D0")
    print(f"[D0] embed SSIM={pr.embed_quality['ssim']:.4f} PSNR={pr.embed_quality['psnr']:.2f} "
          f"valid_rate={pr.embed_valid_rate:.3f}")
    # 嵌入应基本不可见
    assert pr.embed_quality["ssim"] > 0.95
    assert pr.embed_quality["psnr"] > 30


def test_d0_a0_robust():
    cfg = _cfg(n=24)
    imgs = data.load_dataset(n=24, size=128, seed=5)
    pr = runner.evaluate_pair(attack_a0.attack, defense_d0.embed, defense_d0.detect,
                              imgs, cfg, "A0", "D0")
    print(f"[D0 x A0] Robust AUC={pr.auc:.4f} attack_valid={pr.attack_valid_rate:.3f} "
          f"attack SSIM={pr.attack_quality['ssim']:.4f} PSNR={pr.attack_quality['psnr']:.2f}")
    # A0 攻击后 D0 仍应优于随机（有限鲁棒性）
    assert pr.auc > 0.50, f"A0 后 AUC={pr.auc} 应 > 0.5"


# --------------------------------------------------------------------------- #
# 非法处理（§9.6）
# --------------------------------------------------------------------------- #
def test_illegal_attack_fallback():
    """攻击抛异常 -> valid_attack=0，回退为攻击前图进检测。"""
    cfg = _cfg(n=8)

    def bad_attack(sample):
        raise RuntimeError("intentional crash")

    imgs = data.load_dataset(n=8, size=128, seed=6)
    pr = runner.evaluate_pair(bad_attack, defense_d0.embed, defense_d0.detect,
                              imgs, cfg, "bad", "D0")
    # 攻击全部失败
    assert pr.attack_success_rate == 0.0
    assert pr.attack_valid_rate == 0.0
    # 回退为攻击前图（带水印正样本 vs 原图负样本），AUC 应较高
    assert pr.auc > 0.8, f"非法攻击回退后 AUC={pr.auc} 应较高（攻击方无收益）"


def test_illegal_embed_fallback():
    """嵌入抛异常 -> 正样本用原图，正负分支同源 -> AUC 接近 0.5（防御方受罚）。"""
    cfg = _cfg(n=8)

    def bad_embed(sample):
        raise RuntimeError("intentional crash")

    def const_detect(request):
        return {"watermark_probability": 0.5}

    imgs = data.load_dataset(n=8, size=128, seed=7)
    pr = runner.evaluate_pair(identity_attack.attack, bad_embed, const_detect,
                              imgs, cfg, "Id", "badD")
    assert pr.embed_success_rate == 0.0
    assert pr.embed_valid_rate == 0.0
    # 正负样本都来自原图 + 恒等攻击 -> 检测分数恒 0.5 -> AUC=0.5（并列）
    assert abs(pr.auc - 0.5) < 1e-6


# --------------------------------------------------------------------------- #
# 矩阵 + 排行
# --------------------------------------------------------------------------- #
def test_matrix_and_ranking():
    cfg = _cfg(n=16)
    imgs = data.load_dataset(n=16, size=128, seed=8)
    attacks = {"Id": identity_attack.attack, "A0": attack_a0.attack}
    defenses = {"D0": (defense_d0.embed, defense_d0.detect),
                "Null": (lambda s: {"image": np.asarray(s["image"]).copy()},
                         lambda r: {"watermark_probability": 0.5})}
    res = matrix.evaluate_matrix(attacks, defenses, imgs, cfg)
    # 矩阵形状
    assert set(res.matrix.keys()) == {"Id", "A0"}
    assert set(res.matrix["Id"].keys()) == {"D0", "Null"}
    # Null 检测器恒 0.5 -> AUC=0.5
    assert abs(res.matrix["Id"]["Null"] - 0.5) < 1e-6
    assert abs(res.matrix["A0"]["Null"] - 0.5) < 1e-6
    # 攻击榜：A0 应不劣于 Id（A0 AUC <= Id AUC），故 A0 排前
    print(f"\n[matrix] Id x D0 = {res.matrix['Id']['D0']:.4f}, A0 x D0 = {res.matrix['A0']['D0']:.4f}")
    print(f"[rank] attack={res.attack_ranking} defense={res.defense_ranking}")
    assert res.attack_ranking[0] == "A0"
    # 防御榜：D0 优于 Null
    assert res.defense_ranking[0] == "D0"


def test_qualification():
    cfg = _cfg(n=16)
    imgs = data.load_dataset(n=16, size=128, seed=9)
    # D0 防御自检
    qd = matrix.qualify_defense(defense_d0.embed, defense_d0.detect,
                                attack_a0.attack, imgs, cfg)
    print(f"\n[qualify-defense D0] passed={qd['passed']} "
          f"clean={qd['clean_auc']:.4f} robust={qd['robust_auc']:.4f}")
    # 攻击资格：A0 相对 Id 应使 AUC 下降
    qa = matrix.qualify_attack(attack_a0.attack, defense_d0.embed,
                               defense_d0.detect, imgs, cfg)
    print(f"[qualify-attack A0] passed={qa['passed']} "
          f"auc_id={qa['auc_id']:.4f} auc_attack={qa['auc_attack']:.4f} delta={qa['delta']:.4f}")
    assert qa["delta"] > 0


if __name__ == "__main__":
    # 直接运行：依次执行所有 test_* 函数
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n==== {passed} passed, {failed} failed ====")
    sys.exit(1 if failed else 0)
