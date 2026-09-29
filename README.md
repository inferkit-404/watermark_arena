# WatermarkGuard-Image 攻防对抗后端

依据《图片隐藏水印消除与鲁棒检测攻防对抗赛题.md》实现的**自包含评测后端**。
对**任意"攻击算法 × 防御算法"组合**，自动化计算对应的攻击指标与防御指标，
输出完整攻防矩阵、排行榜与质量/成功率/耗时明细。

---

## 一、核心特性

- **任意组合自动评测**：笛卡尔积 `A_q × D_q`，每对 `(攻击, 防御)` 输出一个 ROC-AUC（赛题 §10/§12）。
  - 攻击指标 `R_A = mean_d M[a,d]`（越低越好）+ tie-break（§13）
  - 防御指标 `R_D = mean_a M[a,d]`（越高越好）+ tie-break（§14）
- **同源配对 + 对称攻击**（§6/§10）：同一攻击同参同时作用于正/负分支，攻击/检测不接收标签。
- **图像合法性与质量裁判**（§9）：尺寸/通道/dtype/NaN/纯色校验 + SSIM/PSNR/LPIPS 质量门槛。
- **非法样本处理**（§9.6）：嵌入失败→正样本用原图；攻击失败→用攻击前图进检测；检测异常→中性 0.5。
- **ROC-AUC 保留方向**（§10.4）：纯 numpy Mann-Whitney U，处理并列秩，不做 `max(AUC,1-AUC)` 修正。
- **第一阶段资格校验**（§11）：攻击资格（对 D0 的 AUC 下降量）、防御资格（Clean/A0 AUC 门槛）。
- **公开基线**：D0（DCT 中频扩频零比特水印，§11.1）、A0（JPEG75+缩放+模糊，§11.2）、恒等攻击。
- **可插拔提交**：放入 `submissions/` 目录即可参与评测；异常隔离，单个崩溃不影响全局。

## 二、关于"禁止读取图片"约束

> 该约束限制的是 **AI 智能体在开发/分析过程中用工具查看图片**，不限制项目代码读取图片。

- 后端代码**允许**读取图片：`arena/data.py` 支持从目录读取真实图片（PIL）。
- 默认数据源为**程序化合成**（`data_source: null`），全程 numpy，开箱即用、无需外部数据，
  覆盖赛题 §5.1 多种视觉内容（自然场景/人像/建筑商品/图表文档/生成式/不同纹理亮度色彩）。
- 所有攻击/防御接口只传递内存中的 `np.ndarray`（RGB uint8 H×W×3），与赛题统一图像格式一致。

## 三、目录结构

```
watermark_arena/
├── cli.py                    # 命令行入口
├── config.yaml               # 评测参数(N/尺寸/质量阈值/资格门槛)
├── requirements.txt
├── arena/
│   ├── data.py               # 图像来源:目录读取 + 程序化合成
│   ├── metrics.py            # PSNR/SSIM/LPIPS(可选)/ROC-AUC + 质量判定
│   ├── validation.py         # §9.1 图像合法性校验
│   ├── loader.py             # 动态加载 attack.py/defense.py + 安全调用
│   ├── runner.py             # §10 单组合评测流水线(同源配对+对称攻击+非法替换)
│   ├── matrix.py             # §12 全对阵矩阵 + §13/14 排行 + §11 资格校验
│   └── report.py             # §23 报告:矩阵/榜单/热力图/CSV/JSON
├── baselines/
│   ├── defense_d0.py         # §11.1 DCT 中频扩频零比特水印
│   ├── attack_a0.py          # §11.2 JPEG75->缩放90%->双线性->高斯模糊σ0.4
│   └── identity_attack.py    # 恒等攻击(资格对照)
├── submissions/              # 示例提交(可替换/新增)
│   ├── attacks/{identity_attack,jpeg_attack}/attack.py
│   └── defenses/null_defense/defense.py
└── tests/test_pipeline.py
```

## 四、安装

```bash
cd watermark_arena
pip install -r requirements.txt
# 必需: numpy scipy pyyaml
# 可选(缺失则优雅降级): Pillow(JPEG/读图) lpips torch(LPIPS)
```

## 五、快速开始

```bash
# 1) 列出可用基线
python cli.py list-baselines

# 2) 全对阵矩阵(基线 + 示例提交)，结果写入 ./results
python cli.py run-matrix \
    --attacks baseline:identity baseline:a0 submissions/attacks/jpeg_attack \
    --defenses baseline:d0 submissions/defenses/null_defense \
    --n 32 --out ./results

# 3) 单组合明细
python cli.py run-pair --attack baseline:a0 --defense baseline:d0 --n 32

# 4) 资格校验
python cli.py qualify-attack  --attack submissions/attacks/jpeg_attack
python cli.py qualify-defense --defense baseline:d0
```

输出：控制台打印矩阵/排行榜/热力图；`--out` 目录写 `matrix.csv`、`result.json`、`report.txt`。

## 六、提交接口

### 攻击方 `attack.py`
```python
import numpy as np
def attack(sample: dict) -> dict:
    # sample["sample_id"]: str, sample["image"]: RGB uint8 H×W×3
    return {"image": np.ndarray}  # RGB uint8 H×W×3
```

### 防御方 `defense.py`
```python
import numpy as np
def embed(sample: dict) -> dict:
    return {"image": np.ndarray}
def detect(request: dict) -> dict:
    return {"watermark_probability": float}  # 0.0~1.0
```

把提交目录放入 `submissions/attacks/<name>/` 或 `submissions/defenses/<name>/`，
CLI 中用目录路径引用即可。提交包内可含 `methods/`、`weights/`、`assets/` 等，
其目录会被加入 `sys.path` 便于包内导入（密钥/权重仅在防御容器内可见，§7.4）。

## 七、配置项（config.yaml）

| 路径 | 说明 |
|---|---|
| `evaluation.n_samples` | 每组合采样原图数 |
| `evaluation.data_source` | `null`=合成；或图片目录路径 |
| `quality.embed` | 嵌入质量硬约束（SSIM≥0.97/PSNR≥36/LPIPS≤0.06） |
| `quality.attack` | 攻击质量硬约束（SSIM≥0.92/PSNR≥28/LPIPS≤0.15） |
| `quality.strict` | `true` 时质量不达标视为非法，触发 §9.6 替换 |
| `qualification.*` | §11 资格门槛（仅筛选，不计入第二阶段平均分） |

## 八、评测流程（赛题 §10 / §19.2）

```
对每个 (A_a, D_d)，同一批原图 I：
  正分支: I -> D_d.embed -> A_a.attack -> D_d.detect -> s+
  负分支: I ────────────-> A_a.attack -> D_d.detect -> s-
  y_true=[1..1,0..0], y_score=[s+..,s-..]
  M[a,d] = ROC-AUC(y_true, y_score)
```

非法/质量处理：
- 嵌入输出非法 → 正样本用**原始无水印图**替代（`valid_embed=0`，防御 AUC 下降）
- 攻击输出非法 → 用**攻击前输入**进检测（`valid_attack=0`，攻击方无收益）
- 检测异常 → 中性 `0.5`（`detect_success=0`）

## 九、指标说明

- **攻击指标**：`mean_auc`（主榜，升序）、`max_auc`（最坏防御，tie-break）、
  `strength=1-mean_auc`（展示用）、平均攻击 LPIPS/耗时/合法率。
- **防御指标**：`mean_auc`（主榜，降序）、`min_auc`（最强攻击，tie-break）、
  `clean_auc`（无攻击，tie-break）、平均嵌入 LPIPS/耗时/合法率。

## 十、测试

```bash
python -m pytest tests/ -v        # 或
python tests/test_pipeline.py
```

覆盖：ROC-AUC 正确性（完美/反转/并列/已知值）、合法性校验、合成多样性、
D0 嵌入质量与 Clean/A0 AUC、非法处理回退、矩阵与排行、资格校验。
