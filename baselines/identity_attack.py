"""恒等攻击：不做任何修改（用于资格对照 AUC(Id, D0) 与 Clean AUC）。"""
from __future__ import annotations

import numpy as np


def attack(sample: dict) -> dict:
    return {"image": np.asarray(sample["image"]).copy()}
