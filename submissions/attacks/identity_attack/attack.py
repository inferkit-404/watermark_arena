import numpy as np


def attack(sample: dict) -> dict:
    """恒等攻击示例：原样返回（不消除水印）。"""
    return {"image": np.asarray(sample["image"]).copy()}
