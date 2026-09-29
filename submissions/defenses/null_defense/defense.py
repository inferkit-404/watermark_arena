import numpy as np


def embed(sample: dict) -> dict:
    """赛题 §21 最小防御示例：不嵌入任何水印。"""
    return {"image": np.asarray(sample["image"]).copy()}


def detect(request: dict) -> dict:
    return {"watermark_probability": 0.5}
