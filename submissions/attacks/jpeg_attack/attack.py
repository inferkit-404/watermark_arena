import io
import numpy as np


def attack(sample: dict) -> dict:
    """赛题 §20 最小攻击示例：JPEG q=82 重编码。"""
    image = sample["image"]
    from PIL import Image
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=82)
    buffer.seek(0)
    output = Image.open(buffer).convert("RGB")
    return {"image": np.asarray(output, dtype=np.uint8)}
