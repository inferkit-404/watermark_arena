"""动态加载攻击/防御提交，并提供异常隔离的安全调用。

每个提交目录在独立模块命名空间中加载（避免全局污染与同名冲突）。
safe_call 捕获一切异常并返回 (ok, value/error)，供流水线据此触发 §9.6 替换。

提交目录约定：
  攻击方：含 attack.py，定义 attack(sample)->{"image":...}
  防御方：含 defense.py，定义 embed(sample)->{"image":...} 与 detect(request)->{"watermark_probability":float}
"""
from __future__ import annotations

import os
import sys
import time
import importlib.util
from typing import Callable, Any

_LOAD_COUNTER = 0


def load_attack(directory: str):
    return _load_module(directory, "attack.py", "attack_mod", ["attack"])


def load_defense(directory: str):
    return _load_module(directory, "defense.py", "defense_mod", ["embed", "detect"])


def _load_module(directory: str, filename: str, kind: str, expected_funcs: list[str]):
    global _LOAD_COUNTER
    directory = os.path.abspath(directory)
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"在 {directory} 中未找到 {filename}")

    # 将提交目录加入 sys.path，便于提交包内相对导入（methods/weights 等）
    if directory not in sys.path:
        sys.path.insert(0, directory)

    _LOAD_COUNTER += 1
    modname = f"_wma_{kind}_{_LOAD_COUNTER}"

    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {path} 创建模块规格")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)

    for fn in expected_funcs:
        if not hasattr(module, fn) or not callable(getattr(module, fn)):
            raise AttributeError(f"{filename} 缺少函数: {fn}")
    return module


def safe_call(fn: Callable, *args, **kwargs) -> tuple[bool, Any]:
    """调用提交函数，捕获一切异常。返回 (ok, result_or_errorstr)。"""
    try:
        return True, fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 - 故意宽捕获，隔离不可信提交
        return False, f"{type(e).__name__}: {e}"


def timed_safe_call(fn: Callable, *args, **kwargs) -> tuple[bool, Any, float]:
    """safe_call + 计时。返回 (ok, result, elapsed_seconds)。"""
    t0 = time.perf_counter()
    ok, res = safe_call(fn, *args, **kwargs)
    return ok, res, time.perf_counter() - t0
