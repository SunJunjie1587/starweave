"""Environment checks. No CPU fallback: missing torch_npu or zero NPUs fails."""

from __future__ import annotations

import importlib.util
import platform
import sys


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def doctor() -> int:
    print(f"python: {platform.python_version()} ({sys.executable})")
    checks = {
        "torch": _has("torch"),
        "torch_npu": _has("torch_npu"),
        "transformers": _has("transformers"),
        "pydantic": _has("pydantic"),
        "yaml": _has("yaml"),
    }
    for name, available in checks.items():
        print(f"{name}: {'ok' if available else 'missing'}")
    if not checks["torch_npu"]:
        print("error: torch_npu is not installed; Starweave v1 has no CPU fallback")
        return 3
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot import torch/torch_npu: {exc}")
        return 3
    count = torch.npu.device_count() if hasattr(torch, "npu") else 0
    print(f"npu_count: {count}")
    if count < 1:
        print("error: no Ascend NPU is visible")
        return 3
    return 0
