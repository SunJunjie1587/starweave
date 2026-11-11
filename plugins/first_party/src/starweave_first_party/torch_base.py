"""Generic PyTorch-level plugins: launcher, fp32 precision, kernel registry.

This module is part of the first-party plugin package, not Starweave core.
It imports only ``starweave.seams`` and its own dependencies.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from starweave.seams import (
    BaseRebindable,
    KernelConfig,
    LauncherConfig,
    PluginBase,
    PrecisionConfig,
    ProfilerConfig,
)


class LocalLauncher(PluginBase):
    name = "launcher-local"
    provides = ("launcher",)
    requires = ()
    config_model = LauncherConfig

    def run(self, ctx: Any, fn) -> int:
        return fn()


class FP32Precision(PluginBase):
    name = "precision-fp32"
    provides = ("precision",)
    requires = ()
    config_model = PrecisionConfig
    dtype = "float32"

    def setup(self, ctx: Any, config: PrecisionConfig) -> None:
        super().setup(ctx, config)
        self.dtype = "float32"

    def autocast(self):
        return nullcontext()

    def cast_model(self, model: Any) -> Any:
        return model


class TorchKernel(PluginBase):
    name = "kernel-torch"
    provides = ("kernel",)
    requires = ()
    config_model = KernelConfig

    def setup(self, ctx: Any, config: KernelConfig) -> None:
        super().setup(ctx, config)
        self._ops: dict[tuple[str, str, str | None], Any] = {}

    def register(
        self,
        op: str,
        impl: Any,
        *,
        backend: str,
        dtype: str | None = None,
        priority: int = 100,
    ) -> None:
        self._ops[(op, backend, dtype)] = impl

    def get(self, op: str, *, backend: str, dtype: str | None = None) -> Any:
        try:
            return self._ops[(op, backend, dtype)]
        except KeyError as exc:
            raise KeyError(f"kernel op '{op}' is not registered for {backend}/{dtype}") from exc

    def list(self) -> list[Any]:
        return [
            {"op": op, "backend": backend, "dtype": dtype}
            for (op, backend, dtype) in self._ops
        ]


class NoopProfiler(PluginBase, BaseRebindable):
    name = "profiler-off"
    provides = ("profiler",)
    requires = ()
    config_model = ProfilerConfig

    def start(self, ctx: Any) -> None:
        return None

    def stop(self) -> None:
        return None
