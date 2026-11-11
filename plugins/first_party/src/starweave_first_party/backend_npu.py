"""Ascend NPU backend plugins: device, bf16 precision, kernel, profiler."""

from __future__ import annotations

from typing import Any

from starweave.seams import (
    DeviceConfig,
    KernelConfig,
    PluginBase,
    PrecisionConfig,
    ProfilerConfig,
)


class NPUDevice(PluginBase):
    name = "device-npu"
    provides = ("device",)
    requires = ()
    config_model = DeviceConfig
    device_type = "npu"

    def setup(self, ctx: Any, config: DeviceConfig) -> None:
        super().setup(ctx, config)
        self.index = config.index

    def set_device(self, index: int) -> None:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(index)

    def current_device(self) -> int:
        import torch

        return int(torch.npu.current_device())

    def synchronize(self) -> None:
        import torch

        torch.npu.synchronize()

    def memory_stats(self) -> dict[str, int]:
        import torch

        return {
            "allocated": int(torch.npu.memory_allocated()),
            "reserved": int(torch.npu.memory_reserved()),
        }

    def empty_cache(self) -> None:
        import torch

        torch.npu.empty_cache()


class BF16Precision(PluginBase):
    name = "precision-bf16"
    provides = ("precision",)
    requires = ()
    config_model = PrecisionConfig
    dtype = "bfloat16"

    def setup(self, ctx: Any, config: PrecisionConfig) -> None:
        super().setup(ctx, config)
        self.dtype = "bfloat16"

    def autocast(self):
        import torch

        return torch.autocast(device_type="npu", dtype=torch.bfloat16)

    def cast_model(self, model: Any) -> Any:
        return model


class NPUKernel(PluginBase):
    name = "kernel-npu"
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


class NPUProfiler(PluginBase):
    name = "profiler-npu"
    provides = ("profiler",)
    requires = ()
    config_model = ProfilerConfig

    def start(self, ctx: Any) -> None:
        return None

    def stop(self) -> None:
        return None
