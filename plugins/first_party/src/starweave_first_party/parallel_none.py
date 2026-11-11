"""No-op parallel plugins for single-device v1 execution."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from starweave.seams import (
    CollectiveConfig,
    EPConfig,
    FSDPConfig,
    ParallelPlan,
    PluginBase,
)


class NoopCollective(PluginBase):
    name = "collective-none"
    provides = ("collective",)
    requires = ()
    config_model = CollectiveConfig

    def setup(self, ctx: Any, config: CollectiveConfig) -> None:
        super().setup(ctx, config)
        self.world_size = config.world_size
        self.rank = 0

    def all_reduce(self, tensor: Any, op: str = "sum") -> Any:
        return tensor

    def all_gather(self, tensor: Any) -> list[Any]:
        return [tensor]

    def broadcast(self, tensor: Any, src: int = 0) -> Any:
        return tensor

    def all_to_all(self, tensor: Any) -> Any:
        return tensor

    def barrier(self) -> None:
        return None


class NoopFSDP(PluginBase):
    name = "fsdp-none"
    provides = ("fsdp",)
    requires = ("collective",)
    config_model = FSDPConfig

    def apply(self, ctx: Any, model: Any, plan: ParallelPlan) -> Any:
        return model

    def no_sync(self):
        return nullcontext()

    def state_dict(self, model: Any) -> dict[str, Any]:
        return {}

    def load_state_dict(self, model: Any, state: dict[str, Any]) -> None:
        return None


class NoopEP(PluginBase):
    name = "ep-none"
    provides = ("ep",)
    requires = ("collective",)
    config_model = EPConfig

    def apply(self, ctx: Any, model: Any, plan: ParallelPlan) -> Any:
        return model

    def dispatch(self, tokens: Any, scores: Any) -> Any:
        return tokens, scores

    def combine(self, outputs: Any) -> Any:
        return outputs
