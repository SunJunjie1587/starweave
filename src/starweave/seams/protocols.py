"""Seam Protocols and no-op base implementations.

Only standard library, typing, and pydantic config types are imported here.
torch/transformers objects cross seams as opaque ``Any`` handles.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

from .configs import (
    AlgorithmConfig,
    CollectiveConfig,
    DataConfig,
    DeviceConfig,
    EPConfig,
    FSDPConfig,
    InferenceConfig,
    KernelConfig,
    LauncherConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    PrecisionConfig,
    RewardConfig,
    RolloutConfig,
    SchedulerConfig,
    TaskConfig,
    TrainerConfig,
)
from .types import (
    CheckpointState,
    DataBundle,
    MetricEvent,
    ModelHandle,
    OptimizerHandle,
    ParallelPlan,
    Rebindable,
    RolloutOutput,
    RunState,
    SchedulerHandle,
    StepEvent,
    StepState,
)


@runtime_checkable
class Device(Protocol):
    name: str
    device_type: str

    def set_device(self, index: int) -> None: ...
    def current_device(self) -> int: ...
    def synchronize(self) -> None: ...
    def memory_stats(self) -> dict[str, int]: ...
    def empty_cache(self) -> None: ...


@runtime_checkable
class Precision(Protocol):
    name: str
    dtype: str

    def autocast(self) -> AbstractContextManager: ...
    def cast_model(self, model: Any) -> Any: ...


@runtime_checkable
class Collective(Protocol):
    name: str
    world_size: int
    rank: int

    def all_reduce(self, tensor: Any, op: str = "sum") -> Any: ...
    def all_gather(self, tensor: Any) -> list[Any]: ...
    def broadcast(self, tensor: Any, src: int = 0) -> Any: ...
    def all_to_all(self, tensor: Any) -> Any: ...
    def barrier(self) -> None: ...


@runtime_checkable
class FSDPBackend(Protocol):
    name: str

    def apply(self, ctx: Any, model: Any, plan: ParallelPlan) -> Any: ...
    def no_sync(self) -> AbstractContextManager: ...
    def state_dict(self, model: Any) -> dict[str, Any]: ...
    def load_state_dict(self, model: Any, state: dict[str, Any]) -> None: ...


@runtime_checkable
class EPBackend(Protocol):
    name: str

    def apply(self, ctx: Any, model: Any, plan: ParallelPlan) -> Any: ...
    def dispatch(self, tokens: Any, scores: Any) -> Any: ...
    def combine(self, outputs: Any) -> Any: ...


@runtime_checkable
class KernelRegistry(Protocol):
    def register(
        self,
        op: str,
        impl: Callable[..., Any],
        *,
        backend: str,
        dtype: str | None = None,
        priority: int = 100,
    ) -> None: ...
    def get(self, op: str, *, backend: str, dtype: str | None = None) -> Callable[..., Any]: ...
    def list(self) -> list[Any]: ...


@runtime_checkable
class ModelBuilder(Protocol):
    name: str

    def build(self, ctx: Any, config: ModelConfig | Any) -> ModelHandle: ...


@runtime_checkable
class DataSource(Protocol):
    name: str

    def build(self, ctx: Any, config: DataConfig) -> DataBundle: ...


@runtime_checkable
class OptimizerFactory(Protocol):
    name: str

    def build(self, ctx: Any, model: ModelHandle, config: OptimizerConfig) -> OptimizerHandle: ...


@runtime_checkable
class SchedulerFactory(Protocol):
    name: str

    def build(
        self,
        ctx: Any,
        optimizer: OptimizerHandle,
        total_steps: int,
        config: SchedulerConfig,
    ) -> SchedulerHandle: ...


@runtime_checkable
class LossFactory(Protocol):
    name: str

    def build(self, ctx: Any, config: LossConfig) -> Callable[[Any, Any], Any]: ...


@runtime_checkable
class CheckpointStore(Protocol):
    name: str

    def save(self, ctx: Any, state: CheckpointState) -> Any: ...
    def load(self, ctx: Any, path: Any) -> CheckpointState: ...
    def latest(self, ctx: Any) -> Any | None: ...


@runtime_checkable
class MetricsSink(Protocol):
    name: str

    def emit(self, ctx: Any, event: MetricEvent) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class Profiler(Protocol):
    name: str

    def start(self, ctx: Any) -> None: ...
    def stop(self) -> None: ...


@runtime_checkable
class Trainer(Protocol):
    name: str

    def compile(self, ctx: Any, services: Any) -> Any: ...
    def run(self, ctx: Any, plan: Any, max_steps: int) -> Iterator[StepEvent]: ...


@runtime_checkable
class InferenceEngine(Protocol):
    name: str

    def load(self, ctx: Any, model: ModelHandle, config: InferenceConfig) -> None: ...
    def generate(self, ctx: Any, requests: list[Any]) -> Iterator[Any]: ...


@runtime_checkable
class Rollout(Protocol):
    name: str

    def setup(self, ctx: Any, config: RolloutConfig) -> None: ...
    def generate(self, ctx: Any, requests: list[Any]) -> list[RolloutOutput]: ...
    def update_weights(self, ctx: Any, model: ModelHandle) -> None: ...


@runtime_checkable
class Task(Protocol):
    name: str
    mode: str

    def setup(self, ctx: Any, services: Any) -> None: ...
    def run(self, ctx: Any) -> Iterator[Any]: ...


@runtime_checkable
class Reward(Protocol):
    name: str

    def score(self, ctx: Any, samples: list[Any]) -> list[float]: ...


@runtime_checkable
class Algorithm(Protocol):
    name: str

    def compute_advantages(self, rewards: Any, group_size: int) -> Any: ...
    def policy_loss(self, logprobs: Any, advantages: Any, mask: Any) -> Any: ...


@runtime_checkable
class Launcher(Protocol):
    name: str

    def run(self, ctx: Any, fn: Callable[[], int]) -> int: ...


@runtime_checkable
class StepPolicy(Protocol):
    name: str
    priority: int

    def before_step(self, ctx: Any, state: StepState) -> None: ...
    def after_backward(self, ctx: Any, state: StepState) -> None: ...
    def after_step(self, ctx: Any, state: StepState) -> None: ...
    def on_error(self, ctx: Any, state: StepState) -> None: ...


@runtime_checkable
class CheckpointPolicy(Protocol):
    name: str
    priority: int

    def should_save(self, ctx: Any, state: RunState) -> bool: ...


@runtime_checkable
class EvalPolicy(Protocol):
    name: str
    priority: int

    def should_eval(self, ctx: Any, state: RunState) -> bool: ...


class BaseRebindable:
    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def release(self) -> None:
        return None


class BaseStepPolicy:
    priority = 100

    def before_step(self, ctx: Any, state: StepState) -> None:
        return None

    def after_backward(self, ctx: Any, state: StepState) -> None:
        return None

    def after_step(self, ctx: Any, state: StepState) -> None:
        return None

    def on_error(self, ctx: Any, state: StepState) -> None:
        return None


class BaseCheckpointPolicy:
    priority = 100

    def should_save(self, ctx: Any, state: RunState) -> bool:
        return False


class BaseEvalPolicy:
    priority = 100

    def should_eval(self, ctx: Any, state: RunState) -> bool:
        return False


# Keep config classes importable from this module for convenience.
__all__ = [
    "BaseCheckpointPolicy",
    "BaseEvalPolicy",
    "BaseRebindable",
    "BaseStepPolicy",
    "CheckpointPolicy",
    "CheckpointStore",
    "Collective",
    "CollectiveConfig",
    "DataConfig",
    "DataSource",
    "Device",
    "DeviceConfig",
    "EPBackend",
    "EPConfig",
    "EvalPolicy",
    "FSDPBackend",
    "FSDPConfig",
    "InferenceConfig",
    "InferenceEngine",
    "KernelConfig",
    "KernelRegistry",
    "Launcher",
    "LauncherConfig",
    "LossConfig",
    "LossFactory",
    "MetricsSink",
    "ModelBuilder",
    "ModelConfig",
    "OptimizerConfig",
    "OptimizerFactory",
    "Precision",
    "PrecisionConfig",
    "Profiler",
    "Rebindable",
    "SchedulerConfig",
    "SchedulerFactory",
    "StepPolicy",
    "Trainer",
    "TrainerConfig",
]
