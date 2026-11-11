"""Seam vocabulary types. This module must not import torch or framework internals."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class ParallelPlan:
    shard_modules: tuple[str, ...] = ()
    expert_modules: tuple[str, ...] = ()
    replicate_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelHandle:
    model: Any
    tokenizer: Any
    vocab_size: int
    param_count: int
    dtype: str
    parallel_plan: ParallelPlan = field(default_factory=ParallelPlan)


@dataclass(frozen=True)
class DataBundle:
    dataset: Any
    collator: Callable[[list[dict[str, Any]]], dict[str, Any]]
    iterator_factory: Callable[[], Iterator[dict[str, Any]]]


@dataclass
class OptimizerHandle:
    optimizer: Any

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state)


@dataclass
class SchedulerHandle:
    scheduler: Any

    def step(self) -> None:
        self.scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        return self.scheduler.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state)


@dataclass
class CheckpointState:
    model_dir: Path
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    step: int
    dataloader_position: int
    rng: dict[str, Any]
    composition_hash: str
    config_snapshot: dict[str, Any]
    model_state: dict[str, Any] | None = None


@dataclass
class StepState:
    step: int = 0
    micro_step: int = 0
    loss: float | None = None
    lr: float | None = None
    should_step: bool = True
    should_abort: bool = False
    error: BaseException | None = None
    accum_steps: int = 1
    loss_scale: float = 1.0
    batch: Any = None


@dataclass(frozen=True)
class StepEvent:
    step: int
    loss: float
    lr: float
    tokens: int
    elapsed_ms: float


@dataclass(frozen=True)
class RunState:
    run_id: str
    phase: str
    step: int
    epoch: int
    best_metric: float | None = None


@dataclass(frozen=True)
class MetricEvent:
    name: str
    value: float
    step: int
    phase: str
    timestamp: float


@dataclass(frozen=True)
class RolloutOutput:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    text: str = ""
    logprobs: Any = None


@dataclass(frozen=True)
class StepPlan:
    before_step: tuple[Callable[..., None], ...] = ()
    forward_backward: Callable[..., Any] | None = None
    after_backward: tuple[Callable[..., None], ...] = ()
    optimizer_step: Callable[..., None] | None = None
    after_step: tuple[Callable[..., None], ...] = ()
    on_error: tuple[Callable[..., None], ...] = ()


@runtime_checkable
class Rebindable(Protocol):
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def release(self) -> None: ...


@dataclass
class RLServices:
    model: ModelHandle
    data: DataBundle
    rollout: Any
    reward: Any
    algorithm: Any
    optimizer: OptimizerHandle
    scheduler: SchedulerHandle
    checkpoint: Any
    metrics: Any
    device: Any
    precision: Any
    task_config: Any
    numerics: Any = None
    composition_hash: str = ""


@dataclass
class TrainingServices:
    model: ModelHandle
    data: DataBundle
    optimizer: OptimizerHandle
    scheduler: SchedulerHandle
    loss: Callable[..., Any]
    device: Any
    precision: Any
    step_policies: tuple[Any, ...]
    checkpoint_policies: tuple[Any, ...]
    eval_policies: tuple[Any, ...]
    checkpoint: Any
    metrics: Any
    trainer_config: Any
    composition_hash: str = ""


__all__ = [
    "CheckpointState",
    "DataBundle",
    "MetricEvent",
    "ModelHandle",
    "OptimizerHandle",
    "ParallelPlan",
    "RLServices",
    "Rebindable",
    "RolloutOutput",
    "RunState",
    "SchedulerHandle",
    "StepEvent",
    "StepPlan",
    "StepState",
    "TrainingServices",
]
