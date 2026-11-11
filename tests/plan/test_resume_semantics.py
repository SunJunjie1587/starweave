from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from starweave.kernel.errors import RunError
from starweave.plan.runtime import apply_resume
from starweave.seams.types import (
    CheckpointState,
    ModelHandle,
    OptimizerHandle,
    SchedulerHandle,
)


class FakeModel:
    def __init__(self) -> None:
        self.loaded = None

    def load_state_dict(self, state) -> None:
        self.loaded = state


class FakeOptimizer:
    def __init__(self) -> None:
        self.loaded = None

    def load_state_dict(self, state) -> None:
        self.loaded = state


class FakeScheduler:
    def __init__(self) -> None:
        self.loaded = None

    def load_state_dict(self, state) -> None:
        self.loaded = state


class FakeCheckpoint:
    def __init__(self, state: CheckpointState) -> None:
        self.state = state

    def load(self, ctx, path):
        return self.state


def _state(composition_hash: str = "hash") -> CheckpointState:
    return CheckpointState(
        model_dir=Path("/tmp/ckpt"),
        optimizer={"step": 7},
        scheduler={"step": 7},
        step=7,
        dataloader_position=15,
        rng={},
        composition_hash=composition_hash,
        config_snapshot={},
        model_state={"w": 1},
    )


def _runtime(state: CheckpointState):
    model = FakeModel()
    optimizer = FakeOptimizer()
    scheduler = FakeScheduler()
    runtime = SimpleNamespace(
        ctx=None,
        services={
            "checkpoint": FakeCheckpoint(state),
            "model": ModelHandle(
                model=model, tokenizer=None, vocab_size=0, param_count=0, dtype="float32"
            ),
            "optimizer": OptimizerHandle(optimizer),
            "scheduler": SchedulerHandle(scheduler),
        },
        composition=SimpleNamespace(composition_hash="hash"),
    )
    return runtime, model, optimizer, scheduler


def test_resume_returns_next_step_and_next_micro_batch() -> None:
    runtime, model, optimizer, scheduler = _runtime(_state())
    assert apply_resume(runtime, "latest") == (8, 16)
    assert model.loaded == {"w": 1}
    assert optimizer.loaded == {"step": 7}
    assert scheduler.loaded == {"step": 7}


def test_resume_rejects_changed_composition_by_default() -> None:
    runtime, _, _, _ = _runtime(_state(composition_hash="other"))
    with pytest.raises(RunError):
        apply_resume(runtime, "latest")
    assert apply_resume(runtime, "latest", allow_config_change=True) == (8, 16)
