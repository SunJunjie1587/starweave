"""Measure Starweave's dispatch overhead against raw Python loops.

Two numbers are reported:

1. ``dispatch``: a static StepPlan calling the same hooks as a raw loop.
2. ``trainer``: the SFTTrainer loop (policies, events, metrics) versus an
   equivalent hand-written loop with the same fake compute.

Fake compute keeps the comparison focused on framework overhead; a real NPU
step is orders of magnitude larger, so the real ratio is lower.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

from starweave.kernel.context import Context
from starweave.plan.plan import StepPlan
from starweave_first_party.policies import GradAccumConfig, GradAccumPolicy
from starweave_first_party.trainer_sft import SFTTrainer
from starweave.seams.configs import TrainerConfig
from starweave.seams.types import (
    DataBundle,
    ModelHandle,
    OptimizerHandle,
    SchedulerHandle,
    TrainingServices,
)


class FakeLoss:
    def __init__(self, value: float) -> None:
        self.value = value

    def backward(self) -> None:
        _compute(40_000)

    def __mul__(self, other: float):
        return FakeLoss(self.value * other)

    def detach(self):
        return self

    def item(self) -> float:
        return self.value


class FakeOutputs:
    def __init__(self, loss: FakeLoss) -> None:
        self.loss = loss


class FakeModel:
    def __call__(self, **batch):
        return FakeOutputs(FakeLoss(1.0))

    def state_dict(self):
        return {}

    def parameters(self):
        return []

    def train(self):
        return None

    def eval(self):
        return None


class FakeOptimizer:
    def step(self) -> None:
        _compute(40_000)

    def zero_grad(self, set_to_none: bool = True) -> None:
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state) -> None:
        return None


class FakeScheduler:
    def step(self) -> None:
        return None

    def get_last_lr(self):
        return [1.0e-4]

    def state_dict(self):
        return {}

    def load_state_dict(self, state) -> None:
        return None


class NoopMetrics:
    def emit(self, ctx, event) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def release(self) -> None:
        return None


class NoopCheckpoint:
    def save(self, ctx, state) -> Path:
        return Path("/tmp/bench")


def _compute(iterations: int = 400) -> int:
    total = 0
    for index in range(iterations):
        total += index * index
    return total


class FakeTensor:
    def numel(self) -> int:
        return 3


def _batch():
    return {"input_ids": FakeTensor(), "labels": [1, 2, 3]}


def _raw_dispatch(steps: int) -> int:
    state = 0
    for _ in range(steps):
        state = _compute()
        state = _compute()
        state = _compute()
    return state


def _plan_dispatch(steps: int) -> int:
    state = {"value": 0}

    def before_step(_state) -> None:
        return None

    def forward_backward(_state, _batch) -> None:
        state["value"] = _compute() + _compute()

    def after_backward(_state) -> None:
        return None

    def optimizer_step(_state) -> None:
        state["value"] = _compute()

    def after_step(_state) -> None:
        return None

    plan = StepPlan(
        before_step=(before_step,),
        forward_backward=forward_backward,
        after_backward=(after_backward,),
        optimizer_step=optimizer_step,
        after_step=(after_step,),
    )
    for _ in range(steps):
        for hook in plan.before_step:
            hook(None)
        plan.forward_backward(None, None)
        for hook in plan.after_backward:
            hook(None)
        plan.optimizer_step(None)
        for hook in plan.after_step:
            hook(None)
    return state["value"]


def _raw_train(steps: int) -> None:
    model = FakeModel()
    optimizer = FakeOptimizer()
    scheduler = FakeScheduler()
    batch = _batch()
    for _ in range(steps):
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def _plugin_train(steps: int) -> None:
    ctx = Context("overhead")
    trainer = SFTTrainer()
    trainer.setup(ctx, TrainerConfig(max_steps=steps))
    accum = GradAccumPolicy()
    accum.setup(ctx, GradAccumConfig(steps=1))
    services = TrainingServices(
        model=ModelHandle(FakeModel(), tokenizer=None, vocab_size=0, param_count=0, dtype="float32"),
        data=DataBundle(
            dataset=[],
            collator=lambda batch: batch,
            iterator_factory=lambda: iter([_batch() for _ in range(steps + 1)]),
        ),
        optimizer=OptimizerHandle(FakeOptimizer()),
        scheduler=SchedulerHandle(FakeScheduler()),
        loss=lambda logits, labels: FakeLoss(1.0),
        device=object(),
        precision=type("Precision", (), {"autocast": lambda self: nullcontext()})(),
        step_policies=(accum,),
        checkpoint_policies=(),
        eval_policies=(),
        checkpoint=NoopCheckpoint(),
        metrics=NoopMetrics(),
        trainer_config=TrainerConfig(max_steps=steps),
    )
    plan = trainer.compile(ctx, services)
    list(trainer.run(ctx, plan, steps))
    ctx.close()


def _time(fn, arg: int) -> float:
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        fn(arg)
        best = min(best, time.perf_counter() - started)
    return best


def main() -> int:
    dispatch_steps = 200_000
    raw = _time(_raw_dispatch, dispatch_steps) / dispatch_steps * 1e9
    plan = _time(_plan_dispatch, dispatch_steps) / dispatch_steps * 1e9
    dispatch_overhead = (plan - raw) / raw * 100.0

    train_steps = 500
    raw_train = _time(_raw_train, train_steps) / train_steps * 1e6
    plugin_train = _time(_plugin_train, train_steps) / train_steps * 1e6
    trainer_overhead = (plugin_train - raw_train) / raw_train * 100.0
    absolute_us = plugin_train - raw_train

    print(f"dispatch: raw={raw:.1f} ns/step plan={plan:.1f} ns/step overhead={dispatch_overhead:.2f}%")
    print(f"trainer:  raw={raw_train:.2f} us/step plugin={plugin_train:.2f} us/step "
          f"overhead={trainer_overhead:.2f}% (absolute={absolute_us:.2f} us/step)")
    print(f"at a measured 77 ms NPU step, framework overhead is "
          f"{absolute_us / 77_000.0 * 100.0:.4f}%")
    return 0 if max(dispatch_overhead, trainer_overhead) < 5.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
