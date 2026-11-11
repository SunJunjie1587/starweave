from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from starweave.kernel.context import Context
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
        self.backward_calls = 0

    def backward(self) -> None:
        self.backward_calls += 1

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


class FakeOptimizer:
    def __init__(self) -> None:
        self.steps = 0
        self.zeroed = 0

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zeroed += 1

    def state_dict(self):
        return {}

    def load_state_dict(self, state) -> None:
        return None


class FakeScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def get_last_lr(self):
        return [1.0e-4]

    def state_dict(self):
        return {}

    def load_state_dict(self, state) -> None:
        return None


class FakeMetrics:
    def __init__(self) -> None:
        self.events = []

    def emit(self, ctx, event) -> None:
        self.events.append(event)


class FakeCheckpoint:
    def save(self, ctx, state) -> Path:
        return Path("/tmp/fake-checkpoint")


def _batch():
    return {"input_ids": [1, 2, 3], "labels": [1, 2, 3]}


def test_trainer_loop_with_fake_services() -> None:
    ctx = Context("trainer-test")
    trainer = SFTTrainer()
    trainer.setup(ctx, TrainerConfig(max_steps=2))

    accum = GradAccumPolicy()
    accum.setup(ctx, GradAccumConfig(steps=2))

    optimizer = OptimizerHandle(FakeOptimizer())
    scheduler = SchedulerHandle(FakeScheduler())
    metrics = FakeMetrics()
    services = TrainingServices(
        model=ModelHandle(FakeModel(), tokenizer=None, vocab_size=0, param_count=0, dtype="float32"),
        data=DataBundle(
            dataset=[],
            collator=lambda batch: batch,
            iterator_factory=lambda: iter([_batch() for _ in range(10)]),
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        loss=lambda logits, labels: FakeLoss(1.0),
        device=object(),
        precision=type("Precision", (), {"autocast": lambda self: nullcontext()})(),
        step_policies=(accum,),
        checkpoint_policies=(),
        eval_policies=(),
        checkpoint=FakeCheckpoint(),
        metrics=metrics,
        trainer_config=TrainerConfig(max_steps=2),
    )

    plan = trainer.compile(ctx, services)
    events = list(trainer.run(ctx, plan, 2))
    assert [event.step for event in events] == [0, 1]
    assert optimizer.optimizer.steps == 2
    assert len(metrics.events) == 2
    assert metrics.events[0].name == "loss"
    ctx.close()


def test_skip_batches_skips_exactly_n_batches_across_epochs() -> None:
    ctx = Context("trainer-skip-test")
    trainer = SFTTrainer()
    trainer.setup(ctx, TrainerConfig(max_steps=1))

    produced: list[int] = []

    def factory():
        def gen():
            for _ in range(2):
                index = len(produced)
                produced.append(index)
                yield {"input_ids": [index], "labels": [index]}

        return gen()

    services = TrainingServices(
        model=ModelHandle(FakeModel(), tokenizer=None, vocab_size=0, param_count=0, dtype="float32"),
        data=DataBundle(dataset=[], collator=lambda batch: batch, iterator_factory=factory),
        optimizer=OptimizerHandle(FakeOptimizer()),
        scheduler=SchedulerHandle(FakeScheduler()),
        loss=lambda logits, labels: FakeLoss(1.0),
        device=object(),
        precision=type("Precision", (), {"autocast": lambda self: nullcontext()})(),
        step_policies=(),
        checkpoint_policies=(),
        eval_policies=(),
        checkpoint=FakeCheckpoint(),
        metrics=FakeMetrics(),
        trainer_config=TrainerConfig(max_steps=1),
    )

    plan = trainer.compile(ctx, services)
    events = list(trainer.run(ctx, plan, 1, skip_batches=4))
    assert [event.step for event in events] == [0]
    # 4 skipped batches + 1 trained batch; StopIteration must not count as a skip.
    assert produced == [0, 1, 2, 3, 4]
    ctx.close()
