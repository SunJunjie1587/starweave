from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from starweave.kernel.context import Context  # noqa: E402
from starweave.seams import (  # noqa: E402
    DataBundle,
    ModelHandle,
    OptimizerHandle,
    RLServices,
    RolloutOutput,
    SchedulerHandle,
    TaskConfig,
)
from starweave_first_party.algorithm_grpo import GRPO  # noqa: E402
from starweave_first_party.task_rl import RLTask  # noqa: E402


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.randn(1, 8, 11))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch, length = input_ids.shape
        return SimpleNamespace(logits=self.logits[:, :length, :].expand(batch, -1, -1))


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompt, add_special_tokens=False):
        return {"input_ids": [1, 2]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(item)) for item in ids)


class FakeRollout:
    def generate(self, ctx, requests):
        return [
            RolloutOutput(prompt_ids=(1, 2), response_ids=(3, 4), text="a"),
            RolloutOutput(prompt_ids=(1, 2), response_ids=(5, 6, 7), text="b"),
        ]

    def update_weights(self, ctx, model):
        return None


class FakeReward:
    def score(self, ctx, samples):
        return [1.0, 2.0]


class FakeScheduler:
    def step(self):
        return None

    def get_last_lr(self):
        return [0.1]


class FakeCheckpoint:
    def save(self, ctx, state):
        return Path("/tmp/rl")


class FakeMetrics:
    def __init__(self):
        self.events = []

    def emit(self, ctx, event):
        self.events.append(event)


def test_rl_task_recomputes_logprobs_and_updates_model() -> None:
    model = TinyLM()
    before = model.logits.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    metrics = FakeMetrics()
    ctx = Context("rl-task-test")
    services = RLServices(
        model=ModelHandle(
            model=model,
            tokenizer=TinyTokenizer(),
            vocab_size=11,
            param_count=1,
            dtype="float32",
        ),
        data=DataBundle(
            dataset=[{"prompt": "hello", "reference": "a"}],
            collator=lambda batch: batch,
            iterator_factory=lambda: iter([[{"prompt": "hello", "reference": "a"}]]),
        ),
        rollout=FakeRollout(),
        reward=FakeReward(),
        algorithm=GRPO(),
        optimizer=OptimizerHandle(optimizer),
        scheduler=SchedulerHandle(FakeScheduler()),
        checkpoint=FakeCheckpoint(),
        metrics=metrics,
        device=SimpleNamespace(device_type="cpu"),
        precision=SimpleNamespace(autocast=lambda: nullcontext()),
        task_config=TaskConfig(
            max_steps=1, prompts_per_step=1, group_size=2, update_epochs=2
        ),
    )
    task = RLTask()
    task.setup(ctx, services)
    events = list(task.run(ctx))
    assert len(events) == 1
    assert torch.isfinite(torch.tensor(events[0].loss))
    assert not torch.allclose(before, model.logits.detach())
    assert [event.name for event in metrics.events] == [
        "rl_loss",
        "rl_reward",
        "rl_clip_fraction",
    ]
    ctx.close()
