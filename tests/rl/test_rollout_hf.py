from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from starweave.seams import ModelHandle, RolloutConfig  # noqa: E402
from starweave_first_party.rollout_hf import HFRollout  # noqa: E402


class FakeTokenizer:
    def __call__(self, prompt, return_tensors=None):
        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(item)) for item in ids)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, **kwargs):
        continuation = torch.full((input_ids.shape[0], 2), 9, dtype=input_ids.dtype)
        return torch.cat([input_ids, continuation], dim=1)


class FakeCtx:
    def __init__(self, model_handle) -> None:
        self._model = model_handle

    def get(self, name):
        assert name == "model"
        return self._model

    def effect(self, cleanup) -> None:
        return None


def test_rollout_hf_generates_response_ids() -> None:
    model = FakeModel()
    ctx = FakeCtx(
        ModelHandle(
            model=model,
            tokenizer=FakeTokenizer(),
            vocab_size=10,
            param_count=1,
            dtype="float32",
        )
    )
    rollout = HFRollout()
    rollout.setup(ctx, RolloutConfig(max_new_tokens=2, do_sample=False, use_cache=False))
    outputs = rollout.generate(ctx, [{"prompt": "hello"}])
    assert len(outputs) == 1
    assert outputs[0].prompt_ids == (1, 2, 3)
    assert outputs[0].response_ids == (9, 9)
    assert outputs[0].text == "9 9"
