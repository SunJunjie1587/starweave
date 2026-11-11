from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.randn(1, 4, 7))

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        return SimpleNamespace(logits=self.logits[:, :length, :].expand(batch, -1, -1))


def test_causal_lm_logprobs_matches_manual_gather() -> None:
    from starweave_first_party.rl_logprobs import causal_lm_logprobs, masked_mean

    model = TinyLM()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    logprobs = causal_lm_logprobs(model, input_ids)
    manual = torch.log_softmax(model.logits[0, :3], dim=-1).gather(
        -1, input_ids[0, 1:].unsqueeze(-1)
    ).squeeze(-1).unsqueeze(0)
    assert logprobs.shape == (1, 3)
    assert torch.allclose(logprobs, manual)
    assert masked_mean(logprobs, torch.ones_like(logprobs)).item() == pytest.approx(
        logprobs.mean().item()
    )
