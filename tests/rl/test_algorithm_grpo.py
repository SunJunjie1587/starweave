from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from starweave_first_party.algorithm_grpo import GRPO  # noqa: E402


def test_group_advantages_are_normalized_and_loss_is_finite() -> None:
    algorithm = GRPO()
    rewards = [[1.0, 2.0], [3.0, 1.0]]
    advantages = algorithm.compute_advantages(rewards, group_size=2)
    assert advantages.shape == (4,)
    group_means = advantages.reshape(2, 2).mean(dim=1)
    assert torch.allclose(group_means, torch.zeros(2), atol=1.0e-6)

    logprobs = torch.log_softmax(torch.randn(4, 5), dim=-1)
    loss = algorithm.policy_loss(
        logprobs, advantages, torch.ones_like(logprobs)
    )
    assert torch.isfinite(loss)
