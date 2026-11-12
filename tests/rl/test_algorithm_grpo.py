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
        logprobs, logprobs, advantages, torch.ones_like(logprobs)
    )
    assert torch.isfinite(loss)


def test_clipped_ratio_bounds_the_objective() -> None:
    algorithm = GRPO()
    current = torch.full((1, 3), -0.5)
    old = current - 2.0  # ratio = exp(2.0), far outside [0.8, 1.2]
    advantages = torch.ones(1)
    loss = algorithm.policy_loss(current, old, advantages, torch.ones_like(current))
    assert loss.item() == pytest.approx(-(1.0 + algorithm.config.clip_ratio))
