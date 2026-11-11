from __future__ import annotations

import pytest

from starweave.seams import NumericsConfig

torch = pytest.importorskip("torch")


def test_numerics_config_defaults() -> None:
    config = NumericsConfig()
    assert config.logits_dtype == "float32"
    assert config.recompute_logprobs is True
    assert config.tp == 1
    assert config.ep == 1


def test_compare_logprobs_identical_and_different() -> None:
    from starweave_first_party.rl_logprobs import compare_logprobs

    left = torch.log_softmax(torch.randn(2, 5, 11), dim=-1)
    same = compare_logprobs(left, left)
    assert same["max_abs"] == 0.0
    assert same["kl"] == pytest.approx(0.0)

    right = torch.log_softmax(torch.randn(2, 5, 11), dim=-1)
    diff = compare_logprobs(left, right)
    assert diff["max_abs"] > 0.0
    assert diff["kl"] > 0.0
