"""Minimal GRPO algorithm plugin.

Advantage estimation is batch-level and pure; policy loss is an
advantage-weighted on-policy logprob objective. It does not import a model or
rollout plugin.
"""

from __future__ import annotations

from typing import Any

from starweave.seams import AlgorithmConfig, PluginBase


class GRPO(PluginBase):
    name = "algorithm-grpo"
    provides = ("algorithm",)
    requires = ()
    config_model = AlgorithmConfig

    def compute_advantages(self, rewards: Any, group_size: int) -> Any:
        import torch

        values = torch.as_tensor(rewards, dtype=torch.float32)
        if values.ndim == 1:
            values = values.reshape(-1, group_size)
        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        return ((values - mean) / std).reshape(-1)

    def policy_loss(
        self,
        logprobs: Any,
        old_logprobs: Any,
        advantages: Any,
        mask: Any,
    ) -> Any:
        import torch

        advantages = advantages.to(logprobs.dtype).reshape(-1, 1)
        mask = mask.to(logprobs.dtype)
        ratio = torch.exp(logprobs - old_logprobs)
        clip = self.config.clip_ratio
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
        objective = torch.min(unclipped, clipped)
        return -((objective * mask).sum() / mask.sum().clamp_min(1.0))
