"""Rule-based reward plugin for RL smoke tests and simple tasks."""

from __future__ import annotations

from typing import Any

from starweave.seams import PluginBase, RewardConfig


class RuleReward(PluginBase):
    name = "reward-rule"
    provides = ("reward",)
    requires = ()
    config_model = RewardConfig

    def score(self, ctx: Any, samples: list[Any]) -> list[float]:
        rewards: list[float] = []
        for sample in samples:
            output = sample["output"]
            reference = str(sample.get(self.config.reference_field, ""))
            text = output.text
            reward = 0.0
            if reference and reference in text:
                reward += self.config.match_reward
            reward += self.config.length_reward * len(output.response_ids)
            reward += self.config.token_reward * sum(output.response_ids)
            rewards.append(reward)
        return rewards
