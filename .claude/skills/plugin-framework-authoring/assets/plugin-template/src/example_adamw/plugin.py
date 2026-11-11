from __future__ import annotations

from typing import Any

# The only framework import a plugin may use.
from framework.seams.optimizer import Optimizer
from .config import AdamWConfig


class AdamWPlugin:
    name = "adamw"
    provides = ("optimizer",)
    requires = ("device", "model")
    config_model = AdamWConfig
    rebindable = False
    priority = 100

    def setup(self, ctx: Any, config: AdamWConfig) -> None:
        self.ctx = ctx
        self.config = config
        self._optimizer = None
        ctx.effect(self.teardown)

    def build(self, model):
        import torch

        self._optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )
        return self._optimizer

    def step(self) -> None:
        self._require_optimizer().step()

    def zero_grad(self) -> None:
        self._require_optimizer().zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, Any]:
        return self._require_optimizer().state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._require_optimizer().load_state_dict(state)

    def teardown(self) -> None:
        self._optimizer = None

    def _require_optimizer(self):
        if self._optimizer is None:
            raise RuntimeError("AdamWPlugin.build() must be called before use")
        return self._optimizer
