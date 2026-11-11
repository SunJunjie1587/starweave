"""Step, checkpoint, and eval policies."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from starweave.seams import PluginBase
from starweave.seams.configs import ConfigBase
from starweave.seams.protocols import BaseCheckpointPolicy, BaseEvalPolicy, BaseStepPolicy
from starweave.seams.types import RunState, StepState


class GradClipConfig(ConfigBase):
    max_norm: float = Field(default=1.0, gt=0)


class GradClipPolicy(PluginBase, BaseStepPolicy):
    name = "policy-grad-clip"
    provides = ("step_policy",)
    requires = ()
    config_model = GradClipConfig
    priority = 10

    def after_backward(self, ctx: Any, state: StepState) -> None:
        if not state.should_step:
            return
        import torch

        model = ctx.get("model").model
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_norm)


class NanGuardConfig(ConfigBase):
    action: Literal["abort", "skip"] = "abort"
    threshold: float = 1.0e-12


class NanGuardPolicy(PluginBase, BaseStepPolicy):
    name = "policy-nan-guard"
    provides = ("step_policy",)
    requires = ()
    config_model = NanGuardConfig
    priority = 20

    def after_backward(self, ctx: Any, state: StepState) -> None:
        import math

        if state.loss is None or math.isfinite(state.loss):
            return
        if self.config.action == "abort":
            state.should_abort = True
            raise RuntimeError(f"non-finite loss at step {state.step}: {state.loss}")
        state.should_step = False


class GradAccumConfig(ConfigBase):
    steps: int = Field(default=1, gt=0)


class GradAccumPolicy(PluginBase, BaseStepPolicy):
    name = "policy-grad-accum"
    provides = ("step_policy",)
    requires = ()
    config_model = GradAccumConfig
    priority = 30

    def before_step(self, ctx: Any, state: StepState) -> None:
        steps = self.config.steps
        state.accum_steps = steps
        state.loss_scale = 1.0 / steps
        state.should_step = (state.micro_step + 1) % steps == 0


class EveryNConfig(ConfigBase):
    n: int = Field(default=200, gt=0)


class EveryNCheckpointPolicy(PluginBase, BaseCheckpointPolicy):
    name = "checkpoint-policy-every-n"
    provides = ("checkpoint_policy",)
    requires = ()
    config_model = EveryNConfig
    priority = 100

    def should_save(self, ctx: Any, state: RunState) -> bool:
        return state.step > 0 and state.step % self.config.n == 0


class EveryNEvalPolicy(PluginBase, BaseEvalPolicy):
    name = "eval-policy-every-n"
    provides = ("eval_policy",)
    requires = ()
    config_model = EveryNConfig
    priority = 100

    def should_eval(self, ctx: Any, state: RunState) -> bool:
        return state.step > 0 and state.step % self.config.n == 0
