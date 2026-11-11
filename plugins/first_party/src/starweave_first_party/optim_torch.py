"""Optimizer, scheduler, and loss plugins. Build logic lands in M1."""

from __future__ import annotations

from typing import Any

from starweave.seams import PluginBase
from starweave.seams.configs import LossConfig, OptimizerConfig, SchedulerConfig
from starweave.seams.types import ModelHandle, OptimizerHandle, SchedulerHandle


class AdamWOptimizer(PluginBase):
    name = "optimizer-adamw"
    provides = ("optimizer",)
    requires = ("model",)
    config_model = OptimizerConfig

    def build(self, ctx: Any, model: ModelHandle, config: OptimizerConfig) -> OptimizerHandle:
        import torch

        decay, no_decay = [], []
        for name, parameter in model.model.named_parameters():
            if not parameter.requires_grad:
                continue
            bucket = no_decay if any(
                keyword in name.lower() for keyword in config.no_decay_keywords
            ) else decay
            bucket.append(parameter)
        groups = [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
        )
        return OptimizerHandle(optimizer)


class CosineScheduler(PluginBase):
    name = "scheduler-cosine"
    provides = ("scheduler",)
    requires = ("optimizer",)
    config_model = SchedulerConfig

    def build(
        self,
        ctx: Any,
        optimizer: OptimizerHandle,
        total_steps: int,
        config: SchedulerConfig,
    ) -> SchedulerHandle:
        import math

        import torch

        warmup = int(total_steps * config.warmup_ratio)

        def factor(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

        return SchedulerHandle(torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, factor))


class LinearScheduler(CosineScheduler):
    name = "scheduler-linear"

    def build(
        self,
        ctx: Any,
        optimizer: OptimizerHandle,
        total_steps: int,
        config: SchedulerConfig,
    ) -> SchedulerHandle:
        import torch

        warmup = int(total_steps * config.warmup_ratio)

        def factor(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * (1.0 - progress)

        return SchedulerHandle(torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, factor))


class CrossEntropyLoss(PluginBase):
    name = "loss-ce"
    provides = ("loss",)
    requires = ()
    config_model = LossConfig

    def build(self, ctx: Any, config: LossConfig):
        import torch.nn.functional as function

        def loss_fn(logits: Any, labels: Any) -> Any:
            return function.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=config.ignore_index,
            )

        return loss_fn
