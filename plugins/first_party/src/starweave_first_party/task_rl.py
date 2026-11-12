"""Minimal on-policy GRPO task.

Rollout only produces tokens. All logprobs used by the loss are recomputed by
the training model's full-sequence forward path (scheme C), so the RL objective
does not depend on the rollout engine's numerical path.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

from starweave.seams import (
    MetricEvent,
    PluginBase,
    RLServices,
    StepEvent,
    TaskConfig,
)

from .rl_logprobs import causal_lm_logprobs


class RLTask(PluginBase):
    name = "task-rl"
    mode = "rl"
    provides = ("task",)
    requires = (
        "device",
        "precision",
        "model",
        "data",
        "rollout",
        "reward",
        "algorithm",
        "optimizer",
        "scheduler",
        "checkpoint",
        "metrics",
    )
    config_model = TaskConfig

    def setup(self, ctx: Any, services: RLServices) -> None:
        self.ctx = ctx
        self.services = services
        self.config = services.task_config
        ctx.effect(self.teardown)

    def run(self, ctx: Any) -> Iterator[StepEvent]:
        import torch

        services = self.services
        model_handle = services.model
        model = model_handle.model
        tokenizer = model_handle.tokenizer
        device_type = getattr(services.device, "device_type", "cpu")
        pad_id = (
            getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None)
            or 0
        )
        iterator = services.data.iterator_factory()
        model.train()
        for step in range(self.config.max_steps):
            started = time.perf_counter()
            try:
                rows = next(iterator)
            except StopIteration:
                iterator = services.data.iterator_factory()
                rows = next(iterator)
            rows = list(rows)[: self.config.prompts_per_step]

            outputs: list[Any] = []
            group_rewards: list[list[float]] = []
            sequences: list[tuple[list[int], list[int]]] = []
            for row in rows:
                prompt = str(row.get("prompt", row.get("question", "")))
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                requests = [{"prompt": prompt} for _ in range(self.config.group_size)]
                generated = services.rollout.generate(ctx, requests)
                samples = [
                    {**row, "output": output, "reference": row.get("reference", "")}
                    for output in generated
                ]
                rewards = services.reward.score(ctx, samples)
                group_rewards.append(rewards)
                for output in generated:
                    outputs.append(output)
                    sequences.append((list(prompt_ids), list(output.response_ids)))

            advantages = services.algorithm.compute_advantages(
                group_rewards, self.config.group_size
            ).to(device_type)

            max_len = max(len(prompt) + len(response) for prompt, response in sequences)
            input_ids = torch.full(
                (len(sequences), max_len), pad_id, dtype=torch.long, device=device_type
            )
            attention = torch.zeros_like(input_ids)
            response_mask = torch.zeros_like(input_ids)
            for index, (prompt_ids, response_ids) in enumerate(sequences):
                total = prompt_ids + response_ids
                input_ids[index, : len(total)] = torch.tensor(
                    total, dtype=torch.long, device=device_type
                )
                attention[index, : len(total)] = 1
                response_mask[index, len(prompt_ids) : len(total)] = 1

            mask = response_mask[:, 1:]
            with torch.no_grad(), services.precision.autocast():
                old_logprobs = causal_lm_logprobs(
                    model, input_ids, attention, logits_dtype="float32"
                )
            value = 0.0
            clip_fraction = 0.0
            clip_ratio = float(getattr(services.algorithm.config, "clip_ratio", 0.2))
            for _ in range(max(1, self.config.update_epochs)):
                with services.precision.autocast():
                    logprobs = causal_lm_logprobs(
                        model, input_ids, attention, logits_dtype="float32"
                    )
                    loss = services.algorithm.policy_loss(
                        logprobs, old_logprobs, advantages, mask
                    )
                loss.backward()
                services.optimizer.step()
                services.scheduler.step()
                services.optimizer.zero_grad()
                value = float(loss.detach().item())
                with torch.no_grad():
                    ratio = torch.exp(logprobs.detach() - old_logprobs)
                    valid = mask.bool()
                    if valid.any():
                        outside = (ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)
                        clip_fraction = float(outside[valid].float().mean().item())
            services.rollout.update_weights(ctx, model_handle)
            reward_mean = float(
                sum(sum(group) for group in group_rewards) / max(1, len(outputs))
            )
            services.metrics.emit(
                ctx,
                MetricEvent(
                    name="rl_loss",
                    value=value,
                    step=step,
                    phase="rl",
                    timestamp=time.time(),
                ),
            )
            services.metrics.emit(
                ctx,
                MetricEvent(
                    name="rl_reward",
                    value=reward_mean,
                    step=step,
                    phase="rl",
                    timestamp=time.time(),
                ),
            )
            services.metrics.emit(
                ctx,
                MetricEvent(
                    name="rl_clip_fraction",
                    value=clip_fraction,
                    step=step,
                    phase="rl",
                    timestamp=time.time(),
                ),
            )
            lr = 0.0
            try:
                lr = float(services.scheduler.scheduler.get_last_lr()[0])
            except Exception:  # noqa: BLE001
                pass
            event = StepEvent(
                step=step,
                loss=value,
                lr=lr,
                tokens=int(input_ids.numel()),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            ctx.emit("step/end", event)
            yield event
