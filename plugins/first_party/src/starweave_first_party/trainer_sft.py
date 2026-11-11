"""SFT trainer plugin: compile a static step plan and run the training loop."""

from __future__ import annotations

import time
from typing import Any, Iterator

from starweave.seams import StepPlan
from starweave.seams import PluginBase
from starweave.seams.configs import TrainerConfig
from starweave.seams.types import (
    CheckpointState,
    MetricEvent,
    RunState,
    StepEvent,
    StepState,
    TrainingServices,
)


class SFTTrainer(PluginBase):
    name = "trainer-sft"
    provides = ("trainer",)
    requires = (
        "device",
        "model",
        "data",
        "optimizer",
        "scheduler",
        "loss",
        "collective",
        "fsdp",
        "ep",
    )
    config_model = TrainerConfig

    def setup(self, ctx: Any, config: TrainerConfig) -> None:
        super().setup(ctx, config)
        self._services: TrainingServices | None = None
        self._last_saved_step = -1

    def compile(self, ctx: Any, services: TrainingServices) -> StepPlan:
        self._services = services

        def before_step(state: StepState) -> None:
            for policy in services.step_policies:
                policy.before_step(ctx, state)

        def forward_backward(state: StepState, batch: dict[str, Any]) -> float:
            model = services.model.model
            device_type = getattr(services.device, "device_type", None)
            if device_type:
                batch = self._move_batch(batch, device_type)
            with services.precision.autocast():
                outputs = model(**batch)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    logits = getattr(outputs, "logits", outputs)
                    loss = services.loss(logits, batch["labels"])
            (loss * state.loss_scale).backward()
            state.loss = float(loss.detach().item())
            return state.loss

        def after_backward(state: StepState) -> None:
            for policy in services.step_policies:
                policy.after_backward(ctx, state)

        def optimizer_step(state: StepState) -> None:
            if state.should_step:
                services.optimizer.step()
                services.scheduler.step()
                services.optimizer.zero_grad()

        def after_step(state: StepState) -> None:
            for policy in services.step_policies:
                policy.after_step(ctx, state)

        def on_error(state: StepState) -> None:
            for policy in services.step_policies:
                policy.on_error(ctx, state)

        return StepPlan(
            before_step=(before_step,),
            forward_backward=forward_backward,
            after_backward=(after_backward,),
            optimizer_step=optimizer_step,
            after_step=(after_step,),
            on_error=(on_error,),
        )

    def run(
        self,
        ctx: Any,
        plan: StepPlan,
        max_steps: int,
        *,
        start_step: int = 0,
        skip_batches: int = 0,
    ) -> Iterator[StepEvent]:
        if self._services is None:
            raise RuntimeError("SFTTrainer.compile() must be called before run()")
        services = self._services
        iterator = services.data.iterator_factory()
        skipped = 0
        while skipped < skip_batches:
            try:
                next(iterator)
            except StopIteration:
                iterator = services.data.iterator_factory()
                continue
            skipped += 1

        step = start_step
        micro = skip_batches
        epoch = 0
        max_micro = max_steps * 64 + skip_batches + 64
        while step < max_steps:
            if micro > max_micro:
                raise RuntimeError(f"SFTTrainer did not reach max_steps={max_steps}")
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                iterator = services.data.iterator_factory()
                continue

            state = StepState(step=step, micro_step=micro, batch=batch)
            started = time.perf_counter()
            for hook in plan.before_step:
                hook(state)
            try:
                if plan.forward_backward is None:
                    raise RuntimeError("step plan has no forward_backward")
                plan.forward_backward(state, batch)
            except Exception as exc:  # noqa: BLE001
                state.error = exc
                for hook in plan.on_error:
                    hook(state)
                if state.should_abort:
                    raise RuntimeError(f"training aborted at step {step}: {exc}") from exc
                raise
            for hook in plan.after_backward:
                hook(state)
            if plan.optimizer_step is not None:
                plan.optimizer_step(state)
            for hook in plan.after_step:
                hook(state)

            if state.should_step:
                event = StepEvent(
                    step=step,
                    loss=state.loss or 0.0,
                    lr=self._current_lr(services),
                    tokens=self._token_count(batch),
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
                ctx.emit("step/end", event)
                services.metrics.emit(
                    ctx,
                    MetricEvent(
                        name="loss",
                        value=event.loss,
                        step=step,
                        phase="train",
                        timestamp=time.time(),
                    ),
                )
                if step != self._last_saved_step:
                    run_state = RunState(
                        run_id=ctx.run_id, phase="train", step=step, epoch=epoch
                    )
                    if any(
                        policy.should_save(ctx, run_state)
                        for policy in services.checkpoint_policies
                    ):
                        self._save_checkpoint(ctx, services, step, micro)
                        self._last_saved_step = step
                step += 1
                micro += 1
                yield event
            else:
                micro += 1

    def _current_lr(self, services: TrainingServices) -> float:
        try:
            return float(services.scheduler.scheduler.get_last_lr()[0])
        except Exception:  # noqa: BLE001
            return 0.0

    @staticmethod
    def _move_batch(batch: dict[str, Any], device_type: str) -> dict[str, Any]:
        """Move tensor batch fields to the device seam's device type."""
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(device_type) if hasattr(value, "to") else value
        return moved

    def _token_count(self, batch: dict[str, Any]) -> int:
        input_ids = batch.get("input_ids")
        if input_ids is None:
            return 0
        try:
            return int(input_ids.numel())
        except Exception:  # noqa: BLE001
            return 0

    def _rng_state(self) -> dict[str, Any]:
        try:
            import torch

            state: dict[str, Any] = {"torch": torch.get_rng_state()}
            if hasattr(torch, "npu"):
                state["npu"] = torch.npu.get_rng_state()
            return state
        except Exception:  # noqa: BLE001
            return {}

    def _save_checkpoint(
        self, ctx: Any, services: TrainingServices, step: int, micro: int
    ) -> None:
        model_state = services.model.model.state_dict()
        state = CheckpointState(
            model_dir=ctx.paths.checkpoint_dir / f"step-{step:06d}",
            optimizer=services.optimizer.state_dict(),
            scheduler=services.scheduler.state_dict(),
            step=step,
            dataloader_position=micro,
            rng=self._rng_state(),
            composition_hash=services.composition_hash,
            config_snapshot={
                "trainer": services.trainer_config.model_dump(),
                "composition_hash": services.composition_hash,
            },
            model_state=model_state,
        )
        path = services.checkpoint.save(ctx, state)
        ctx.emit("ckpt/save", {"step": step, "path": str(path)})
