"""Runtime service construction and execution helpers.

The static plan is device-free; this module is the only place that instantiates
plugins, builds a model, and touches torch-backed services.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..compose.loader import Composition, ResolvedRow
from ..kernel.context import Context, RunPaths
from ..kernel.errors import BootError, ComposeError, RunError
from ..seams.configs import NumericsConfig
from ..seams.types import (
    CheckpointState,
    MetricEvent,
    ModelHandle,
    RLServices,
    TrainingServices,
)
from .compiler import compile_static
from .plan import ExecutionPlan, PhasePlan, StepPlan
from .validator import PINNED_SEAMS, POLICY_SEAMS, REBINDABLE_SEAMS, row_seam

_BOOT_ORDER = (
    "launcher",
    "collective",
    "device",
    "precision",
    "kernel",
    "fsdp",
    "ep",
    "metrics",
    "profiler",
    "checkpoint",
)


@dataclass
class Runtime:
    ctx: Context
    composition: Composition
    services: dict[str, Any] = field(default_factory=dict)
    plugin_instances: list[Any] = field(default_factory=list)
    step_plan: StepPlan | None = None
    trainer: Any = None
    inference: Any = None
    task: Any = None
    rl_services: RLServices | None = None
    static_plan: ExecutionPlan | None = None
    phase: str = "train"
    row_ids: dict[str, str] = field(default_factory=dict)
    training_services: TrainingServices | None = None

    def close(self) -> None:
        self.ctx.close()

    def switch_phase(self, phase: str) -> PhasePlan:
        """Rebind phase-scoped services; pinned services must not change."""
        if self.static_plan is None or phase not in self.static_plan.phases:
            raise BootError(f"phase '{phase}' is not in the compiled plan")
        if phase == self.phase:
            return self.static_plan.phases[phase]
        target = _active_rows(self.composition, phase)
        for seam in PINNED_SEAMS:
            row = target.get(seam)
            if row is None:
                continue
            previous = self.row_ids.get(seam)
            if previous is not None and previous != row.id:
                raise BootError(
                    f"pinned seam '{seam}' cannot be rebound from '{previous}' "
                    f"to '{row.id}'"
                )
        for seam in REBINDABLE_SEAMS:
            row = target.get(seam)
            if row is None or self.row_ids.get(seam) == row.id:
                continue
            old = self.services.get(seam)
            if old is not None:
                _handover(old)
            instance = _instantiate(row, self.ctx)
            self.plugin_instances.append(instance)
            self.services[seam] = instance
            self.ctx.provide(seam, instance)
            if self.training_services is not None:
                if seam == "metrics":
                    self.training_services.metrics = instance
                elif seam == "checkpoint":
                    self.training_services.checkpoint = instance
        self.ctx.emit(
            "phase/exit",
            {
                "phase": self.phase,
                "composition_hash": self.static_plan.phases[self.phase].composition_hash,
            },
        )
        self.phase = phase
        self.row_ids.update(
            {
                seam: row.id
                for seam, row in target.items()
                if seam not in POLICY_SEAMS
            }
        )
        plan = self.static_plan.phases[phase]
        self.ctx.emit(
            "phase/enter",
            {"phase": phase, "composition_hash": plan.composition_hash},
        )
        return plan


def _handover(instance: Any) -> None:
    """Release a rebindable service in the spec's flush/close/release order."""
    for method in ("flush", "close", "release"):
        callback = getattr(instance, method, None)
        if callable(callback):
            callback()


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        payload = {"event": event, "time": time.time(), **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")


def _active_rows(composition: Composition, phase: str) -> dict[str, ResolvedRow]:
    active: dict[str, ResolvedRow] = {}
    for row in composition.rows:
        if row.group or row.disabled:
            continue
        active.setdefault(row_seam(row), row)
    overrides = composition.recipe.phases.get(phase, {})
    for seam, row_id in overrides.items():
        row = composition.by_id.get(row_id)
        if row is None:
            raise ComposeError(f"phase '{phase}' references unknown row '{row_id}'")
        active[seam] = row
    return active


def _policy_rows(composition: Composition) -> dict[str, list[ResolvedRow]]:
    policies: dict[str, list[ResolvedRow]] = {seam: [] for seam in POLICY_SEAMS}
    for row in composition.rows:
        if row.group or row.disabled:
            continue
        seam = row_seam(row)
        if seam in policies:
            policies[seam].append(row)
    return policies


def _instantiate(row: ResolvedRow, ctx: Context) -> Any:
    instance = row.plugin_cls()
    instance.setup(ctx, row.config)
    return instance


def compile_runtime(
    composition: Composition,
    phase: str = "train",
) -> Runtime:
    if phase not in {"train", "eval", "checkpoint", "inference", "rl"}:
        raise BootError(f"unknown runtime phase '{phase}'")
    phase_names = list(composition.recipe.phases)
    if phase not in phase_names:
        phase_names.append(phase)
    static_plan = compile_static(composition, phase_names)
    run_paths = RunPaths(
        run_dir=composition.run_dir,
        checkpoint_dir=composition.run_dir / "checkpoints",
        metrics_path=composition.run_dir / "metrics.jsonl",
    )
    ctx = Context(composition.run_id, paths=run_paths)
    try:
        runtime = _build_runtime(composition, phase, ctx)
        runtime.static_plan = static_plan
        return runtime
    except Exception:
        ctx.close()
        raise


def _build_runtime(composition: Composition, phase: str, ctx: Context) -> Runtime:
    active = _active_rows(composition, phase)
    policy_rows = _policy_rows(composition)
    runtime = Runtime(ctx=ctx, composition=composition, phase=phase)
    runtime.row_ids = {
        seam: row.id for seam, row in active.items() if seam not in POLICY_SEAMS
    }

    for seam in _BOOT_ORDER:
        row = active.get(seam)
        if row is None:
            continue
        instance = _instantiate(row, ctx)
        runtime.plugin_instances.append(instance)
        runtime.services[seam] = instance
        ctx.provide(seam, instance)

    device = runtime.services.get("device")
    if device is not None and hasattr(device, "set_device"):
        device.set_device(int(getattr(device, "index", 0)))

    model_row = active.get("model")
    if model_row is None:
        raise BootError("runtime plan has no model provider")
    model_builder = _instantiate(model_row, ctx)
    runtime.plugin_instances.append(model_builder)
    model_handle = model_builder.build(ctx, model_row.config)

    fsdp = runtime.services.get("fsdp")
    if fsdp is not None:
        wrapped = fsdp.apply(ctx, model_handle.model, model_handle.parallel_plan)
        if wrapped is not model_handle.model:
            model_handle = ModelHandle(
                model=wrapped,
                tokenizer=model_handle.tokenizer,
                vocab_size=model_handle.vocab_size,
                param_count=model_handle.param_count,
                dtype=model_handle.dtype,
                parallel_plan=model_handle.parallel_plan,
            )
    ep = runtime.services.get("ep")
    if ep is not None:
        wrapped = ep.apply(ctx, model_handle.model, model_handle.parallel_plan)
        if wrapped is not model_handle.model:
            model_handle = ModelHandle(
                model=wrapped,
                tokenizer=model_handle.tokenizer,
                vocab_size=model_handle.vocab_size,
                param_count=model_handle.param_count,
                dtype=model_handle.dtype,
                parallel_plan=model_handle.parallel_plan,
            )
    runtime.services["model"] = model_handle
    ctx.provide("model", model_handle)

    if phase == "rl":
        return _build_rl_runtime(runtime, active, ctx, model_handle)

    if phase == "inference":
        inference_row = active.get("inference")
        if inference_row is None:
            raise BootError("inference phase has no inference provider")
        inference = _instantiate(inference_row, ctx)
        runtime.plugin_instances.append(inference)
        inference.load(ctx, model_handle, inference_row.config)
        runtime.services["inference"] = inference
        runtime.inference = inference
        return runtime

    opt_row = active.get("optimizer")
    if opt_row is None:
        raise BootError("runtime plan has no optimizer provider")
    optimizer_factory = _instantiate(opt_row, ctx)
    runtime.plugin_instances.append(optimizer_factory)
    runtime.services["optimizer"] = optimizer_factory.build(ctx, model_handle, opt_row.config)
    ctx.provide("optimizer", runtime.services["optimizer"])

    sched_row = active.get("scheduler")
    if sched_row is None:
        raise BootError("runtime plan has no scheduler provider")
    scheduler_factory = _instantiate(sched_row, ctx)
    runtime.plugin_instances.append(scheduler_factory)
    trainer_row = active.get("trainer")
    total_steps = int(getattr(trainer_row.config, "max_steps", 1000)) if trainer_row else 1000
    runtime.services["scheduler"] = scheduler_factory.build(
        ctx, runtime.services["optimizer"], total_steps, sched_row.config
    )
    ctx.provide("scheduler", runtime.services["scheduler"])

    loss_row = active.get("loss")
    if loss_row is None:
        raise BootError("runtime plan has no loss provider")
    loss_factory = _instantiate(loss_row, ctx)
    runtime.plugin_instances.append(loss_factory)
    runtime.services["loss"] = loss_factory.build(ctx, loss_row.config)
    ctx.provide("loss", runtime.services["loss"])

    data_row = active.get("data")
    if data_row is None:
        raise BootError("runtime plan has no data provider")
    data_builder = _instantiate(data_row, ctx)
    runtime.plugin_instances.append(data_builder)
    runtime.services["data"] = data_builder.build(ctx, data_row.config)
    ctx.provide("data", runtime.services["data"])

    step_policies = []
    for row in sorted(
        policy_rows["step_policy"],
        key=lambda item: getattr(item.plugin_cls, "priority", 100),
    ):
        instance = _instantiate(row, ctx)
        runtime.plugin_instances.append(instance)
        step_policies.append(instance)
    checkpoint_policies = []
    for row in policy_rows["checkpoint_policy"]:
        instance = _instantiate(row, ctx)
        runtime.plugin_instances.append(instance)
        checkpoint_policies.append(instance)
    eval_policies = []
    for row in policy_rows["eval_policy"]:
        instance = _instantiate(row, ctx)
        runtime.plugin_instances.append(instance)
        eval_policies.append(instance)

    if trainer_row is None:
        raise BootError("training phase has no trainer provider")
    trainer = _instantiate(trainer_row, ctx)
    runtime.plugin_instances.append(trainer)
    runtime.services["trainer"] = trainer
    ctx.provide("trainer", trainer)
    training_services = TrainingServices(
        model=model_handle,
        data=runtime.services["data"],
        optimizer=runtime.services["optimizer"],
        scheduler=runtime.services["scheduler"],
        loss=runtime.services["loss"],
        device=runtime.services["device"],
        precision=runtime.services["precision"],
        step_policies=tuple(step_policies),
        checkpoint_policies=tuple(checkpoint_policies),
        eval_policies=tuple(eval_policies),
        checkpoint=runtime.services["checkpoint"],
        metrics=runtime.services["metrics"],
        trainer_config=trainer_row.config,
        composition_hash=composition.composition_hash,
    )
    runtime.step_plan = trainer.compile(ctx, training_services)
    runtime.training_services = training_services
    runtime.trainer = trainer
    return runtime


def _build_rl_runtime(
    runtime: Runtime,
    active: dict[str, ResolvedRow],
    ctx: Context,
    model_handle: ModelHandle,
) -> Runtime:
    def require(seam: str) -> ResolvedRow:
        row = active.get(seam)
        if row is None:
            raise BootError(f"rl phase has no {seam} provider")
        return row

    optimizer_row = require("optimizer")
    optimizer_builder = _instantiate(optimizer_row, ctx)
    runtime.plugin_instances.append(optimizer_builder)
    optimizer = optimizer_builder.build(ctx, model_handle, optimizer_row.config)
    runtime.services["optimizer"] = optimizer
    ctx.provide("optimizer", optimizer)

    scheduler_row = require("scheduler")
    scheduler_builder = _instantiate(scheduler_row, ctx)
    runtime.plugin_instances.append(scheduler_builder)
    task_row = require("task")
    total_steps = int(getattr(task_row.config, "max_steps", 100))
    scheduler = scheduler_builder.build(
        ctx, optimizer, total_steps, scheduler_row.config
    )
    runtime.services["scheduler"] = scheduler
    ctx.provide("scheduler", scheduler)

    data_row = require("data")
    data_builder = _instantiate(data_row, ctx)
    runtime.plugin_instances.append(data_builder)
    data = data_builder.build(ctx, data_row.config)
    runtime.services["data"] = data
    ctx.provide("data", data)

    for seam in ("rollout", "reward", "algorithm"):
        row = require(seam)
        instance = _instantiate(row, ctx)
        runtime.plugin_instances.append(instance)
        runtime.services[seam] = instance
        ctx.provide(seam, instance)

    task_config = task_row.config
    rl_services = RLServices(
        model=model_handle,
        data=data,
        rollout=runtime.services["rollout"],
        reward=runtime.services["reward"],
        algorithm=runtime.services["algorithm"],
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint=runtime.services["checkpoint"],
        metrics=runtime.services["metrics"],
        device=runtime.services["device"],
        precision=runtime.services["precision"],
        task_config=task_config,
        numerics=NumericsConfig(),
        composition_hash=runtime.composition.composition_hash,
    )
    task = task_row.plugin_cls()
    task.setup(ctx, rl_services)
    runtime.plugin_instances.append(task)
    runtime.services["task"] = task
    ctx.provide("task", task)
    runtime.task = task
    runtime.rl_services = rl_services
    return runtime


def apply_resume(
    runtime: Runtime, path: str, *, allow_config_change: bool = False
) -> tuple[int, int]:
    checkpoint = runtime.services["checkpoint"]
    state = checkpoint.load(runtime.ctx, path)
    if (
        state.composition_hash
        and state.composition_hash != runtime.composition.composition_hash
        and not allow_config_change
    ):
        raise RunError(
            "composition changed since the checkpoint; pass --allow-config-change "
            "only for non-structural changes"
        )
    model = runtime.services["model"].model
    if state.model_state:
        model.load_state_dict(state.model_state)
    runtime.services["optimizer"].load_state_dict(state.optimizer)
    runtime.services["scheduler"].load_state_dict(state.scheduler)
    try:
        import torch

        if "torch" in state.rng:
            torch.set_rng_state(state.rng["torch"])
        if "npu" in state.rng and hasattr(torch, "npu"):
            torch.npu.set_rng_state(state.rng["npu"])
    except Exception:  # noqa: BLE001
        pass
    # The checkpoint is written after step `state.step` has completed, and
    # `dataloader_position` is the index of the last consumed micro-batch.
    return state.step + 1, state.dataloader_position + 1


def _move_batch(batch: dict[str, Any], device_type: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device_type) if hasattr(value, "to") else value
    return moved


def _rng_state() -> dict[str, Any]:
    try:
        import torch

        state: dict[str, Any] = {"torch": torch.get_rng_state()}
        if hasattr(torch, "npu"):
            state["npu"] = torch.npu.get_rng_state()
        return state
    except Exception:  # noqa: BLE001
        return {}


def evaluate(
    runtime: Runtime,
    *,
    max_batches: int = 4,
    step: int = 0,
) -> tuple[float, int]:
    """Run a no-grad evaluation pass over the data seam."""
    import torch

    services = runtime.training_services
    if services is None:
        raise BootError("eval phase requires training services")
    model = services.model.model
    device_type = getattr(services.device, "device_type", None)
    iterator = services.data.iterator_factory()
    total = 0.0
    batches = 0
    model.eval()
    try:
        with torch.no_grad():
            while batches < max_batches:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = services.data.iterator_factory()
                    continue
                if device_type:
                    batch = _move_batch(batch, device_type)
                with services.precision.autocast():
                    outputs = model(**batch)
                    loss = getattr(outputs, "loss", None)
                    if loss is None:
                        loss = services.loss(
                            getattr(outputs, "logits", outputs), batch["labels"]
                        )
                value = float(loss.detach().item())
                total += value
                batches += 1
                services.metrics.emit(
                    runtime.ctx,
                    MetricEvent(
                        name="eval_loss",
                        value=value,
                        step=step,
                        phase="eval",
                        timestamp=time.time(),
                    ),
                )
    finally:
        model.train()
    return (total / batches if batches else 0.0), batches


def save_snapshot(
    runtime: Runtime,
    *,
    step: int = 0,
    dataloader_position: int = 0,
) -> Path:
    """Save a checkpoint from the current runtime services (checkpoint phase)."""
    services = runtime.training_services
    if services is None:
        raise BootError("checkpoint phase requires training services")
    state = CheckpointState(
        model_dir=runtime.ctx.paths.checkpoint_dir / f"step-{step:06d}",
        optimizer=services.optimizer.state_dict(),
        scheduler=services.scheduler.state_dict(),
        step=step,
        dataloader_position=dataloader_position,
        rng=_rng_state(),
        composition_hash=runtime.composition.composition_hash,
        config_snapshot={"composition_hash": runtime.composition.composition_hash},
        model_state=services.model.model.state_dict(),
    )
    path = services.checkpoint.save(runtime.ctx, state)
    runtime.ctx.emit("ckpt/save", {"step": step, "path": str(path)})
    return path
