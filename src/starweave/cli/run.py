"""run/infer entry points backed by the runtime plan."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from ..compose.loader import load_composition
from ..compose.recipe import load_recipe
from ..kernel.errors import BootError, RunError
from ..plan.runtime import (
    RunLog,
    apply_resume,
    compile_runtime,
    evaluate,
    save_snapshot,
)


def _latest_checkpoint(root: Path = Path("runs")) -> Path | None:
    candidates = list(root.glob("*/checkpoints/step-*"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_run_id(resume: str | None, run_id: str | None) -> str:
    if run_id:
        return run_id
    if resume:
        if resume == "latest":
            latest = _latest_checkpoint()
            if latest is None:
                raise RunError("--resume latest requested but no checkpoint exists")
            return latest.parent.parent.name
        path = Path(resume)
        if path.parent.name == "checkpoints":
            return path.parent.parent.name
    return time.strftime("run-%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def run(
    recipe_path: str,
    *,
    phase: str | None = None,
    set_values: list[str] | None = None,
    patch_paths: list[str] | None = None,
    resume: str | None = None,
    allow_config_change: bool = False,
    run_id: str | None = None,
) -> int:
    phase = phase or "train"
    resolved_run_id = _resolve_run_id(resume, run_id)
    recipe = load_recipe(recipe_path)
    composition = load_composition(
        recipe,
        run_id=resolved_run_id,
        set_values=set_values,
        patch_paths=patch_paths,
    )
    if phase not in {"train", "eval", "checkpoint", "rl"}:
        raise BootError(f"run supports train/eval/checkpoint/rl, got '{phase}'")
    runtime = compile_runtime(composition, "rl" if phase == "rl" else "train")
    run_log = RunLog(composition.run_dir / "run.jsonl")
    run_log.emit(
        "run/start",
        run_id=resolved_run_id,
        recipe=recipe_path,
        composition_hash=composition.composition_hash,
        phase=phase,
    )
    runtime.ctx.on("step/end", lambda event: run_log.emit("step/end", **event.__dict__))
    runtime.ctx.on("ckpt/save", lambda payload: run_log.emit("ckpt/save", **payload))
    runtime.ctx.on("phase/exit", lambda payload: run_log.emit("phase/exit", **payload))

    start_step = 0
    skip_batches = 0
    if resume:
        start_step, skip_batches = apply_resume(
            runtime, resume, allow_config_change=allow_config_change
        )
    if runtime.static_plan is None:
        raise BootError("runtime has no static plan")
    phase_plan = runtime.static_plan.phases["rl" if phase == "rl" else "train"]
    if phase not in {"train", "rl"}:
        phase_plan = runtime.switch_phase(phase)
    run_log.emit(
        "phase/enter",
        phase=phase,
        composition_hash=phase_plan.composition_hash,
        step=start_step,
    )
    trainer = runtime.services.get("trainer")
    if phase != "rl" and (trainer is None or runtime.step_plan is None):
        raise BootError(f"phase '{phase}' has no trainer")
    if phase == "rl":
        if runtime.rl_services is None:
            raise BootError("rl phase has no task services")
        max_steps = int(runtime.rl_services.task_config.max_steps)
    else:
        max_steps = int(trainer.config.max_steps)

    def body() -> int:
        if phase == "rl":
            for _ in runtime.task.run(runtime.ctx):
                pass
            return 0
        if phase == "eval":
            loss, batches = evaluate(runtime, max_batches=4, step=start_step)
            run_log.emit("eval/end", loss=loss, batches=batches, step=start_step)
            return 0
        if phase == "checkpoint":
            save_snapshot(
                runtime, step=start_step, dataloader_position=skip_batches
            )
            return 0
        for _ in trainer.run(
            runtime.ctx,
            runtime.step_plan,
            max_steps,
            start_step=start_step,
            skip_batches=skip_batches,
        ):
            pass
        return 0

    try:
        launcher = runtime.services.get("launcher")
        result = launcher.run(runtime.ctx, body) if launcher is not None else body()
    finally:
        runtime.close()
    run_log.emit(
        "run/end",
        status="ok",
        phase=phase,
        steps=max_steps if phase in {"train", "rl"} else 0,
    )
    return int(result)


def infer(
    recipe_path: str,
    *,
    prompt: str,
    set_values: list[str] | None = None,
    patch_paths: list[str] | None = None,
    run_id: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    checkpoint: str | None = None,
) -> int:
    resolved_run_id = run_id or "infer"
    overrides = list(set_values or [])
    overrides.extend(
        [
            f"inference.max_new_tokens={max_new_tokens}",
            f"inference.temperature={temperature}",
            f"inference.top_p={top_p}",
        ]
    )
    if checkpoint:
        overrides.append(f"model.checkpoint_path={checkpoint}")
    recipe = load_recipe(recipe_path)
    composition = load_composition(
        recipe,
        run_id=resolved_run_id,
        set_values=overrides,
        patch_paths=patch_paths,
    )
    runtime = compile_runtime(composition, "inference")
    try:
        inference = runtime.services.get("inference")
        if inference is None:
            raise BootError("no inference provider in the composition")
        for result in inference.generate(runtime.ctx, [{"prompt": prompt}]):
            print(result["text"])
    finally:
        runtime.close()
    return 0
