"""Safetensors and torch checkpoint stores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from starweave.seams import (
    BaseRebindable,
    CheckpointConfig,
    CheckpointState,
    PluginBase,
)


class SafetensorsCheckpoint(PluginBase, BaseRebindable):
    name = "checkpoint-safetensors"
    provides = ("checkpoint",)
    requires = ()
    config_model = CheckpointConfig

    def setup(self, ctx: Any, config: CheckpointConfig) -> None:
        super().setup(ctx, config)
        self.root = Path(config.root)
        self.keep_last = config.keep_last

    def save(self, ctx: Any, state: CheckpointState) -> Path:
        import shutil

        import torch

        target = self.root / f"step-{state.step:06d}"
        target.mkdir(parents=True, exist_ok=True)
        model_state = state.model_state or {}
        try:
            from safetensors.torch import save_file

            cpu_state = {
                key: value.detach().cpu().contiguous()
                for key, value in model_state.items()
                if hasattr(value, "detach")
            }
            save_file(cpu_state, str(target / "model.safetensors"))
        except Exception:  # noqa: BLE001 - fall back to torch when safetensors is unavailable
            torch.save(model_state, target / "model.pt")
        torch.save(state.optimizer, target / "optimizer.pt")
        torch.save(state.scheduler, target / "scheduler.pt")
        torch.save(state.rng, target / "rng.pt")
        (target / "trainer_state.json").write_text(
            json.dumps(
                {
                    "step": state.step,
                    "dataloader_position": state.dataloader_position,
                    "composition_hash": state.composition_hash,
                    "config_snapshot": state.config_snapshot,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "latest").write_text(target.name, encoding="utf-8")
        if self.keep_last > 0:
            candidates = sorted(
                path for path in self.root.iterdir() if path.is_dir()
            )
            for old in candidates[: -self.keep_last]:
                shutil.rmtree(old, ignore_errors=True)
        return target

    def load(self, ctx: Any, path: Any) -> CheckpointState:
        import torch

        target = Path(path)
        if str(target) == "latest":
            latest = self.latest(ctx)
            if latest is None:
                raise FileNotFoundError(f"no checkpoint under {self.root}")
            target = latest
        if not target.is_absolute():
            target = self.root / target
        if not target.is_dir():
            raise FileNotFoundError(f"checkpoint not found: {target}")
        trainer_state = json.loads(
            (target / "trainer_state.json").read_text(encoding="utf-8")
        )
        model_path = target / "model.safetensors"
        if model_path.exists():
            from safetensors.torch import load_file

            model_state = load_file(str(model_path), device="cpu")
        else:
            model_state = torch.load(target / "model.pt", map_location="cpu")
        return CheckpointState(
            model_dir=target,
            optimizer=torch.load(target / "optimizer.pt", map_location="cpu"),
            scheduler=torch.load(target / "scheduler.pt", map_location="cpu"),
            step=int(trainer_state["step"]),
            dataloader_position=int(trainer_state["dataloader_position"]),
            rng=torch.load(target / "rng.pt", map_location="cpu"),
            composition_hash=str(trainer_state["composition_hash"]),
            config_snapshot=dict(trainer_state["config_snapshot"]),
            model_state=model_state,
        )

    def latest(self, ctx: Any) -> Path | None:
        if not self.root.exists():
            return None
        pointer = self.root / "latest"
        if pointer.is_file():
            candidate = self.root / pointer.read_text(encoding="utf-8").strip()
            if candidate.is_dir():
                return candidate
        candidates = [path for path in self.root.iterdir() if path.is_dir()]
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


class TorchCheckpoint(SafetensorsCheckpoint):
    name = "checkpoint-torch"
