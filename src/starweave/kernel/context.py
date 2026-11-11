"""Runtime service container, run paths, and lifecycle scope."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ContextError
from .events import EventBus
from .scope import Scope


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoint_dir: Path
    metrics_path: Path


def default_paths(run_id: str, root: Path | None = None) -> RunPaths:
    run_dir = (root or Path("runs")) / run_id
    return RunPaths(
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        metrics_path=run_dir / "metrics.jsonl",
    )


class Context:
    def __init__(
        self,
        run_id: str,
        *,
        seed: int = 42,
        paths: RunPaths | None = None,
    ) -> None:
        self.run_id = run_id
        self.seed = seed
        self.paths = paths or default_paths(run_id)
        self.scope = Scope()
        self.events = EventBus()
        self._services: dict[str, Any] = {}

    def provide(self, name: str, service: Any) -> None:
        self._services[name] = service

    def get(self, name: str) -> Any:
        try:
            return self._services[name]
        except KeyError as exc:
            raise ContextError(f"service '{name}' is not available") from exc

    def effect(self, cleanup) -> None:
        self.scope.effect(cleanup)

    def on(self, name: str, handler) -> None:
        self.events.on(name, handler)

    def emit(self, name: str, payload: Any = None) -> None:
        self.events.emit(name, payload)

    def close(self) -> None:
        self.scope.close()
