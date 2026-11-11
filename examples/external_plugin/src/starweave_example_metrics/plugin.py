"""A metrics sink that buffers events and flushes them as JSONL.

This package is deliberately minimal: its only Starweave import is
``starweave.seams``. It is the reference for third-party plugins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from starweave.seams import PluginBase


class MemoryMetricsConfig(BaseModel):
    path: str = "runs/example/metrics-memory.jsonl"


class MemoryMetrics(PluginBase):
    name = "metrics-memory"
    provides = ("metrics",)
    requires = ()
    config_model = MemoryMetricsConfig

    def setup(self, ctx: Any, config: MemoryMetricsConfig) -> None:
        super().setup(ctx, config)
        self._events: list[Any] = []

    def emit(self, ctx: Any, event: Any) -> None:
        self._events.append(event)

    def flush(self) -> None:
        if not self._events:
            return
        path = Path(self.config.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(json.dumps(event.__dict__, default=str) + "\n")
        self._events.clear()

    def close(self) -> None:
        self.flush()

    def release(self) -> None:
        return None

    def teardown(self) -> None:
        self.close()
