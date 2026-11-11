"""JSONL and no-op metrics sinks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from starweave.seams import BaseRebindable, MetricEvent, MetricsConfig, PluginBase


class JSONLMetrics(PluginBase, BaseRebindable):
    name = "metrics-jsonl"
    provides = ("metrics",)
    requires = ()
    config_model = MetricsConfig

    def setup(self, ctx: Any, config: MetricsConfig) -> None:
        super().setup(ctx, config)
        self._path = Path(config.path)
        self._file = None

    def emit(self, ctx: Any, event: MetricEvent) -> None:
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding="utf-8")
        self._file.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")
        self._file.flush()

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def release(self) -> None:
        self.close()

    def teardown(self) -> None:
        self.close()


class NoopMetrics(PluginBase, BaseRebindable):
    name = "metrics-noop"
    provides = ("metrics",)
    requires = ()
    config_model = MetricsConfig

    def emit(self, ctx: Any, event: MetricEvent) -> None:
        return None
