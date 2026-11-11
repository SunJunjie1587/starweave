from __future__ import annotations

import json

from starweave.seams import MetricEvent
from starweave_example_metrics.plugin import MemoryMetrics, MemoryMetricsConfig


class FakeCtx:
    def __init__(self) -> None:
        self.cleanups = []

    def effect(self, cleanup) -> None:
        self.cleanups.append(cleanup)


def test_metadata() -> None:
    assert MemoryMetrics.provides == ("metrics",)
    assert MemoryMetrics.requires == ()
    assert MemoryMetrics.config_model is MemoryMetricsConfig


def test_emit_flush_and_teardown(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    ctx = FakeCtx()
    plugin = MemoryMetrics()
    plugin.setup(ctx, MemoryMetricsConfig(path=str(path)))
    plugin.emit(ctx, MetricEvent(name="loss", value=1.5, step=0, phase="train", timestamp=1.0))
    plugin.emit(ctx, MetricEvent(name="loss", value=1.0, step=1, phase="train", timestamp=2.0))
    plugin.flush()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["value"] for row in rows] == [1.5, 1.0]
    assert path.read_text(encoding="utf-8").strip()

    plugin.close()
    ctx.cleanups[0]()
