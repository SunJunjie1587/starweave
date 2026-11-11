from __future__ import annotations

import pytest

from starweave.kernel.context import Context
from starweave.kernel.errors import ContextError


def test_provide_and_get() -> None:
    ctx = Context("test")
    ctx.provide("device", object())
    assert ctx.get("device") is not None


def test_missing_service_raises() -> None:
    ctx = Context("test")
    with pytest.raises(ContextError):
        ctx.get("missing")


def test_effect_cleanup_runs_in_reverse_order() -> None:
    ctx = Context("test")
    order: list[str] = []
    ctx.effect(lambda: order.append("first"))
    ctx.effect(lambda: order.append("second"))
    ctx.close()
    assert order == ["second", "first"]


def test_close_is_idempotent() -> None:
    ctx = Context("test")
    calls: list[int] = []
    ctx.effect(lambda: calls.append(1))
    ctx.close()
    ctx.close()
    assert calls == [1]


def test_events() -> None:
    ctx = Context("test")
    seen: list[object] = []
    ctx.on("step/end", seen.append)
    ctx.emit("step/end", {"loss": 1.0})
    assert seen == [{"loss": 1.0}]
