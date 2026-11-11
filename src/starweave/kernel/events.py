"""Typed-by-convention event bus used by metrics, profiler, and run logging."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def on(self, name: str, handler: Callable[[Any], None]) -> None:
        self._handlers[name].append(handler)

    def emit(self, name: str, payload: Any = None) -> None:
        for handler in list(self._handlers.get(name, ())):
            handler(payload)
