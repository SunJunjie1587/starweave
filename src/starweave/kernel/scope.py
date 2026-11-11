"""Deterministic cleanup scope. Cleanups run in reverse registration order."""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class Scope:
    def __init__(self) -> None:
        self._cleanups: list[Callable[[], None]] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def effect(self, cleanup: Callable[[], None]) -> None:
        if self._closed:
            cleanup()
            return
        self._cleanups.append(cleanup)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for cleanup in reversed(self._cleanups):
            try:
                cleanup()
            except Exception:  # noqa: BLE001 - teardown failures must not block unwinding
                logger.exception("cleanup failed")
        self._cleanups.clear()
