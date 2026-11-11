"""Runtime executor. M1 fills this in; M0 fails explicitly rather than pretending to run."""

from __future__ import annotations

from typing import Any

from ..kernel.errors import BootError


def execute(plan: Any, ctx: Any) -> int:
    raise BootError("runtime execution is implemented in M1")
