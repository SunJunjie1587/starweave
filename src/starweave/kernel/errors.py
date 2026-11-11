"""Structured errors used across compose, boot, and runtime."""

from __future__ import annotations


class StarweaveError(Exception):
    """Base error with optional plugin and composition-row context."""

    code = "starweave-error"

    def __init__(
        self,
        message: str,
        *,
        plugin: str | None = None,
        row: str | None = None,
    ) -> None:
        self.message = message
        self.plugin = plugin
        self.row = row
        super().__init__(str(self))

    def __str__(self) -> str:
        parts = [self.message]
        if self.plugin:
            parts.append(f"plugin={self.plugin}")
        if self.row:
            parts.append(f"row={self.row}")
        return "; ".join(parts)


class ComposeError(StarweaveError):
    """Invalid composition: unknown plugin, duplicate id, bad patch, bad config."""

    code = "compose-error"


class ContextError(StarweaveError):
    """A service lookup or context operation failed."""

    code = "context-error"


class BootError(StarweaveError):
    """A runtime service failed to start or a backend is unavailable."""

    code = "boot-error"


class RunError(StarweaveError):
    """A step or phase failed at runtime."""

    code = "run-error"
