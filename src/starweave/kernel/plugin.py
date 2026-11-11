"""Plugin protocol and metadata validation. The base class lives in seams."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ..seams.plugin import EmptyConfig, PluginBase
from .errors import ComposeError


@runtime_checkable
class Plugin(Protocol):
    name: str
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    config_model: type[BaseModel]
    rebindable: bool
    priority: int
    optional: bool

    def setup(self, ctx: Any, config: BaseModel) -> None: ...
    def teardown(self) -> None: ...


def validate_plugin_class(plugin_cls: type, *, row: str | None = None) -> None:
    name = getattr(plugin_cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ComposeError("plugin class lacks a non-empty 'name'", row=row)
    for attr in ("provides", "requires"):
        value = getattr(plugin_cls, attr, None)
        if not isinstance(value, tuple) or not all(isinstance(v, str) for v in value):
            raise ComposeError(f"plugin '{name}' has invalid '{attr}'", plugin=name, row=row)
    config_model = getattr(plugin_cls, "config_model", None)
    if not (isinstance(config_model, type) and issubclass(config_model, BaseModel)):
        raise ComposeError(
            f"plugin '{name}' config_model must be a pydantic BaseModel subclass",
            plugin=name,
            row=row,
        )


__all__ = ["EmptyConfig", "Plugin", "PluginBase", "validate_plugin_class"]
