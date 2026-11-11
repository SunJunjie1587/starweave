"""Small kernel: context, plugin lifecycle, registry, events, and errors."""

from .context import Context, RunPaths
from .errors import BootError, ComposeError, ContextError, RunError, StarweaveError
from .events import EventBus
from .plugin import EmptyConfig, Plugin, PluginBase, validate_plugin_class
from .registry import SeamKind, ServiceRegistry
from .scope import Scope

__all__ = [
    "BootError",
    "ComposeError",
    "Context",
    "ContextError",
    "EmptyConfig",
    "EventBus",
    "Plugin",
    "PluginBase",
    "RunError",
    "RunPaths",
    "Scope",
    "SeamKind",
    "ServiceRegistry",
    "StarweaveError",
    "validate_plugin_class",
]
