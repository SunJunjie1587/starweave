"""plugin list."""

from __future__ import annotations

from ..compose.loader import plugin_specs


def plugin_list() -> int:
    for name, value in plugin_specs().items():
        print(f"{name}\t{value}\tinstalled")
    return 0
