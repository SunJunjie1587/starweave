"""Plugin base available to first-party and external plugins.

This module lives in the seams package so plugins never need to import the kernel.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class EmptyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PluginBase:
    name = "plugin"
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    config_model: type[BaseModel] = EmptyConfig
    rebindable = False
    priority = 100
    optional = False
    version = "0.1.0"

    def setup(self, ctx: Any, config: BaseModel) -> None:
        self.ctx = ctx
        self.config = config
        ctx.effect(self.teardown)

    def teardown(self) -> None:
        return None


__all__ = ["EmptyConfig", "PluginBase"]
