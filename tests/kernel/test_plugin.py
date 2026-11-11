from __future__ import annotations

import pytest

from starweave.kernel.errors import ComposeError
from starweave.kernel.plugin import PluginBase, validate_plugin_class
from starweave.seams import EmptyConfig


class ValidPlugin(PluginBase):
    name = "valid"
    provides = ("metrics",)
    config_model = EmptyConfig


class MissingConfig:
    name = "bad"
    provides = ("metrics",)


def test_valid_plugin_metadata() -> None:
    validate_plugin_class(ValidPlugin)


def test_invalid_plugin_metadata() -> None:
    with pytest.raises(ComposeError):
        validate_plugin_class(MissingConfig)
