from __future__ import annotations

import pytest
from pydantic import BaseModel

from starweave.compose.loader import plugin_specs, resolve_plugin


@pytest.mark.parametrize("name", sorted(plugin_specs()))
def test_first_party_plugin_metadata(name: str) -> None:
    plugin_cls = resolve_plugin(name)
    assert isinstance(plugin_cls.name, str) and plugin_cls.name
    assert isinstance(plugin_cls.provides, tuple) and plugin_cls.provides
    assert isinstance(plugin_cls.requires, tuple)
    assert issubclass(plugin_cls.config_model, BaseModel)


def test_dsv4_entry_point() -> None:
    plugin_cls = resolve_plugin("dsv4")
    assert plugin_cls.name == "dsv4"
    assert plugin_cls.provides == ("model",)
    assert plugin_cls.requires == ("device",)
