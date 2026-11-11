from __future__ import annotations

import pytest

from starweave_model_dsv4.config import Dsv4ModelConfig
from starweave_model_dsv4.plugin import Dsv4Model


def test_metadata() -> None:
    assert Dsv4Model.name == "dsv4"
    assert Dsv4Model.provides == ("model",)
    assert Dsv4Model.requires == ("device",)
    assert Dsv4Model.config_model is Dsv4ModelConfig


def test_config_defaults() -> None:
    config = Dsv4ModelConfig(config_path="configs/dsv4_small.json")
    assert config.dtype == "bfloat16"
    assert config.attn_implementation == "eager"
    assert config.tokenizer_name_or_path == "tokenizer"


def test_config_rejects_unknown_key() -> None:
    with pytest.raises(Exception):
        Dsv4ModelConfig(config_path="x", nope=True)


def test_build_reports_missing_backend() -> None:
    plugin = Dsv4Model()
    with pytest.raises((ImportError, ModuleNotFoundError)):
        plugin.build(None, Dsv4ModelConfig(config_path="missing.json"))
