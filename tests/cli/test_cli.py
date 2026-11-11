from __future__ import annotations

import importlib.util
from pathlib import Path

from starweave.cli.main import main

ROOT = Path(__file__).resolve().parents[2]
RECIPE = str(ROOT / "recipes" / "sft_npu_dsv4_small.yaml")


def test_dump_config_cli(capsys) -> None:
    assert main(["dump-config", RECIPE]) == 0
    output = capsys.readouterr().out
    assert "Composition hash:" in output
    assert "[model] dsv4" in output


def test_inspect_cli(capsys) -> None:
    assert main(["inspect", "--plan", RECIPE]) == 0
    output = capsys.readouterr().out
    assert "Phases:" in output
    assert "pinned" in output


def test_plugin_list_cli(capsys) -> None:
    assert main(["plugin", "list"]) == 0
    output = capsys.readouterr().out
    assert "dsv4" in output


def test_doctor_without_npu() -> None:
    if importlib.util.find_spec("torch_npu") is not None:
        import pytest

        pytest.skip("torch_npu is installed; doctor should succeed")
    assert main(["doctor"]) == 3
