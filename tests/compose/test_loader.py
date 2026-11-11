from __future__ import annotations

from pathlib import Path

import pytest

from starweave.compose.loader import load_composition
from starweave.compose.recipe import load_recipe
from starweave.kernel.errors import ComposeError

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "recipes" / "sft_npu_dsv4_small.yaml"


def test_composition_has_expected_rows() -> None:
    composition = load_composition(RECIPE, run_id="dump")
    assert composition.by_id["model"].plugin == "dsv4"
    assert composition.by_id["trainer"].plugin == "trainer-sft"
    assert composition.by_id["device"].plugin == "device-npu"
    assert composition.by_id["fsdp"].plugin == "fsdp-none"
    assert composition.by_id["ep"].plugin == "ep-none"
    assert composition.by_id["metrics-noop"].config.path == "runs/dump/metrics.jsonl"


def test_hash_is_stable() -> None:
    first = load_composition(RECIPE, run_id="dump")
    second = load_composition(RECIPE, run_id="dump")
    assert first.composition_hash == second.composition_hash


def test_set_changes_config_and_hash() -> None:
    base = load_composition(RECIPE, run_id="dump")
    changed = load_composition(RECIPE, run_id="dump", set_values=["trainer.max_steps=5"])
    assert changed.by_id["trainer"].config.max_steps == 5
    assert changed.composition_hash != base.composition_hash


def test_set_unknown_row_rejected() -> None:
    with pytest.raises(ComposeError):
        load_composition(RECIPE, run_id="dump", set_values=["missing.value=1"])


def test_patch_overlay(tmp_path: Path) -> None:
    patch = tmp_path / "overlay.yaml"
    patch.write_text("- id: trainer\n  config: {max_steps: 7}\n", encoding="utf-8")
    composition = load_composition(RECIPE, run_id="dump", patch_paths=[str(patch)])
    assert composition.by_id["trainer"].config.max_steps == 7


def test_recipe_loads() -> None:
    recipe = load_recipe(RECIPE)
    assert recipe.name == "sft-npu-dsv4-small"
    assert recipe.bundles == ["base", "npu", "hf", "dsv4", "sft"]
