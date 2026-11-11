from __future__ import annotations

from pathlib import Path

import pytest

from starweave.compose.loader import load_composition
from starweave.kernel.errors import ComposeError
from starweave.plan.compiler import compile_static
from starweave.plan.validator import build_phase

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "recipes" / "sft_npu_dsv4_small.yaml"


def test_static_plan_has_three_phases() -> None:
    composition = load_composition(RECIPE, run_id="dump")
    plan = compile_static(composition)
    assert set(plan.phases) == {"train", "eval", "checkpoint"}
    assert "model" in plan.phases["train"].services
    assert plan.phases["train"].services["model"].pinned
    assert plan.phases["train"].services["metrics"].rebindable


def test_policies_are_hooks_not_services() -> None:
    composition = load_composition(RECIPE, run_id="dump")
    plan = compile_static(composition)
    services = plan.phases["train"].services
    assert "step_policy" not in services
    assert any("policy-grad-clip" in hook for hook in plan.phases["train"].hooks)


def test_checkpoint_phase_rebinds_metrics_and_has_own_hash() -> None:
    composition = load_composition(RECIPE, run_id="dump")
    plan = compile_static(composition)
    assert plan.phases["checkpoint"].services["metrics"].id == "metrics-noop"
    assert (
        plan.phases["train"].composition_hash
        != plan.phases["checkpoint"].composition_hash
    )


def test_pinned_rebind_rejected() -> None:
    composition = load_composition(RECIPE, run_id="dump")
    with pytest.raises(ComposeError):
        build_phase(composition, "train", {"model": "model"})


def test_missing_required_seam_rejected(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "name: empty\nbundles: []\npatch: []\nphases: {train: {}}\n",
        encoding="utf-8",
    )
    composition = load_composition(recipe, run_id="dump")
    with pytest.raises(ComposeError):
        compile_static(composition)
