from __future__ import annotations

from pathlib import Path

import pytest

from starweave.compose.loader import load_composition
from starweave.kernel.errors import BootError
from starweave.plan.runtime import compile_runtime

ROOT = Path(__file__).resolve().parents[2]
RECIPE = str(ROOT / "recipes" / "sft_npu_dsv4_small.yaml")

pytestmark = pytest.mark.npu


def test_phase_switch_rebinds_rebindable_and_guards_pinned() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not (ROOT / "tokenizer").exists():
        pytest.skip("dsv4 tokenizer/ is not present")
    composition = load_composition(RECIPE, run_id="npu-phase-test")
    runtime = compile_runtime(composition, "train")
    try:
        model = runtime.services["model"]
        train_metrics = runtime.services["metrics"]
        assert train_metrics.name == "metrics-jsonl"

        checkpoint_plan = runtime.switch_phase("checkpoint")
        assert runtime.services["metrics"].name == "metrics-noop"
        assert runtime.training_services.metrics is runtime.services["metrics"]
        assert runtime.services["model"] is model
        assert checkpoint_plan.composition_hash != runtime.static_plan.phases["train"].composition_hash

        eval_plan = runtime.switch_phase("eval")
        assert runtime.services["metrics"].name == "metrics-jsonl"
        assert runtime.services["metrics"] is not train_metrics
        assert eval_plan.composition_hash != checkpoint_plan.composition_hash

        runtime.row_ids["model"] = "stale-model"
        with pytest.raises(BootError):
            runtime.switch_phase("checkpoint")
    finally:
        runtime.close()
