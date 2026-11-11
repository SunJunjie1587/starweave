from __future__ import annotations

from pathlib import Path

import pytest

from starweave.cli.main import main

ROOT = Path(__file__).resolve().parents[2]
RECIPE = str(ROOT / "recipes" / "sft_npu_dsv4_small.yaml")

pytestmark = pytest.mark.npu


def test_tiny_sft_run() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not (ROOT / "tokenizer").exists():
        pytest.skip("dsv4 tokenizer/ is not present")
    if not (ROOT / "data" / "sft_dsv4.jsonl").exists():
        pytest.skip("data/sft_dsv4.jsonl is not present")
    assert (
        main(
            [
                "run",
                RECIPE,
                "--set",
                "trainer.max_steps=3",
                "--set",
                "checkpoint-policy.n=1",
                "--run-id",
                "npu-smoke",
            ]
        )
        == 0
    )
