from __future__ import annotations

from pathlib import Path

from starweave.cli.dump import format_composition
from starweave.compose.loader import load_composition

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "dump_config_sft_npu_dsv4_small.txt"


def test_dump_config_matches_golden(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    composition = load_composition("recipes/sft_npu_dsv4_small.yaml", run_id="dump")
    actual = format_composition(composition)
    expected = GOLDEN.read_text(encoding="utf-8").rstrip("\n")
    assert actual == expected
