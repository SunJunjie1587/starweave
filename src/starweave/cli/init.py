"""starweave init: create a minimal recipe scaffold without overwriting."""

from __future__ import annotations

from pathlib import Path

from ..kernel.errors import ComposeError

_TEMPLATE = """name: my-run
bundles: [base, npu, hf, dsv4, sft]
patch: []
phases:
  train:      {metrics: metrics, profiler: profiler, checkpoint: checkpoint}
  eval:       {metrics: metrics, profiler: profiler}
  checkpoint: {checkpoint: checkpoint, metrics: metrics-noop}
"""


def init(directory: str) -> int:
    target = Path(directory)
    if target.exists() and any(target.iterdir()):
        raise ComposeError(f"target directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    (target / "data").mkdir(exist_ok=True)
    (target / "recipe.yaml").write_text(_TEMPLATE, encoding="utf-8")
    (target / ".gitignore").write_text("runs/\ntokenizer/\ncheckpoints/\n", encoding="utf-8")
    print(f"created {target}/recipe.yaml")
    return 0
