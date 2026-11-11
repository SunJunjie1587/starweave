"""RL prompt JSONL data plugin.

Rows are prompt/reference dicts; the task, not the data plugin, decides how to
tokenize and reward them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from starweave.seams import DataBundle, DataConfig, PluginBase


class RLJSONLData(PluginBase):
    name = "data-rl-jsonl"
    provides = ("data",)
    requires = ()
    config_model = DataConfig

    def build(self, ctx: Any, config: DataConfig) -> DataBundle:
        rows: list[dict[str, Any]] = []
        with Path(config.path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"no rows found in {config.path}")

        def iterator_factory() -> Iterator[list[dict[str, Any]]]:
            order = list(range(len(rows)))
            if config.shuffle:
                random.Random(config.seed).shuffle(order)
            batch_size = max(1, config.batch_size)
            for start in range(0, len(order), batch_size):
                yield [rows[index] for index in order[start : start + batch_size]]

        def collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return batch

        return DataBundle(dataset=rows, collator=collate, iterator_factory=iterator_factory)
