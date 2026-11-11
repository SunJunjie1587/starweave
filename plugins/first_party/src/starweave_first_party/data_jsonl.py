"""JSONL and HuggingFace SFT data plugins."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from starweave.seams import PluginBase
from starweave.seams.configs import DataConfig
from starweave.seams.types import DataBundle


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class JSONLData(PluginBase):
    name = "data-jsonl"
    provides = ("data",)
    requires = ("model",)
    config_model = DataConfig

    def build(self, ctx: Any, config: DataConfig) -> DataBundle:
        import torch
        from torch.utils.data import DataLoader

        model_handle = ctx.get("model")
        tokenizer = model_handle.tokenizer
        rows = _read_jsonl(config.path)
        if not rows:
            raise ValueError(f"no rows found in {config.path}")

        eos = getattr(tokenizer, "eos_token", None) or ""
        pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(
            tokenizer, "eos_token_id", 0
        )

        def encode(row: dict[str, Any]) -> dict[str, Any]:
            prompt = str(row.get(config.prompt_field, ""))
            response = str(row.get(config.response_field, ""))
            if config.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                prompt_tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            else:
                prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_tokens = tokenizer(
                response + eos, add_special_tokens=False
            )["input_ids"]
            input_ids = (list(prompt_tokens) + list(response_tokens))[: config.max_seq_len]
            labels = ([-100] * len(prompt_tokens) + list(response_tokens))[
                : config.max_seq_len
            ]
            if not input_ids:
                raise ValueError("empty training example after tokenization")
            return {"input_ids": input_ids, "labels": labels}

        encoded = [encode(row) for row in rows]

        def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
            max_len = max(len(item["input_ids"]) for item in batch)
            input_ids, labels, attention = [], [], []
            for item in batch:
                pad = max_len - len(item["input_ids"])
                input_ids.append(item["input_ids"] + [pad_id] * pad)
                labels.append(item["labels"] + [-100] * pad)
                attention.append([1] * len(item["input_ids"]) + [0] * pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        def iterator_factory():
            generator = torch.Generator()
            generator.manual_seed(config.seed)
            loader = DataLoader(
                encoded,
                batch_size=config.batch_size,
                shuffle=config.shuffle,
                collate_fn=collate,
                generator=generator,
                drop_last=False,
            )
            return iter(loader)

        return DataBundle(dataset=encoded, collator=collate, iterator_factory=iterator_factory)


class HFData(JSONLData):
    name = "data-hf"

    def build(self, ctx: Any, config: DataConfig) -> DataBundle:
        from datasets import load_dataset

        try:
            dataset = load_dataset(config.path, split=config.split)
        except Exception:  # noqa: BLE001
            dataset = load_dataset("json", data_files=config.path, split="train")
        rows = [dict(item) for item in dataset]
        # Reuse JSONLData's encoding/collation by temporarily writing is wasteful;
        # duplicate the small core instead.
        import torch
        from torch.utils.data import DataLoader

        model_handle = ctx.get("model")
        tokenizer = model_handle.tokenizer
        eos = getattr(tokenizer, "eos_token", None) or ""
        pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(
            tokenizer, "eos_token_id", 0
        )

        def encode(row: dict[str, Any]) -> dict[str, Any]:
            prompt = str(row.get(config.prompt_field, ""))
            response = str(row.get(config.response_field, ""))
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_tokens = tokenizer(
                response + eos, add_special_tokens=False
            )["input_ids"]
            return {
                "input_ids": (list(prompt_tokens) + list(response_tokens))[
                    : config.max_seq_len
                ],
                "labels": ([-100] * len(prompt_tokens) + list(response_tokens))[
                    : config.max_seq_len
                ],
            }

        encoded = [encode(row) for row in rows]

        def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
            max_len = max(len(item["input_ids"]) for item in batch)
            input_ids, labels, attention = [], [], []
            for item in batch:
                pad = max_len - len(item["input_ids"])
                input_ids.append(item["input_ids"] + [pad_id] * pad)
                labels.append(item["labels"] + [-100] * pad)
                attention.append([1] * len(item["input_ids"]) + [0] * pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        def iterator_factory():
            generator = torch.Generator()
            generator.manual_seed(config.seed)
            return iter(
                DataLoader(
                    encoded,
                    batch_size=config.batch_size,
                    shuffle=config.shuffle,
                    collate_fn=collate,
                    generator=generator,
                    drop_last=False,
                )
            )

        return DataBundle(dataset=encoded, collator=collate, iterator_factory=iterator_factory)
