from __future__ import annotations

from typing import Literal

from starweave.seams import ConfigBase


class Dsv4ModelConfig(ConfigBase):
    config_path: str
    checkpoint_path: str | None = None
    tokenizer_name_or_path: str = "tokenizer"
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    attn_implementation: Literal["eager", "sdpa"] = "eager"
