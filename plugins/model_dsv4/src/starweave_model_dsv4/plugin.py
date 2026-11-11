"""DeepSeek-V4 native model plugin.

The model implementation is loaded from ``_vendor/`` when present, otherwise
from an installed ``mindspeed_llm`` checkout. The plugin module itself imports
only Starweave seams and pydantic; torch/transformers are deferred to build().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from starweave.seams import ModelHandle, ParallelPlan, PluginBase

from .config import Dsv4ModelConfig


def _load_model_classes():
    try:
        from ._vendor.configuration_deepseek_v4_native import DeepseekV4Config
        from ._vendor.modeling_deepseek_v4_native import DeepseekV4ForCausalLM

        return DeepseekV4Config, DeepseekV4ForCausalLM
    except ImportError:
        pass

    try:
        from mindspeed_llm.fsdp2.models.deepseek_v4_native.configuration_deepseek_v4_native import (  # noqa: E501
            DeepseekV4Config,
        )
        from mindspeed_llm.fsdp2.models.deepseek_v4_native.modeling_deepseek_v4_native import (  # noqa: E501
            DeepseekV4ForCausalLM,
        )

        return DeepseekV4Config, DeepseekV4ForCausalLM
    except ImportError as exc:
        raise ImportError(
            "model-dsv4 requires either vendored dsv4 files in "
            "starweave_model_dsv4/_vendor/ or MindSpeed-LLM on PYTHONPATH"
        ) from exc


class Dsv4Model(PluginBase):
    name = "dsv4"
    provides = ("model",)
    requires = ("device",)
    config_model = Dsv4ModelConfig

    def build(self, ctx: Any, config: Dsv4ModelConfig) -> ModelHandle:
        import json

        import torch
        from transformers import AutoTokenizer

        DeepseekV4Config, DeepseekV4ForCausalLM = _load_model_classes()
        config_path = Path(config.config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"dsv4 config not found: {config_path}")

        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "model_type",
            "architectures",
            "torch_dtype",
            "transformers_version",
            "_name_or_path",
            "_commit_hash",
        ):
            config_data.pop(key, None)
        model_config = DeepseekV4Config(**config_data)
        model_config._attn_implementation = config.attn_implementation
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
        model = DeepseekV4ForCausalLM(model_config)

        if config.checkpoint_path:
            from safetensors.torch import load_file

            checkpoint = Path(config.checkpoint_path)
            if checkpoint.is_dir():
                checkpoint = checkpoint / "model.safetensors"
            state = load_file(str(checkpoint))
            model.load_state_dict(state, strict=False)

        dtype = getattr(torch, config.dtype)
        model.to(dtype=dtype)
        device = ctx.get("device")
        model.to(device.device_type)

        plan = ParallelPlan(
            shard_modules=(
                "model.embed_tokens",
                "model.layers.{*}",
                "model.norm",
                "model.hc_head",
                "lm_head",
            ),
            expert_modules=("model.layers.{*}.mlp.experts",),
            replicate_modules=(
                "model.layers.{*}.attn_hc",
                "model.layers.{*}.ffn_hc",
                "model.hc_head",
            ),
        )
        return ModelHandle(
            model=model,
            tokenizer=tokenizer,
            vocab_size=int(model_config.vocab_size),
            param_count=sum(parameter.numel() for parameter in model.parameters()),
            dtype=config.dtype,
            parallel_plan=plan,
        )

    def teardown(self) -> None:
        return None
