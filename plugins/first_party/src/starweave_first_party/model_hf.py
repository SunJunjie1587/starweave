"""Generic HuggingFace model plugin. Heavy imports happen in build()."""

from __future__ import annotations

from typing import Any

from starweave.seams import ModelConfig, ModelHandle, ParallelPlan, PluginBase


class HFModel(PluginBase):
    name = "model-hf"
    provides = ("model",)
    requires = ("device",)
    config_model = ModelConfig

    def build(self, ctx: Any, config: ModelConfig) -> ModelHandle:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_path = config.tokenizer or config.name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=config.trust_remote_code
        )
        dtype = getattr(torch, config.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            config.name_or_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=dtype,
        )
        device = ctx.get("device")
        model.to(device.device_type)
        return ModelHandle(
            model=model,
            tokenizer=tokenizer,
            vocab_size=int(getattr(model.config, "vocab_size", 0)),
            param_count=sum(parameter.numel() for parameter in model.parameters()),
            dtype=config.dtype,
            parallel_plan=ParallelPlan(shard_modules=("model.layers.{*}",)),
        )

    def teardown(self) -> None:
        return None
