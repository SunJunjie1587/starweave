"""HuggingFace generate inference plugin."""

from __future__ import annotations

from typing import Any, Iterator

from starweave.seams import (
    BaseRebindable,
    InferenceConfig,
    ModelHandle,
    PluginBase,
)


class HFInference(PluginBase, BaseRebindable):
    name = "inference-hf"
    provides = ("inference",)
    requires = ("device", "model")
    config_model = InferenceConfig

    def load(self, ctx: Any, model: ModelHandle, config: InferenceConfig) -> None:
        self._model = model
        self._config = config

    def generate(self, ctx: Any, requests: list[Any]) -> Iterator[Any]:
        import torch

        model = self._model.model
        tokenizer = self._model.tokenizer
        config = self._config
        device = next(model.parameters()).device
        for request in requests:
            prompt = request["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    use_cache=config.use_cache,
                )
            generated = outputs[0][inputs["input_ids"].shape[-1] :]
            yield {
                "text": tokenizer.decode(
                    generated, skip_special_tokens=config.skip_special_tokens
                ),
                "token_ids": generated.tolist(),
            }
