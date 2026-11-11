"""HuggingFace rollout plugin for RL correctness baselines.

The rollout seam only exposes generation and weight synchronization. It never
imports a concrete training engine or a task plugin.
"""

from __future__ import annotations

from typing import Any

from starweave.seams import PluginBase, RolloutConfig, RolloutOutput


class HFRollout(PluginBase):
    name = "rollout-hf"
    provides = ("rollout",)
    requires = ("device", "model")
    config_model = RolloutConfig

    def generate(self, ctx: Any, requests: list[Any]) -> list[RolloutOutput]:
        import torch

        model_handle = ctx.get("model")
        model = model_handle.model
        tokenizer = model_handle.tokenizer
        device = next(model.parameters()).device
        outputs: list[RolloutOutput] = []
        model.eval()
        try:
            for request in requests:
                prompt = request["prompt"] if isinstance(request, dict) else str(request)
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=self.config.do_sample,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        use_cache=self.config.use_cache,
                    )
                prompt_ids = inputs["input_ids"][0].tolist()
                response_ids = generated[0][len(prompt_ids) :].tolist()
                outputs.append(
                    RolloutOutput(
                        prompt_ids=tuple(prompt_ids),
                        response_ids=tuple(response_ids),
                        text=tokenizer.decode(
                            response_ids,
                            skip_special_tokens=self.config.skip_special_tokens,
                        ),
                    )
                )
        finally:
            model.train()
        return outputs

    def update_weights(self, ctx: Any, model: Any) -> None:
        # HFRollout shares the training model object; no copy is needed.
        return None
