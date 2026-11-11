"""Training-side logprob recomputation for RL.

Scheme C: rollout only produces tokens; the RL loss uses logprobs computed by
the training model's full-sequence forward path. This keeps the loss on the
training numerical path even when rollout uses a different engine.
"""

from __future__ import annotations

from typing import Any


def causal_lm_logprobs(
    model: Any,
    input_ids: Any,
    attention_mask: Any = None,
    *,
    logits_dtype: str = "float32",
) -> Any:
    """Return next-token logprobs for ``input_ids`` using the training path.

    Args:
        model: callable returning an object with ``.logits`` of shape
            ``[batch, seq, vocab]``.
        input_ids: ``[batch, seq]`` token ids.
        attention_mask: optional ``[batch, seq]`` mask.
        logits_dtype: dtype used for log-softmax; defaults to float32.
    """
    import torch

    try:
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
    except TypeError:
        # Some lightweight/custom models do not accept HF's use_cache kwarg.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = getattr(outputs, "logits", outputs)
    dtype = getattr(torch, logits_dtype)
    logits = logits.to(dtype)
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    return log_probs.gather(-1, targets).squeeze(-1)


def masked_mean(values: Any, mask: Any) -> Any:
    """Mean of ``values`` over the true positions of a boolean/0-1 mask."""
    import torch

    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compare_logprobs(left: Any, right: Any, mask: Any = None) -> dict[str, float]:
    """Numerical parity metrics between two logprob tensors of equal shape."""
    import torch

    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    delta = (left - right).abs()
    if mask is not None:
        mask = mask.to(delta.dtype)
        delta = delta * mask
        denom = mask.sum().clamp_min(1.0)
        max_abs = float(delta.max().item())
        mean_abs = float(delta.sum().item() / denom.item())
        kl = float((((left.exp() * (left - right)) * mask).sum() / denom).item())
    else:
        max_abs = float(delta.max().item())
        mean_abs = float(delta.mean().item())
        kl = float((left.exp() * (left - right)).sum(dim=-1).mean().item())
    return {"max_abs": max_abs, "mean_abs": mean_abs, "kl": kl}
