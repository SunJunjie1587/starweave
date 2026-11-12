"""Pydantic config models shared by seams and first-party plugins."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeviceConfig(ConfigBase):
    index: int = 0


class PrecisionConfig(ConfigBase):
    dtype: Literal["float32", "bfloat16"] = "bfloat16"


class CollectiveConfig(ConfigBase):
    backend: Literal["none", "hccl"] = "none"
    world_size: int = 1


class FSDPConfig(ConfigBase):
    enabled: bool = False
    strategy: Literal["none", "fsdp2"] = "none"


class EPConfig(ConfigBase):
    enabled: bool = False
    dispatcher: Literal["none", "eager", "mc2"] = "none"
    ep_size: int = 1


class KernelConfig(ConfigBase):
    attention: Literal["sdpa", "npu_fusion"] = "sdpa"
    rms_norm: bool = True


class ModelConfig(ConfigBase):
    name_or_path: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    tokenizer: str | None = None


class DataConfig(ConfigBase):
    path: str
    split: str = "train"
    max_seq_len: int = 2048
    prompt_field: str = "prompt"
    response_field: str = "response"
    batch_size: int = 1
    shuffle: bool = True
    seed: int = 42
    use_chat_template: bool = False


class OptimizerConfig(ConfigBase):
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    no_decay_keywords: list[str] = Field(
        default_factory=lambda: ["bias", "norm", "ln_", "embedding"]
    )


class SchedulerConfig(ConfigBase):
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.0


class LossConfig(ConfigBase):
    ignore_index: int = -100


class CheckpointConfig(ConfigBase):
    root: str = "runs/${run.id}/checkpoints"
    keep_last: int = 2


class MetricsConfig(ConfigBase):
    path: str = "runs/${run.id}/metrics.jsonl"


class ProfilerConfig(ConfigBase):
    enabled: bool = True
    output: str | None = None
    activities: list[str] | None = None


class TrainerConfig(ConfigBase):
    max_steps: int = 1000
    micro_batch_size: int = 1
    log_every: int = 10


class InferenceConfig(ConfigBase):
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 8
    do_sample: bool = False
    use_cache: bool = True
    skip_special_tokens: bool = True


class RolloutConfig(ConfigBase):
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    use_cache: bool = False
    skip_special_tokens: bool = True


class NumericsConfig(ConfigBase):
    """Numerical contract shared by training and rollout paths."""

    dtype: str = "bfloat16"
    logits_dtype: str = "float32"
    attention_backend: str | None = None
    moe_backend: str | None = None
    tp: int = 1
    ep: int = 1
    rollout_quantization: str | None = None
    recompute_logprobs: bool = True
    parity_max_abs: float = 1.0e-2


class TaskConfig(ConfigBase):
    max_steps: int = 100
    prompts_per_step: int = 1
    group_size: int = 2
    max_new_tokens: int = 32
    update_epochs: int = 1
    log_every: int = 1


class RewardConfig(ConfigBase):
    reference_field: str = "reference"
    match_reward: float = 1.0
    length_reward: float = 0.0
    token_reward: float = 0.0


class AlgorithmConfig(ConfigBase):
    group_size: int = 2
    clip_ratio: float = 0.2
    kl_coef: float = 0.0


class LauncherConfig(ConfigBase):
    pass


__all__ = [
    "AlgorithmConfig",
    "CheckpointConfig",
    "CollectiveConfig",
    "ConfigBase",
    "DataConfig",
    "DeviceConfig",
    "EPConfig",
    "FSDPConfig",
    "RewardConfig",
    "TaskConfig",
    "InferenceConfig",
    "KernelConfig",
    "LauncherConfig",
    "LossConfig",
    "MetricsConfig",
    "ModelConfig",
    "NumericsConfig",
    "OptimizerConfig",
    "PrecisionConfig",
    "ProfilerConfig",
    "RolloutConfig",
    "SchedulerConfig",
    "TrainerConfig",
]
