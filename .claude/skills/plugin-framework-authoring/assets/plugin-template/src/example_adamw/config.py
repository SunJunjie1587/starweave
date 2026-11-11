from pydantic import BaseModel, Field


class AdamWConfig(BaseModel):
    """Config is owned by this plugin and nothing else."""

    model_config = {"extra": "forbid"}

    lr: float = Field(default=1.0e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1.0e-8, gt=0)
