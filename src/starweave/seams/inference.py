"""The inference seam."""

from .configs import InferenceConfig
from .protocols import InferenceEngine

__all__ = ["InferenceEngine", "InferenceConfig"]
