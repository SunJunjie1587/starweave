"""The RL algorithm seam: advantage estimation and policy loss."""

from .configs import AlgorithmConfig
from .protocols import Algorithm

__all__ = ["Algorithm", "AlgorithmConfig"]
