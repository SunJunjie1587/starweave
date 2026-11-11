"""The reward seam used by RL tasks."""

from .configs import RewardConfig
from .protocols import Reward

__all__ = ["Reward", "RewardConfig"]
