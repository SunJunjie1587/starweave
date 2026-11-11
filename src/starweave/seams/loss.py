"""The loss seam."""

from .configs import LossConfig
from .protocols import LossFactory

__all__ = ["LossFactory", "LossConfig"]
