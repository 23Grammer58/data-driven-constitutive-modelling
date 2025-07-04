"""
CANN_torch - библиотека для реализации Constitutive Artificial Neural Networks
"""

from .core.trainer import Trainer
from .pipeline.CANN_pipeline import TrainingConfig, train_and_evaluate  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "Trainer",
    "TrainingConfig",
    "train_and_evaluate",
]