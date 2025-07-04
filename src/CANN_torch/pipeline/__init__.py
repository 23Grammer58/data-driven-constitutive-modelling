"""Pipeline модуль для обучения и тестирования CANN моделей."""

from .CANN_pipeline import train_and_evaluate, TrainingConfig

__all__ = ["train_and_evaluate", "TrainingConfig"] 