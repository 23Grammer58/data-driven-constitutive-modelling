from __future__ import annotations

"""architecture_factory.py
Фабрика для динамического создания инвариантных сетей и энергий деформации
на основе базовых классов `BaseInvNet` и `BaseStrainEnergy`.

Предполагаемый сценарий использования во внешнем (веб) сервисе:

```python
from CANN_torch.models.architecture_factory import (
    InvNetConfig, StrainEnergyConfig, ArchitectureFactory,
)
from CANN_torch.models.CANN import ModelArchitecture_I2

# 1. Формируем конфигурацию (обычно приходит из JSON запроса)
inv_cfg = InvNetConfig(
    activation_functions=["linear", "exp"],  # допустим только линейная и экспонента
    polynomial_degree=2,
    bias=3.0,
)
se_cfg = StrainEnergyConfig(
    invariants_config=[3.0, 3.0],  # две инвариантные сети, обе со смещением 3
    inv_net_config=inv_cfg,
)

# 2. Создаём Psi-модель и конечную архитектуру
psi_model = ArchitectureFactory.create_strain_energy(se_cfg)
model = ModelArchitecture_I2(Psi_model=psi_model)
```

Таким образом, меняя поля конфигурации, можно на лету собирать разнообразные CANN-архитектуры
без ручного написания новых классов.
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Type
import numpy as np
import torch.nn as nn

from .CANN import BaseInvNet, BaseStrainEnergy  # базовые классы

__all__ = [
    "InvNetConfig",
    "StrainEnergyConfig",
    "ArchitectureFactory",
]


# -----------------------------------------------------------------------------
# Dataclass-конфигурации (легко сериализуются/десериализуются из JSON)
# -----------------------------------------------------------------------------


@dataclass
class InvNetConfig:
    """Параметры одной инвариантной сети."""

    activation_functions: List[str] = field(default_factory=lambda: ["linear", "exp"])
    polynomial_degree: int = 2
    bias: float = 3.0

    def validate(self) -> None:
        allowed = {"linear", "exp", "ln"}
        if not set(self.activation_functions) <= allowed:
            raise ValueError(
                f"Незнакомые функции активации: {set(self.activation_functions) - allowed}"
            )
        if self.polynomial_degree < 1:
            raise ValueError("polynomial_degree должен быть >= 1")


@dataclass
class StrainEnergyConfig:
    """Конфигурация для создания Psi-модели (энергии деформации)."""

    invariants_config: Sequence[float]  # список bias (смещений) для каждой инвариантной сети
    inv_net_config: InvNetConfig = field(default_factory=InvNetConfig)

    def validate(self) -> None:
        if len(self.invariants_config) == 0:
            raise ValueError("Нужно указать хотя бы один инвариант (invariants_config)")
        self.inv_net_config.validate()


# -----------------------------------------------------------------------------
# ModelArchitecture конфигурация
# -----------------------------------------------------------------------------

@dataclass
class ModelArchitectureConfig:
    """Параметры итоговой архитектуры (изотропная / анизотропная модель)."""

    # "изотропный" – 2-инвариантная модель (I1, I2)
    # "анизотропный" – 4-инвариантная модель (I1, I2, I4, I5)
    # "auto" – выбрать на основе длины invariants_config (<=2 → изотропный, иначе анизотропный)
    architecture_type: str = "auto"
    strain_energy_cfg: StrainEnergyConfig = field(default_factory=StrainEnergyConfig)

    # Специфичные для анизотропной модели
    setAl: bool = True
    init_alpha: float = 0.1  # начальное значение угла

    def validate(self) -> None:
        allowed = {"изотропный", "анизотропный", "auto"}
        if self.architecture_type not in allowed:
            raise ValueError(f"architecture_type должно быть одним из {allowed}")
        self.strain_energy_cfg.validate()

        # Автовыбор типа по числу инвариантов
        if self.architecture_type == "auto":
            self.architecture_type = "изотропный" if len(self.strain_energy_cfg.invariants_config) <= 2 else "анизотропный"

        # Валидация согласованности
        if self.architecture_type == "изотропный" and len(self.strain_energy_cfg.invariants_config) > 2:
            raise ValueError("Изотропная архитектура поддерживает только 2 инварианта (I1, I2)")
        if self.architecture_type == "анизотропный" and len(self.strain_energy_cfg.invariants_config) != 4:
            raise ValueError("Анизотропная архитектура ожидает ровно 4 инварианта (I1, I2, I4, I5)")


# Обновляем __all__ для экспорта
__all__.extend(["ModelArchitectureConfig"])

# Импортируем модели только после объявления __all__, чтобы избежать циклов
from .CANN import ModelArchitecture_I2, ModelArchitecture_I5  # noqa: E402  pylint: disable=C0413


class ArchitectureFactory:
    """Утилитарный класс для динамического создания сетевых блоков."""

    # ------------------------- Invariant Net ---------------------------------
    @staticmethod
    def _create_inv_net_class(cfg: InvNetConfig) -> Type[nn.Module]:
        """Динамически создаёт класс SingleInvNet на основе `BaseInvNet`."""

        cfg.validate()

        class _CustomSingleInvNet(BaseInvNet):
            def __init__(self, bias: float = cfg.bias):
                super().__init__(
                    activation_functions=cfg.activation_functions,
                    polynomial_degree=cfg.polynomial_degree,
                    bias=bias,
                )

        # Делаем читаемое имя класса (например, SingleInvNet_linear_exp_deg2)
        act_token = "_".join(cfg.activation_functions)
        _CustomSingleInvNet.__name__ = f"SingleInvNet_{act_token}_deg{cfg.polynomial_degree}"
        return _CustomSingleInvNet

    # -------------------------- Strain Energy -------------------------------
    @staticmethod
    def create_strain_energy(cfg: StrainEnergyConfig) -> BaseStrainEnergy:
        """Создаёт экземпляр `BaseStrainEnergy` с кастомным `SingleInvNet`.

        Parameters
        ----------
        cfg : StrainEnergyConfig
            Конфигурация модели.
        """
        cfg.validate()
        inv_cls = ArchitectureFactory._create_inv_net_class(cfg.inv_net_config)
        return BaseStrainEnergy(SingleInvNet=inv_cls, invariants_config=np.array(cfg.invariants_config))

    @staticmethod
    def create_model_architecture(config: ModelArchitectureConfig) -> nn.Module:
        """Создаёт полную архитектуру модели по конфигурации."""
        config.validate()
        
        # Создаём Psi-модель
        psi_model = ArchitectureFactory.create_strain_energy(config.strain_energy_cfg)
        
        # Выбираем архитектуру
        if config.architecture_type == "анизотропный":
            return ModelArchitecture_I5(
                psi_model,
                setAl=config.setAl,
                init=config.init_alpha,
            )
        elif config.architecture_type == "изотропный":
            return ModelArchitecture_I2(psi_model)
        else:
            raise ValueError(f"Неизвестный тип архитектуры: {config.architecture_type}") 