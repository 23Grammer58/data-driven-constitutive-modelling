"""
Neural network models for constitutive equations
"""

# from .CNN import StrainEnergyCANN_Ani
# from .CANN import ModelArchitecture_I5, H_Layer_FungBiax_I4I5, SingleInvNet4, SingleInvNet6
# from .potential_zoo import *
#
# __all__ = [
#     "SingleInvNet4",
#     "SingleInvNet6",
#     "H_Layer_FungBiax_I4I5",
#     "StrainEnergyCANN_Ani",
#     "ModelArchitecture_I5"
# ]

from .CNN import StrainEnergyCANN_Ani
from .CANN import *  # Импортируем всё из CANN
from .potential_zoo import *
from .architecture_factory import InvNetConfig, StrainEnergyConfig, ArchitectureFactory, ModelArchitectureConfig

# Динамически добавляем имена из CANN в __all__
import sys

# Получаем текущий модуль
current_module = sys.modules[__name__]

# Получаем все имена, определённые в CANN
cann_names = [name for name in dir(current_module) if not name.startswith('_')]

# Обновляем __all__
__all__ = [
    "StrainEnergyCANN_Ani",  # Добавляем явно, если нужно
    *cann_names,  # Добавляем все имена из CANN
    "InvNetConfig",
    "StrainEnergyConfig",
    "ArchitectureFactory",
    "ModelArchitectureConfig"
]