"""
Neural network models for constitutive equations
"""

from .CNN import StrainEnergyCANN, StrainEnergyCANN_Ani
from .CANN_gpt import ModelArchitecture_I5
from .potential_zoo import *

__all__ = [
    "StrainEnergyCANN",
    "StrainEnergyCANN_Ani",
    "ModelArchitecture_I5"
] 