import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# # import tqdm
# from dataload import ExcelDataset
# from torchmetrics.regression import MeanSquaredError
from torchviz import make_dot

import os
import math
from math import exp

# Гиперпараметры
input_size = 2  # Размерность входных данных
output_size = 1  # Размерность выходных данных
hidden_size = 270  # Новое количество нейронов на слое
learning_rate = 0.001
epochs = 100


def flatten(l):
    return [item for sublist in l for item in sublist]


# Self defined activation functions for exp
def activation_Exp(x):
    return 1.0*(torch.exp(x) -1.0)


def activation_ln(x):
    return -1.0*math.log(1.0 - (x))


class SingleInvNet4(nn.Module):
    def __init__(self, input_size, idi, L2):
        super().__init__()
        self.w11 = nn.Linear(input_size, 1, bias=False)
        self.w21 = nn.Linear(input_size, 1, bias=False)
        self.w31 = nn.Linear(input_size, 1, bias=False)
        self.w41 = nn.Linear(input_size, 1, bias=False)
        self.activation_Exp = activation_Exp
        self.L2 = L2
        self.idi = idi

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        i_ref = i - 3.0
        w11_out = self.w11(i_ref)
        w21_out = self.activation_Exp(self.w21(i_ref))
        i_sqr = torch.mul(i_ref, i_ref)
        w31_out = self.w31(i_sqr)
        w41_out = self.activation_Exp(self.w41(i_sqr))
        out = torch.cat((w11_out, w21_out, w31_out, w41_out), dim=1)
        return out

class StrainEnergyCANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.single_inv_net1 = SingleInvNet4(1, 0, 0.001)
        self.single_inv_net2 = SingleInvNet4(1, 4, 0.001)
        self.wx2 = nn.Linear(8, 1, bias=False)

    def forward(self, i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
        i1_out = self.single_inv_net1(i1)
        i2_out = self.single_inv_net2(i2)
        out = torch.cat((i1_out, i2_out), dim=1)
        out = self.wx2(out)
        return out


if __name__ == "__main__":
    torch.manual_seed(42)
    model = StrainEnergyCANN()
    y = model(torch.tensor([[0.1]]), torch.tensor([[0.1]]))
    print(y)

    # Получение графа вычислений
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render('CNN', format='png')