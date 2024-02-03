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
import numpy as np

# Гиперпараметры
# input_size = 2  # Размерность входных данных
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
    def __init__(self, input_size, idi, device):
        """
        y=xA^T+b (b=0)
        :param input_size:
        :param idi:
        :param L2:
        """
        super().__init__()
        self.w11 = nn.Linear(input_size, 1, bias=False).to(device)
        self.w21 = nn.Linear(input_size, 1, bias=False).to(device)
        self.w31 = nn.Linear(input_size, 1, bias=False).to(device)
        self.w41 = nn.Linear(input_size, 1, bias=False).to(device)
        self.activation_Exp = activation_Exp
        # self.L2 = L2
        self.idi = idi

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        i_ref = i - 3.0
        w11_out = self.w11(i_ref)
        w21_out = self.activation_Exp(self.w21(i_ref))
        i_sqr = torch.mul(i_ref, i_ref)
        w31_out = self.w31(i_sqr)
        w41_out = self.activation_Exp(self.w41(i_sqr))
        # out = torch.cat((w11_out, w21_out, w31_out, w41_out), dim=1)
        out = torch.cat((w11_out, w21_out, w31_out, w41_out))
        return out


class StrainEnergyCANN(nn.Module):
    def __init__(self, batch_size, device):
        super().__init__()
        self.potential_constants = np.zeros(16)
        self.device = device
        self.batch_size = batch_size
        self.single_inv_net1 = SingleInvNet4(batch_size, 0, device)
        self.single_inv_net2 = SingleInvNet4(batch_size, 4, device)
        self.wx2 = nn.Linear(8, 1, bias=False)

    # def forward(self, i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
    def forward(self, invariants:torch.Tensor) -> torch.Tensor:
        if self.batch_size == 1:
            invariants = invariants[0].clone().detach().to(self.device)
            i1 = invariants[0]
            i2 = invariants[1]

            i1_out = self.single_inv_net1(i1.unsqueeze(0))
            i2_out = self.single_inv_net2(i2.unsqueeze(0))
        else:
            i1 = invariants[:, 0]
            i2 = invariants[:, 1]

            i1_out = self.single_inv_net1(i1.unsqueeze(1).unsqueeze(0))
            i2_out = self.single_inv_net2(i2.unsqueeze(1).unsqueeze(0))
        out = torch.cat((i1_out, i2_out))
        # out = torch.cat((i1_out, i2_out), dim=1)
        # out = out.view(-1, 8)  # Изменение формы перед применением линейного слоя
        out = self.wx2(out)
        return out

    def get_potential(self):
        """

        :return: [weights of model]
        """
        params = []
        for weights in self.parameters():
            weight = weights.detach().to('cpu').numpy().copy()
            params.append(weight)
            # params.append(param.data())

        return params


if __name__ == "__main__":
    torch.manual_seed(42)
    model = StrainEnergyCANN(batch_size=1)
    # y = model(torch.tensor([[0.1]]), torch.tensor([[0.1]]))
    y = model(torch.tensor([[0.1], [0.1]]))
    # x = [torch.tensor([[0.1], [0.1]]), torch.tensor([[0.1], [0.1]])]
    # y = model(x)
    print(y)

    # Получение графа вычислений
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render('CNN', format='png')

"""
TODO:
1) Try bilinear layer in wx2 

"""