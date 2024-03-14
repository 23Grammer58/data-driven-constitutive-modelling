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
import sympy as sp


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

class SingleInvNet1(nn.Module):
    def __init__(self, input_size, idi, device, l2=0.001):
        """
        y=xA^T+b (b=0)
        :param input_size: input data size
        :param idi: index of neuron
        :param l2: L2 regularization coefficient
        """
        super().__init__()

        self.l2 = l2
        self.idi = idi

        self.w11 = nn.Linear(input_size, 1, bias=False).to(device)

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        i_ref = i - 3.0

        w11_out = self.w11(i_ref)

        out = torch.cat(w11_out)
        return out





class SingleInvNet2(nn.Module):
    def __init__(self, input_size, idi, device, l2=0.001):
        """
        y=xA^T+b (b=0)
        :param input_size: input data size
        :param idi: index of neuron
        :param l2: L2 regularization coefficient
        """
        super().__init__()

        self.l2 = l2
        self.idi = idi

        self.w11_linear = nn.Linear(input_size, 1, bias=False).to(device)
        self.w21_square = nn.Linear(input_size, 1, bias=False).to(device)

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        i_ref = i - 3.0

        w11_out = self.w11_linear(i_ref)

        i_sqr = torch.mul(i_ref, i_ref)

        w21_out = self.w21_square(i_sqr)

        out = torch.cat((w11_out, w21_out))
        return out




class SingleInvNet4(nn.Module):
    def __init__(self, input_size, idi, device, l2=0.001):
        """
        y=xA^T+b (b=0)
        :param input_size: input data size
        :param idi: index of neuron
        :param l2: L2 regularization coefficient
        """
        super().__init__()
        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.5)
        # self.dropout3 = nn.Dropout(0.1)
        # self.dropout4 = nn.Dropout(0.5)

        self.l2 = l2
        self.idi = idi

        self.w11 = nn.Linear(input_size, 1, bias=False).to(device)
        self.w21 = nn.Linear(input_size, 1, bias=False).to(device)
        self.w31 = nn.Linear(input_size, 1, bias=False).to(device)
        self.w41 = nn.Linear(input_size, 1, bias=False).to(device)

        self.activation_Exp = activation_Exp

        # self.w11.weight_regularizer = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.w21.weight_regularizer = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.w31.weight_regularizer = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.w41.weight_regularizer = nn.Parameter(torch.Tensor(1), requires_grad=False)

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        i_ref = i - 3.0

        w11_out = self.w11(i_ref)
        # w11_out = self.dropout1(w11_out)

        w21_out = self.activation_Exp(self.w21(i_ref))
        # w21_out = self.dropout2(w21_out)

        i_sqr = torch.mul(i_ref, i_ref)

        w31_out = self.w31(i_sqr)
        # w31_out = self.dropout3(w31_out)

        w41_out = self.activation_Exp(self.w41(i_sqr))
        # w41_out = self.dropout4(w41_out)
        # out = torch.cat((w11_out, w21_out, w31_out, w41_out), dim=1)
        out = torch.cat((w11_out, w21_out, w31_out, w41_out))
        return out


class StrainEnergyCANN(nn.Module):
    def __init__(self, batch_size, device):
        super().__init__()
        self.potential_constants = np.zeros(16)
        self.device = device
        self.batch_size = batch_size
        self.single_inv_net1 = SingleInvNet2(batch_size, 0, device)
        self.single_inv_net2 = SingleInvNet2(batch_size, 4, device)
        self.wx2 = nn.Linear(4, 1, bias=False)

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
        psi_model = torch.cat((i1_out, i2_out))
        # out = torch.cat((i1_out, i2_out), dim=1)
        # out = out.view(-1, 8)  # Изменение формы перед применением линейного слоя
        psi_model = self.wx2(psi_model)
        return psi_model

    def get_potential(self):
        """

        :return: [weights of model]
        """
        params = []
        for id, weights in enumerate(self.parameters()):
            # print(f"id = {id}, weight = {weights}")
            if id == 4:
                weights = weights.tolist()
                for weight_last_layer in weights[0]:
                    # print(weight_last_layer)
                    params.append(weight_last_layer)
                break
            weight = weights.detach().to('cpu').numpy().copy()
            params.append(weight[0, 0].item())
            # params.append(param.data())

        # psi = w2[0] * w1[0] * (I1 - 3) + w2[1] * (sp.exp(w1[1] * (I1 - 3)) - 1) + w2[2] * w1[2] * (I1 - 3) ** 2 + w2[
        #     3] * (sp.exp(w1[3] * (I1 - 3) ** 2) - 1) + w2[4] * w1[4] * (I2 - 3) + w2[5] * (
        #                   sp.exp(w1[5] * (I2 - 3)) - 1) + w2[6] * w1[6] * (I2 - 3) ** 2 + w2[7] * (
        #                   sp.exp(w1[7] * (I2 - 3) ** 2) - 1)
        return params


if __name__ == "__main__":

    I1, I2 = sp.symbols('I1 I2')
    w1 = sp.symbols('w11:19')  # w1, w2, ..., w8
    w2 = sp.symbols('w21:29')  # w21, w22, ..., w28

    torch.manual_seed(42)
    trained_model = StrainEnergyCANN(batch_size=1, device="cpu")
    trained_model.load_state_dict(torch.load('../pretrained_models/CNN_l2/20240226_132313_6.pth'))


    # y = model(torch.tensor([[0.1]]), torch.tensor([[0.1]]))
    # y = model(torch.tensor([[0.1], [0.1]]))
    # x = [torch.tensor([[0.1], [0.1]]), torch.tensor([[0.1], [0.1]])]
    # y = model(x)
    # print(y)


    psi = (w2[0] * w1[0] * (I1 - 3)
           + w2[1] * (sp.exp(w1[1] * (I1 - 3)) - 1)
           + w2[2] * w1[2] * (I1 - 3) ** 2
           + w2[3] * (sp.exp(w1[3] * (I1 - 3) ** 2) - 1)
           + w2[4] * w1[4] * (I2 - 3)
           + w2[5] * (sp.exp(w1[5] * (I2 - 3)) - 1)
           + w2[6] * w1[6] * (I2 - 3) ** 2
           + w2[7] * (sp.exp(w1[7] * (I2 - 3) ** 2) - 1))

    coefs = trained_model.get_potential()
    print(coefs)
    l2_reg = None
    for param in coefs:
        print(param)
        if l2_reg is None:
            l2_reg = param**2
        else:
            l2_reg = l2_reg + param**2
        print("l2 reg = ", l2_reg)
    print(l2_reg)

    w1 = coefs[:8]
    w2 = coefs[8:]
    psi = psi.subs([(w1, w2)])

    # print(sp.pretty(psi))

    # Вывод функции ψ(I1, I2)
    print(psi)
    # Получение графа вычислений
    # dot = make_dot(y, params=dict(model.named_parameters()))
    # dot.render('CNN', format='png')

"""
TODO:
1) Try bilinear layer in wx2 

"""