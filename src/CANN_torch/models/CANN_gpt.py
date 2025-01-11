import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.potential_zoo import get_psi
# from ..utils.potential_zoo import get_psi

def flatten(l):
    return [item for sublist in l for item in sublist]


def activation_exp(x):
    return 1.0 * (torch.exp(x) - 1.0)


def activation_ln(x):
    return -1.0 * torch.log(1.0 - x)


# Define Invariant building-blocks
class SingleInvNet4(nn.Module):
    def __init__(self, bias=3):
        super(SingleInvNet4, self).__init__()
        self.layer1 = nn.Linear(1, 1, bias=False)
        self.layer2 = nn.Linear(1, 1, bias=False)
        self.layer3 = nn.Linear(1, 1, bias=False)
        self.layer4 = nn.Linear(1, 1, bias=False)

        self.terms_count = 4
        self.bias = bias
        nn.init.uniform_(self.layer1.weight, 0.01, 1.0)
        nn.init.uniform_(self.layer2.weight, 0.01, 0.1)
        nn.init.uniform_(self.layer3.weight, 0.01, 1.0)
        nn.init.uniform_(self.layer4.weight, 0.01, 0.1)
        # nn.init.constant_(self.layer2.weight, 0.1)
        # nn.init.constant_(self.layer3.weight, 1.0)
        # nn.init.constant_(self.layer4.weight, 0.1)

    def forward(self, I_in):
        I_ref = I_in - self.bias
        I_w11 = self.layer1(I_ref)
        I_w21 = activation_exp(self.layer2(I_ref))
        I_w31 = self.layer3(I_ref ** 2)
        I_w41 = activation_exp(self.layer4(I_ref ** 2))

        return torch.cat((I_w11, I_w21, I_w31, I_w41), dim=1)

    def clamp_weights(self):
        with torch.no_grad():
            for param in self.parameters():
                param.clamp_(min=0)


# Define Invariant building-blocks
class SingleInvNet6(nn.Module):
    def __init__(self, bias=3):
        super(SingleInvNet6, self).__init__()
        self.layer1 = nn.Linear(1, 1, bias=False)
        self.layer2 = nn.Linear(1, 1, bias=False)
        self.layer3 = nn.Linear(1, 1, bias=False)
        self.layer4 = nn.Linear(1, 1, bias=False)
        self.layer5 = nn.Linear(1, 1, bias=False)
        self.layer6 = nn.Linear(1, 1, bias=False)

        self.terms_count = 6
        self.bias = bias
        nn.init.uniform_(self.layer1.weight, 0.01, 1.0)
        nn.init.uniform_(self.layer2.weight, 0.01, 0.1)
        nn.init.uniform_(self.layer3.weight, 0.01, 1.0)
        nn.init.uniform_(self.layer4.weight, 0.01, 1.0)
        nn.init.uniform_(self.layer5.weight, 0.01, 0.1)
        nn.init.uniform_(self.layer6.weight, 0.01, 1.0)
        # nn.init.constant_(self.layer2.weight, 0.1)
        # nn.init.constant_(self.layer3.weight, 1.0)
        # nn.init.constant_(self.layer4.weight, 0.1)

    def forward(self, I_in):
        I_ref = I_in - self.bias
        I_w11 = self.layer1(I_ref)
        I_w21 = activation_exp(self.layer2(I_ref))
        I_w31 = activation_ln(self.layer3(I_ref))
        I_w41 = self.layer4(I_ref ** 2)
        I_w51 = activation_exp(self.layer5(I_ref ** 2))
        I_w61 = activation_ln(self.layer6(I_ref))

        return torch.cat((I_w11, I_w21, I_w31, I_w41, I_w51, I_w61), dim=1)

    def clamp_weights(self):
        with torch.no_grad():
            for param in self.parameters():
                param.clamp_(min=0)


# Define CANN Strain energy
class StrainEnergy_i5(nn.Module):
    def __init__(self, SingleInvNet=SingleInvNet4):
        super(StrainEnergy_i5, self).__init__()
        self.I1_net = SingleInvNet(bias=3)
        self.I2_net = SingleInvNet(bias=3)
        self.I4_net = SingleInvNet(bias=1)
        self.I5_net = SingleInvNet(bias=1)

        self.terms_count = self.I1_net.terms_count
        self.invariants_count = sum([1 for module in self.modules() if "SingleInvNet" in module._get_name()])
        self.all_terms_count = self.terms_count * self.invariants_count

        self.invariants = torch.zeros(self.invariants_count)
        self.final_layer = nn.Linear(self.all_terms_count, 1, bias=False)
        # nn.init.constant_(self.final_layer.weight, 1.0)
        nn.init.uniform_(self.final_layer.weight, 0.01, 1.0)

        # nn.init.xavier_normal_(self.final_layer.weight)

    def forward(self, I1_ref, I2_ref, I4_ref, I5_ref):
        I1_out = self.I1_net(I1_ref)
        I2_out = self.I2_net(I2_ref)
        I4_out = self.I4_net(I4_ref)
        I5_out = self.I5_net(I5_ref)

        ALL_I_out = torch.cat((I1_out, I2_out, I4_out, I5_out), dim=1)
        W_ANN = self.final_layer(ALL_I_out)

        return W_ANN

    def clamp_weights(self):
        with torch.no_grad():
            for param in self.parameters():
                param.clamp_(min=0)


# Gradient function
def myGradient(a, b):
    return torch.autograd.grad(outputs=a, inputs=b, grad_outputs=torch.ones_like(a), create_graph=True)[0]


# Definition of stress
def Stress_xx_I5_BT(inputs):
    (dPsidI1, dPsidI2, dWdI4, dWdI5, Stretch, Stretch_z, I1, h11, h11_i5) = inputs

    one = torch.tensor(1.0)
    two = torch.tensor(2.0)
    four = torch.tensor(4.0)

    stress_1 = two * (dPsidI1 + I1 * dPsidI2) * (Stretch ** two - Stretch_z ** two)
    stress_2 = two * dPsidI2 * (Stretch_z ** four - Stretch ** four)
    stress_3 = two * dWdI4 * h11
    stress_4 = four * dWdI5 * h11_i5

    return torch.tensor(stress_1 + stress_2 + stress_3 + stress_4, requires_grad=True)


def stress_calc_bx(inputs):
    dPsidI1, dPsidI2, dPsidI4, dPsidI5, Stretch1, Stretch2, al = inputs

    one = torch.tensor(1.0, dtype=torch.float32)
    two = torch.tensor(2.0, dtype=torch.float32)
    three = torch.tensor(3.0, dtype=torch.float32)
    four = torch.tensor(4.0, dtype=torch.float32)

    # minus = two * (dPsidI1 * 1 / (Stretch ** 2) + dPsidI2 * 1 / (Stretch ** 3))
    # stress = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus
    first_11 = (Stretch1 - one / (Stretch1 ** two * Stretch2 ** two))
    second_11 = (Stretch1 * Stretch2 ** two + one / (Stretch1 * Stretch2 ** two) - one / (Stretch1 ** two) - one / (
                Stretch2 ** two))
    fourth_11 = Stretch1 * torch.cos(al) ** two
    fifth_11 = Stretch1 ** three * torch.cos(al) ** two

    first_22 = (Stretch2 - one / (Stretch1 ** two * Stretch2 ** two))
    second_22 = (Stretch1 ** two * Stretch2 + one / (Stretch1 ** two * Stretch2) - one / (Stretch1 ** two) - one / (
                Stretch2 ** two))
    fourth_22 = Stretch2 * torch.sin(al) ** two
    fifth_22 = Stretch2 ** three * torch.sin(al) ** two

    P11 = two * (first_11 * dPsidI1 + second_11 * dPsidI2 + fourth_11 * dPsidI4 + two * fifth_11 * dPsidI5)
    P22 = two * (first_22 * dPsidI1 + second_22 * dPsidI2 + fourth_22 * dPsidI4 + two * fifth_22 * dPsidI5)
    return torch.cat((P11, P22), dim=1)


# Define H-layer
class H_Layer_FungBiax_I4I5(nn.Module):
    def __init__(self, nameU, setAl, init):
        super(H_Layer_FungBiax_I4I5, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(init), requires_grad=setAl)
        self.alpha.data.clamp_(min=0)  # Ensure non-negativity

    def forward(self, lam):
        # lamx, lamy = lam.split(1)
        lamx, lamy = lam.split(1, dim=1)
        al = F.relu(self.alpha)
        # al = torch.pi / torch.tensor(4.)
        h11_i4 = lamx ** 2 * (torch.cos(al) ** 2)
        h22_i4 = lamy ** 2 * (torch.sin(al) ** 2)

        h11_i5 = (lamx ** 4) * (torch.cos(al) ** 2)
        h22_i5 = (lamy ** 4) * (torch.sin(al) ** 2)

        return h11_i4, h22_i4, h11_i5, h22_i5, al


# Complete model architecture definition
class ModelArchitecture_I5(nn.Module):
    def __init__(self, Psi_model, setAl, init, initial_weight=0.1):
        super(ModelArchitecture_I5, self).__init__()
        self.Psi_model = Psi_model
        self.H_layer = H_Layer_FungBiax_I4I5('alpha', setAl, init)
        self.potential_constants = None
        self.terms_count = Psi_model.terms_count

        # for layer in self.modules():
        #     classname = layer.__class__.__name__
        #
        #     if classname.find('Linear') != -1:
        #         # get the number of the inputs
        #         torch.nn.init.uniform_(layer.weight, a=0.01, b=initial_weight)
        #         layer.weight.data = torch.clamp(layer.weight.data, min=0)

    def forward(self, inputs):
        Stretch_x, Stretch_y = inputs[:2]
        Stretch_x = Stretch_x.unsqueeze(1).requires_grad_(True)
        Stretch_y = Stretch_y.unsqueeze(1).requires_grad_(True)

        Stretch_z = 1 / (Stretch_x * Stretch_y)
        I1_BT = Stretch_x ** 2 + Stretch_y ** 2 + Stretch_z ** 2
        I2_BT = (Stretch_x ** 2) * (Stretch_y ** 2) + 1 / Stretch_x ** 2 + 1 / Stretch_y ** 2

        h11, h22, h11_i5, h22_i5, al = self.H_layer(torch.cat((Stretch_x, Stretch_y), dim=1))
        # h11, h22, h11_i5, h22_i5 = self.H_layer(torch.cat((Stretch_x, Stretch_y)))
        I4_BT = h11 + h22
        I5_BT = h11_i5 + h22_i5

        Psi_BT = self.Psi_model(I1_BT, I2_BT, I4_BT, I5_BT)

        dWI1_BT = myGradient(Psi_BT, I1_BT)
        dWdI2_BT = myGradient(Psi_BT, I2_BT)
        dWdI4_BT = myGradient(Psi_BT, I4_BT)
        dWdI5_BT = myGradient(Psi_BT, I5_BT)

        # Stress_xx_BT = Stress_xx_I5_BT(
        #     (dWI1_BT, dWdI2_BT, dWdI4_BT, dWdI5_BT, Stretch_x, Stretch_z, I1_BT, h11, h11_i5))
        # Stress_yy_BT = Stress_xx_I5_BT(
        #     (dWI1_BT, dWdI2_BT, dWdI4_BT, dWdI5_BT, Stretch_y, Stretch_z, I1_BT, h22, h22_i5))
        self.get_weights()
        return stress_calc_bx((dWI1_BT, dWdI2_BT, dWdI4_BT, dWdI5_BT, Stretch_x, Stretch_y, al))
        # return torch.cat((Stress_xx_BT, Stress_yy_BT), dim=1)

    def get_weights(self):
        w1 = []
        w2 = []
        for k in self.state_dict():
            if "I" in k:
                w1.append(self.state_dict()[k].squeeze().item())
            elif "final" in k:
                w2 = self.state_dict()[k].squeeze()
        self.potential_constants = torch.tensor([w1, w2])

    def get_potential(self, p=3):

        if self.potential_constants is None:
            self.get_weights()

        w = self.potential_constants
        potential_str = get_psi(w, terms=self.Psi_model.all_terms_count, p=p)
        return potential_str

    def extract_weights_as_blocks(self, w=None, precision=6):
        if not w:
            self.get_weights()
            w = self.potential_constants
        p = precision
        blocks = {
            "(I1 - 3)": w[1, 0] * w[0, 0],
            "e^(I1 - 3) - 1": (w[1, 1], w[0, 1]),
            "(I1 - 3)^2": w[1, 2] * w[0, 2],
            "e^(I1 - 3)^2 - 1": (w[1, 3], w[0, 3]),
            "(I2 - 3)": w[1, 4] * w[0, 4],
            "e^(I2 - 3) - 1": (w[1, 5], w[0, 5]),
            "(I2 - 3)^2": w[1, 6] * w[0, 6],
            "e^(I2 - 3)^2 - 1": (w[1, 7], w[0, 7]),
            "(I4 - 3)": w[1, 8] * w[0, 8],
            "e^(I4 - 3) - 1": (w[1, 9], w[0, 9]),
            "(I4 - 3)^2": w[1, 10] * w[0, 10],
            "e^(I4 - 3)^2 - 1": (w[1, 11], w[0, 11]),
            "(I5 - 3)": w[1, 12] * w[0, 12],
            "e^(I5 - 3) - 1": (w[1, 13], w[0, 13]),
            "(I5 - 3)^2": w[1, 14] * w[0, 14],
            "e^(I5 - 3)^2 - 1": (w[1, 15], w[0, 15]),
        }
        # Форматирование значений
        formatted_blocks = {
            block: f"{weight[0]:.{p}f}, {weight[1]:.{p}f}" if isinstance(weight, tuple) else f"{weight:.{p}f}" for
            block, weight in blocks.items()}
        return formatted_blocks

    def clamp_weights(self):
        with torch.no_grad():
            for param in self.parameters():
                param.clamp_(min=0)

    def calc_regularization(self, l=2):
        """

        :param l: power
        :return: sum of potential coefficients to the power of p
        """
        return torch.sum(self.potential_constants ** l)

    def calc_l1(self):
        return torch.sum(torch.abs(self.potential_constants))


# Example usage
# psi_model = StrainEnergy_i5()  # Initialize your Psi model here
# model = ModelArchitecture_I5(psi_model, setAl=True, init=0.1)
# Stretch_x = torch.tensor([[1.0]], requires_grad=True)
# Stretch_y = torch.tensor([[1.0]], requires_grad=True)
# output = model(Stretch_x, Stretch_y)
# print(output)

if __name__ == "__main__":
    w_16 = np.ones((2, 16))
    # print(get_psi(w_16, 16))

    psi_model = StrainEnergy_i5(SingleInvNet=SingleInvNet6)
    model = ModelArchitecture_I5(psi_model, setAl=True, init=torch.pi / 4)

    print(model)
    # for module in model.modules():
    #     print(module._get_name())
    #     if module._get_name() == "Linear":
    #         for w in module.parameters():
    #             print(w)
    # for p in model.parameters():
    #     for v in p.values():
    #         print(v)
    # print(model.get_potential())
