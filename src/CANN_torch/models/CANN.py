import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils.potential_zoo import stress_calc_bx

# from CANN_torch.models.potential_zoo import get_psi
from CANN_torch.utils import compute_stress, compute_invariants, get_psi
# from ..utils.potential_zoo import get_psi

def flatten(l):
    return [item for sublist in l for item in sublist]


def activation_exp(x):
    return 1.0 * (torch.exp(x) - 1.0)


def activation_ln(x):
    return -1.0 * torch.log(1.0 - x)


# Define Invariant building-blocks
class SingleInvNet4_old(nn.Module):
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


class BaseInvNet(nn.Module):
    def __init__(self, activation_functions, polynomial_degree, bias=3):
        """
        Базовый класс для инвариантных сетей.

        :param activation_functions: Список функций активации, например, ["linear", "exp", "ln"].
        :param polynomial_degree: Степень полинома для входных данных.
        :param bias: Смещение для входных данных.
        """
        super(BaseInvNet, self).__init__()
        self.activation_functions = activation_functions
        self.polynomial_degree = polynomial_degree
        self.bias = bias

        # Количество терминов = количество функций активации * степень полинома
        self.terms_count = len(activation_functions) * polynomial_degree

        # Создаем линейные слои для каждого термина
        self.layers = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(self.terms_count)])
        self.init_weights()  # Инициализация весов

    def init_weights(self):
        # Инициализация весов каждого слоя
        for layer in self.layers:
            nn.init.uniform_(layer.weight, 0.01, 1.0)

    def forward(self, I_in):
        # Вычисление выхода для каждого термина
        I_ref = I_in - self.bias  # Смещение входных данных
        outputs = []

        for i in range(1, self.polynomial_degree + 1):
            poly_term = I_ref ** i  # Полиномиальный член (I_ref^1, I_ref^2, ...)
            for activation in self.activation_functions:
                # Применяем соответствующую функцию активации
                if activation == "linear":
                    outputs.append(self.layers[len(outputs)](poly_term))
                elif activation == "exp":
                    outputs.append(activation_exp(self.layers[len(outputs)](poly_term)))
                elif activation == "ln":
                    outputs.append(activation_ln(self.layers[len(outputs)](poly_term)))
                else:
                    raise ValueError(f"Unknown activation function: {activation}")

        return torch.cat(outputs, dim=1)  # Объединяем результаты

    def clamp_weights(self):
        # Ограничение весов (неотрицательные значения)
        with torch.no_grad():
            for param in self.parameters():
                param.clamp_(min=0)


class SingleInvNet3(BaseInvNet):
    def __init__(self, bias=3):
        activation_functions = ["linear", "exp", "ln"]
        polynomial_degree = 1
        super(SingleInvNet3, self).__init__(activation_functions, polynomial_degree, bias=bias)


class SingleInvNet4(BaseInvNet):
    def __init__(self, bias=3):
        activation_functions = ["linear", "exp"]
        polynomial_degree = 2
        super(SingleInvNet4, self).__init__(activation_functions, polynomial_degree, bias=bias)


class SingleInvNet6(BaseInvNet):
    def __init__(self, bias=3):
        activation_functions = ["linear", "exp", "ln"]
        polynomial_degree = 2
        super(SingleInvNet6, self).__init__(activation_functions, polynomial_degree, bias=bias)


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


class StrainEnergy_i2(nn.Module):
    def __init__(self, SingleInvNet=SingleInvNet4):
        super(StrainEnergy_i2, self).__init__()
        self.I1_net = SingleInvNet(bias=3)
        self.I2_net = SingleInvNet(bias=3)

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

        ALL_I_out = torch.cat((I1_out, I2_out), dim=1)
        W_ANN = self.final_layer(ALL_I_out)

        return W_ANN

    def clamp_weights(self):
        with torch.no_grad():
            for param in self.parameters():
                param.clamp_(min=0)


class BaseStrainEnergy(nn.Module):
    """
    Базовый класс для расчёта энергии деформации.

    Параметры:
        invariants_config (list или tuple): последовательность значений смещений (bias) для каждого инварианта.
            Например, [3, 3, 1, 1] создаст 4 инвариантных сети с соответствующими значениями bias.
        SingleInvNet (nn.Module): класс нейронной сети для одного инварианта.
            Он должен принимать аргумент bias при инициализации и иметь атрибут terms_count.
    """

    def __init__(self, SingleInvNet, invariants_config=np.ones(2)*3):
        super(BaseStrainEnergy, self).__init__()
        #
        # if not invariants_config:
        #     invariants_config =
        # Создаем список инвариантных сетей с соответствующими смещениями.
        self.invariant_nets = nn.ModuleList([SingleInvNet(bias=bias) for bias in invariants_config])

        # Предполагается, что все сети имеют одинаковое число терминов.
        self.terms_count = self.invariant_nets[0].terms_count
        self.invariants_count = len(self.invariant_nets)
        self.all_terms_count = self.terms_count * self.invariants_count

        self.invariants = torch.zeros(self.invariants_count)
        self.final_layer = nn.Linear(self.all_terms_count, 1, bias=False)
        nn.init.uniform_(self.final_layer.weight, 0.01, 1.0)

    def forward(self, *invariants_refs):
        """
        Параметры:
            invariants_refs: последовательность входных данных для каждой инвариантной сети.
                Количество переданных аргументов должно соответствовать количеству инвариантов.

        Возвращает:
            W_ANN: выходной тензор после объединения результатов всех сетей и пропуска через финальный линейный слой.
        """
        if len(invariants_refs) != self.invariants_count:
            raise ValueError(f"Ожидается {self.invariants_count} входов, получено {len(invariants_refs)}")

        outputs = [net(ref) for net, ref in zip(self.invariant_nets, invariants_refs)]
        ALL_I_out = torch.cat(outputs, dim=1)
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


def stress_calc_bx_iso(inputs):
    dPsidI1, dPsidI2, Stretch1, Stretch2 = inputs

    one = torch.tensor(1.0, dtype=torch.float32)
    two = torch.tensor(2.0, dtype=torch.float32)
    three = torch.tensor(3.0, dtype=torch.float32)
    four = torch.tensor(4.0, dtype=torch.float32)

    # minus = two * (dPsidI1 * 1 / (Stretch ** 2) + dPsidI2 * 1 / (Stretch ** 3))
    # stress = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus
    first_11 = (Stretch1 - one / (Stretch1 ** two * Stretch2 ** two))
    second_11 = (Stretch1 * Stretch2 ** two + one / (Stretch1 * Stretch2 ** two) - one / (Stretch1 ** two) - one / (
                Stretch2 ** two))

    first_22 = (Stretch2 - one / (Stretch1 ** two * Stretch2 ** two))
    second_22 = (Stretch1 ** two * Stretch2 + one / (Stretch1 ** two * Stretch2) - one / (Stretch1 ** two) - one / (
                Stretch2 ** two))

    P11 = two * (first_11 * dPsidI1 + second_11 * dPsidI2)
    P22 = two * (first_22 * dPsidI1 + second_22 * dPsidI2)
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
    def __init__(self, Psi_model, setAl, init=True, initial_weight=0.1):
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
        """Сохраняем коэффициенты потенциала в виде тензора [2, N].

        1. w1 — веса внутренних линейных слоёв (SingleInvNet.*.layers.*.weight)
        2. w2 — веса финального слоя Psi-модели (Psi_model.final_layer.weight)

        Метод теперь работает как для старых I1_net/I2_net…, так и для
        новых BaseStrainEnergy.invariant_nets.* слоёв.
        """

        state = self.state_dict()
        inner_weights: list[float] = []
        final_weights = None

        for name, param in state.items():
            # Финальный слой потенциала
            if "final_layer.weight" in name:
                final_weights = param.squeeze().clone()
                continue

            # Линейные слои инвариантных сетей (старый и новый стиль имён)
            if ("_net." in name and ".layer" in name and name.endswith("weight")) or ("invariant_nets" in name and ".layers." in name and name.endswith("weight")):
                inner_weights.append(param.squeeze().item())

        if final_weights is None:
            raise RuntimeError("Не найден вес финального слоя Psi_model.final_layer.weight – проверьте модель")

        if len(inner_weights) != final_weights.numel():
            # Для безопасности, но продолжаем работу, заполняя недостающие нулями / обрезая лишнее
            min_len = min(len(inner_weights), final_weights.numel())
            inner_weights = inner_weights[:min_len]
            final_weights = final_weights[:min_len]

        self.potential_constants = torch.stack(
            (
                torch.tensor(inner_weights, dtype=final_weights.dtype, device=final_weights.device),
                final_weights.clone(),
            )
        )

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
        if l == 1:
            return torch.sum(torch.abs(self.potential_constants))
        else:
            return torch.sum(self.potential_constants ** l)


class ModelArchitecture_I2(nn.Module):
    def __init__(self, Psi_model=None,  initial_weight=0.1):
        super(ModelArchitecture_I2, self).__init__()

        if not Psi_model:
            Psi_model = BaseStrainEnergy(SingleInvNet6, np.array([3, 3]))
        self.Psi_model = Psi_model
        self.potential_constants = None
        self.terms_count = Psi_model.terms_count

    def forward(self, inputs):
        Stretch_x, Stretch_y = inputs[:2]
        exp_type = inputs[-2]
        Stretch_x = Stretch_x.unsqueeze(1).requires_grad_(True)
        Stretch_y = Stretch_y.unsqueeze(1).requires_grad_(True)

        # Stretch_z = 1 / (Stretch_x * Stretch_y)
        # I1_BT = Stretch_x ** 2 + Stretch_y ** 2 + Stretch_z ** 2
        # I2_BT = (Stretch_x ** 2) * (Stretch_y ** 2) + 1 / Stretch_x ** 2 + 1 / Stretch_y ** 2
        I1, I2 = compute_invariants(Stretch_x, Stretch_y, exp_type)

        Psi_BT = self.Psi_model(I1, I2)

        dWI1_BT = myGradient(Psi_BT, I1)
        dWdI2_BT = myGradient(Psi_BT, I2)

        self.get_weights()
        return compute_stress(dWI1_BT, dWdI2_BT, Stretch_x, Stretch_y, exp_type)
        # return torch.cat((Stress_xx_BT, Stress_yy_BT), dim=1)

    def get_weights(self):
        """Аналогичная логика для I2-архитектуры (2 инварианта)."""

        state = self.state_dict()
        inner_weights: list[float] = []
        final_weights = None

        for name, param in state.items():
            if "final_layer.weight" in name:
                final_weights = param.squeeze().clone()
                continue
            if ("invariant_nets" in name and ".layers." in name and name.endswith("weight")) or ("I" in name and ".layer" in name and name.endswith("weight")):
                inner_weights.append(param.squeeze().item())

        if final_weights is None:
            raise RuntimeError("Не найден вес финального слоя Psi_model.final_layer.weight – проверьте модель")

        min_len = min(len(inner_weights), final_weights.numel())
        inner_weights = inner_weights[:min_len]
        final_weights = final_weights[:min_len]

        self.potential_constants = torch.stack(
            (
                torch.tensor(inner_weights, dtype=final_weights.dtype, device=final_weights.device),
                final_weights.clone(),
            )
        )

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
        # blocks = {
        #     "(I1 - 3)":            w[1, 0] * w[0, 0],
        #     "e^(I1 - 3) - 1":     (w[1, 1],  w[0, 1]),
        #     "ln(1 - (I1 - 3))":   (w[1, 2],  w[0, 2]),
        #     "(I1 - 3)^2":          w[1, 3] * w[0, 3],
        #     "e^(I1 - 3)^2 - 1":   (w[1, 4],  w[0, 4]),
        #     "ln(1 - (I1 - 3)^2)": (w[1, 5],  w[0, 5]),
        #     "(I2 - 3)":            w[1, 6] * w[0, 6],
        #     "e^(I2 - 3) - 1":     (w[1, 7],  w[0, 7]),
        #     "ln(1 - (I2 - 3))":   (w[1, 8],  w[0, 8]),
        #     "(I2 - 3)^2":          w[1, 9] * w[0, 9],
        #     "e^(I2 - 3)^2 - 1":   (w[1, 10], w[0, 10]),
        #     "ln(1 - (I2 - 3)^2)": (w[1, 11], w[0, 11]),
        #
        # }
        blocks = {
            "(I1 - 3)":            w[1, 0] * w[0, 0],
            "e^(I1 - 3) - 1":     (w[1, 1],  w[0, 1]),
            "(I1 - 3)^2":          w[1, 2] * w[0, 2],
            "e^(I1 - 3)^2 - 1":   (w[1, 3],  w[0, 3]),
            "(I2 - 3)":            w[1, 4] * w[0, 4],
            "e^(I2 - 3) - 1":     (w[1, 5],  w[0, 5]),
            "(I2 - 3)^2":          w[1, 6] * w[0, 6],
            "e^(I2 - 3)^2 - 1":   (w[1, 7], w[0,  7]),

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
        if l == 1:
            return torch.sum(torch.abs(self.potential_constants))
        else:
            return torch.sum(self.potential_constants ** l)


# Example usage
# psi_model = StrainEnergy_i5()  # Initialize your Psi model here
# model = ModelArchitecture_I5(psi_model, setAl=True, init=0.1)
# Stretch_x = torch.tensor([[1.0]], requires_grad=True)
# Stretch_y = torch.tensor([[1.0]], requires_grad=True)
# output = model(Stretch_x, Stretch_y)
# print(output)

if __name__ == "__main__":
    psi_model = StrainEnergy_i5()  # Initialize your Psi model here
    model = ModelArchitecture_I5(psi_model, setAl=True, init=0.1)
    Stretch_x = torch.tensor([[1.1], [1.2]], requires_grad=True)
    Stretch_y = torch.tensor([[1.1], [1.2]], requires_grad=True)
    output = model((Stretch_x, Stretch_y))
    print(output)
