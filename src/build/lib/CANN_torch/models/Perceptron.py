import torch
import torch.nn as nn
from math import exp


# Гиперпараметры
input_size = 2  # Размерность входных данных
output_size = 1  # Размерность выходных данных
hidden_size = 270  # Новое количество нейронов на слое
learning_rate = 0.001
epochs = 100


class QuadricNeuronR2(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.randn(3, 2))
        self.w = nn.Parameter(torch.randn(1, 2))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        isqr = torch.mul(i, i)
        qi = torch.mm(self.q, isqr.t())
        wi = torch.mm(self.w, i.t())
        o = torch.add(torch.add(qi, wi), self.b)
        return o


class ConstitutiveNN(nn.Module):
    def __init__(self,
                 hidden_size=hidden_size,
                 multiactivation=True):
        super(ConstitutiveNN, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc_quad = nn.Linear(input_size, hidden_size) ** 2
        self.linear_exp = exp(x) - 1
        if multiactivation:
            self.activations = [nn.Tanh(), nn.ReLU(), nn.Sigmoid()]
        else:
            self.activations = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        for activation in self.activations:
            x = activation(x)
        return x


# Создание однослойного перцептрона
class SingleLayerPerceptron(nn.Module):
    def __init__(self, multiactivation=True):
        super(SingleLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_end = nn.Linear(hidden_size*2, output_size)

        # self.quadric = QuadricNeuronR2()
        # if multiactivation:
        #     self.activations = [nn.Tanh(), nn.ReLU(), nn.Sigmoid()]
        # else:
        #     self.activations = nn.Tanh()
        self.activation = nn.Tanh()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc1(torch.mul(x, x))
        x1 = self.activation(x1)
        x2 = self.activation(x2)
        # x1 = self.fc_end(x1)
        x = torch.cat((x1, x2))
        x = self.fc_end(x)
        # for activation in self.activations:
        #     x = activation(x)
        return x
