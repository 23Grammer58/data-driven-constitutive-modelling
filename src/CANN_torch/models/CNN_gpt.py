import torch
import torch.nn as nn

# Определение функций стресса
def Stress_calc_TC(inputs):
    dPsidI1, dPsidI2, Stretch = inputs
    one = torch.tensor(1.0, dtype=torch.float32)
    two = torch.tensor(2.0, dtype=torch.float32)
    minus  = two * (dPsidI1 * 1 / (Stretch**2) + dPsidI2 * 1 / (Stretch**3))
    stress = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus
    return stress

def Stress_calc_SS(inputs):
    dPsidI1, dPsidI2, gamma = inputs
    two = torch.tensor(2.0, dtype=torch.float32)
    stress = two * gamma * (dPsidI1 + dPsidI2)
    return stress

def Stress_calc_inv(inputs):
    dPsidI1, dPsidI2, C = inputs
    I2 = 0.5 * (torch.trace(C)**2 - torch.trace(C @ C))
    dPsidI3 = torch.tensor(1.0, dtype=torch.float32)
    two = torch.tensor(2.0, dtype=torch.float32)
    dI1dC = torch.eye(3)
    dI2dC = I2 * torch.inverse(C)
    dI3dC = torch.tensor(0, dtype=torch.float32)
    stress = two * (dPsidI1 * dI1dC + dPsidI2 * dI2dC + dPsidI3 * dI3dC)
    return stress

# Определение функций активации
def activation_Exp(x):
    return 1.0 * (torch.exp(x) - 1.0)

def activation_ln(x):
    return -1.0 * torch.log(1.0 - x)

class MyLayer(nn.Module):
    def __init__(self, my_function):
        super(MyLayer, self).__init__()
        self.my_function = my_function

    def forward(self, x):
        return self.my_function(x)

# Определение класса модели
class SingleInvNet6(nn.Module):
    def __init__(self, input_size, idi, device, l2=0.001):
        super().__init__()

        self.l2 = l2
        self.idi = idi

        self.w11 = nn.Linear(input_size, 1, bias=False).to(device)
        self.w21 = nn.Linear(input_size, 1, bias=False).to(device)
        self.w31 = nn.Linear(input_size, 1, bias=False).to(device)
        self.w41 = nn.Linear(input_size, 1, bias=False).to(device)
        self.w51 = nn.Linear(input_size, 1, bias=False).to(device)
        self.w61 = nn.Linear(input_size, 1, bias=False).to(device)

        self.activation_Exp = activation_Exp
        self.activation_ln = activation_ln

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        i_ref = i - 3.0

        w11_out = self.w11(i_ref)
        w21_out = self.activation_Exp(self.w21(i_ref))
        w31_out = self.activation_ln(self.w31(i_ref))

        i_sqr = torch.mul(i_ref, i_ref)

        w41_out = self.w41(i_sqr)
        w51_out = self.activation_Exp(self.w51(i_sqr))
        w61_out = self.activation_ln(self.w61(i_ref))

        out = torch.cat((w11_out, w21_out, w31_out, w41_out, w51_out, w61_out), dim=1)
        return out

    def clamp_weights(self):
        with torch.no_grad():
            for param in self.parameters():
                param.clamp_(min=0)

class StrainEnergyCANN_C(nn.Module):
    def __init__(self, batch_size=1, device="cpu", stress_calc=Stress_calc_inv, dtype=torch.float32, term_count=6, invariants_count=2):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.potential_constants = None
        self.single_inv_net1 = SingleInvNet6(1, 0, device)
        self.single_inv_net2 = SingleInvNet6(1, 6, device)
        self.wx2 = nn.Linear(term_count * invariants_count, 1, bias=False).requires_grad_(True)
        self.psi_model = None
        self.stress_TC = MyLayer(Stress_calc_TC)
        self.stress_SS = MyLayer(Stress_calc_SS)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight, a=0.0, b=0.2)
            module.weight.data = torch.clamp(module.weight.data, min=0)

    def forward(self, inputs) -> torch.Tensor:
        lam, i1, i2, F, exp_type = inputs

        i1 = i1.requires_grad_(True)
        i2 = i2.requires_grad_(True)

        self.i1 = i1
        self.i2 = i2

        i1_out = self.single_inv_net1(i1.unsqueeze(1))
        i2_out = self.single_inv_net2(i2.unsqueeze(1))

        inv_out = torch.cat((i1_out, i2_out), dim=1)
        self.psi_model = self.wx2(inv_out)

        dpsi_dI1, dpsi_dI2 = self.calc_dpsi_di()

        if exp_type in [1.5, 0.5, ("Compression", ), ("Tensile", )]:
            return self.stress_TC((dpsi_dI1, dpsi_dI2, lam))
        elif exp_type in [1.0, ("Shear", )]:
            return self.stress_SS((dpsi_dI1, dpsi_dI2, lam))
        else:
            raise TypeError("Wrong type of exp_type!")

    def calc_dpsi_di(self):
        dpsi_dI1, dpsi_dI2 = torch.autograd.grad(
            outputs=self.psi_model,
            inputs=(self.i1, self.i2),
            grad_outputs=torch.ones_like(self.psi_model),
            create_graph=True
        )
        return dpsi_dI1, dpsi_dI2

    def clamp_weights(self):
        self.single_inv_net1.clamp_weights()
        self.single_inv_net2.clamp_weights()
        with torch.no_grad():
            self.wx2.weight.clamp_(min=0)