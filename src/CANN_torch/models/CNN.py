import torchimport torch.nn as nnimport osimport mathfrom math import expimport numpy as npimport sympy as sp# Гиперпараметры# input_size = 2  # Размерность входных данныхoutput_size = 1  # Размерность выходных данныхhidden_size = 270  # Новое количество нейронов на слоеlearning_rate = 0.1epochs = 100def flatten(l):    return [item for sublist in l for item in sublist]def mygradient(a, b):    return torch.autograd.grad(a, b, torch.ones_like(a), create_graph=True, allow_unused=True)[0]    # return torch.autograd.grad(a, b, torch.ones_like(a))[0]def stress_calc_tc(inputs):    dPsidI1, dPsidI2, dPsidI4, dPsidI5, Stretch1, Stretch2 = inputs    one = torch.tensor(1.0, dtype=torch.float32)    two = torch.tensor(2.0, dtype=torch.float32)    # minus = two * (dPsidI1 * 1 / (Stretch ** 2) + dPsidI2 * 1 / (Stretch ** 3))    # stress = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus    first_11 = (Stretch1 - 1 / (Stretch1**2 * Stretch2**2))    second_11 = (Stretch1**2 * Stretch2 + 1 / (Stretch1 * Stretch2**2) - 1 / (Stretch1**2) - 1 / (Stretch2**2))    fourth_11 = Stretch1 * torch.cos(torch.pi / 4)**2    fifth_11 = Stretch1 * torch.cos(torch.pi / 4)**2    first_22 = (Stretch2 - 1 / (Stretch1**2 * Stretch2**2))    second_22 = (Stretch1**2 * Stretch2 + 1 / (Stretch1 ** 2 * Stretch2) - 1 / (Stretch1**2) - 1 / (Stretch2**2))    fourth_22 = Stretch1 * torch.sin(torch.pi / 4) ** 2    fifth_22 = Stretch1 * torch.sin(torch.pi / 4) ** 2    P11 = two * (first_11 * dPsidI1 + second_11 * dPsidI2 + fourth_11 * dPsidI4 + two * fifth_11 * dPsidI5)    P22 = two * (first_22 * dPsidI1 + second_22 * dPsidI2 + fourth_22 * dPsidI4 + two * fifth_22 * dPsidI5)    return P11, P22def stress_calc_ss(inputs):    dPsidI1, dPsidI2, gamma = inputs    two = torch.tensor(2.0, dtype=torch.float32)    # Shear stress    stress = two * gamma * (dPsidI1 + dPsidI2)    return stress# Self defined activation functions for expdef activation_exp(x):    return 1.0 * (torch.exp(x) - 1.0)def activation_ln(x):    return -1.0 * torch.log(1.0 - x)class MyLayer(nn.Module):    def __init__(self, my_function):        super(MyLayer, self).__init__()        self.my_function = my_function    def forward(self, x):        return self.my_function(x)class SingleInvNet6_ln(nn.Module):    def __init__(self, input_size, idi, device, l2=0.001):        """        y=xA^T+b (b=0)        :param input_size: input data size        :param idi: index of neuron        :param l2: L2 regularization coefficient        """        super().__init__()        self.terms_count = 6        self.l2 = l2        self.idi = idi        self.w11 = nn.Linear(input_size, 1, bias=False).to(device)        self.w21 = nn.Linear(input_size, 1, bias=False).to(device)        self.w31 = nn.Linear(input_size, 1, bias=False).to(device)        self.w41 = nn.Linear(input_size, 1, bias=False).to(device)        self.w51 = nn.Linear(input_size, 1, bias=False).to(device)        self.w61 = nn.Linear(input_size, 1, bias=False).to(device)        self.activation_ln = activation_ln    def forward(self, i: torch.Tensor) -> torch.Tensor:        i_ref = i - 3.0        w11_out = self.w11(i_ref)        w21_out = self.activation_ln(self.w21(i_ref))        i_sqr = torch.mul(i_ref, i_ref)        w31_out = self.w31(i_sqr)        w41_out = self.activation_ln(self.w41(i_sqr))        i_cube = torch.mul(i_sqr, i_ref)        w51_out = self.w51(i_cube)        w61_out = self.activation_ln(self.w61(i_cube))        out = torch.cat((w11_out, w21_out, w31_out, w41_out, w51_out, w61_out))        return out    def clamp_weights(self):        with torch.no_grad():            for param in self.parameters():                param.clamp_(min=0)def glorot_normal_initializer(tensor):    if tensor.dim() == 1:        nn.init.constant_(tensor, 0)    else:        nn.init.xavier_normal_(tensor)def custom_init(m):    if isinstance(m, nn.Linear):        nn.init.uniform_(m.weight, a=0.0, b=0.1)class SingleInvNet4(nn.Module):    def __init__(self, input_size, idi, device, l2=0.001):        """        y=xA^T+b (b=0)        :param input_size: input data size        :param idi: index of neuron        :param l2: L2 regularization coefficient        """        super().__init__()        self.terms_count = 4        self.l2 = l2        self.idi = idi        self.w11 = nn.Linear(input_size, 1, bias=False).to(device)        self.w21 = nn.Linear(input_size, 1, bias=False).to(device)        self.w31 = nn.Linear(input_size, 1, bias=False).to(device)        self.w41 = nn.Linear(input_size, 1, bias=False).to(device)        self.activation_exp = activation_exp    def forward(self, i: torch.Tensor) -> torch.Tensor:        i_ref = i - 3.0        w11_out = self.w11(i_ref)        w21_out = self.activation_exp(self.w21(i_ref))        i_sqr = torch.mul(i_ref, i_ref)        w31_out = self.w31(i_sqr)        w41_out = self.activation_exp(self.w41(i_sqr))        out = torch.cat((w11_out, w21_out, w31_out, w41_out))        return out    def clamp_weights(self):        with torch.no_grad():            for param in self.parameters():                param.clamp_(min=0)def glorot_normal_initializer(tensor):    if tensor.dim() == 1:        nn.init.constant_(tensor, 0)    else:        nn.init.xavier_normal_(tensor)def custom_init(m):    if isinstance(m, nn.Linear):        nn.init.uniform_(m.weight, a=0.0, b=0.1)class SingleInvNet6(nn.Module):    def __init__(self, input_size, idi, device, l2=0.001):        """        y=xA^T+b (b=0)        :param input_size: input data size        :param idi: index of neuron        :param l2: L2 regularization coefficient        """        super().__init__()        self.l2 = l2        self.idi = idi        self.w11 = nn.Linear(input_size, 1, bias=False).to(device)        self.w21 = nn.Linear(input_size, 1, bias=False).to(device)        self.w31 = nn.Linear(input_size, 1, bias=False).to(device)        self.w41 = nn.Linear(input_size, 1, bias=False).to(device)        self.w51 = nn.Linear(input_size, 1, bias=False).to(device)        self.w61 = nn.Linear(input_size, 1, bias=False).to(device)        self.activation_exp = activation_exp        self.activation_ln = activation_ln    def forward(self, i: torch.Tensor) -> torch.Tensor:        i_ref = i - 3.0        w11_out = self.w11(i_ref)        w21_out = self.activation_exp(self.w21(i_ref))        w31_out = self.activation_ln(self.w31(i_ref))        i_sqr = torch.mul(i_ref, i_ref)        w41_out = self.w41(i_sqr)        w51_out = self.activation_exp(self.w51(i_sqr))        w61_out = self.activation_ln(self.w61(i_ref))        # out = torch.cat((w11_out, w21_out, w31_out, w41_out), dim=1)        out = torch.cat((w11_out, w21_out, w31_out, w41_out, w51_out, w61_out))        return out    def clamp_weights(self):        with torch.no_grad():            for param in self.parameters():                param.clamp_(min=0)class StrainEnergyCANN_Ani(nn.Module):    def __init__(self,                 batch_size=1,                 device="cpu",                 dtype=torch.float32,                 invariants_count=2,                 invariant_model: nn.Module = SingleInvNet4,                 initial_weight: float = 1                 ):        super().__init__()        self.dtype = dtype        self.device = device        self.batch_size = batch_size        self.initial_weight = initial_weight        self.potential_constants = None        self.single_inv_net1 = invariant_model(batch_size, 0, device)        self.single_inv_net2 = invariant_model(batch_size, 4, device)        self.single_inv_net3 = invariant_model(batch_size, 8, device)        self.single_inv_net4 = invariant_model(batch_size, 12, device)        self.term_count = self.single_inv_net2.terms_count        self.invariants_count = sum([1 for module in self.modules() if "SingleInvNet" in module._get_name()])        self.wx2 = nn.Linear(self.term_count * self.invariants_count, 1, bias=False).requires_grad_(True)        self.psi_model = None        self.stress_TC = MyLayer(stress_calc_tc)        self.stress_SS = MyLayer(stress_calc_ss)        self.apply(self._init_weights)        for layer in self.modules():            classname = layer.__class__.__name__            if classname.find('Linear') != -1:                # get the number of the inputs                torch.nn.init.uniform_(layer.weight, a=0.0, b=initial_weight)                layer.weight.data = torch.clamp(layer.weight.data, min=0)    def forward(self, inputs) -> torch.Tensor:        lam, i1, i2, i3, i4, F, exp_type = inputs        i1 = i1.requires_grad_(True)        i2 = i2.requires_grad_(True)        i3 = i3.requires_grad_(True)        i4 = i4.requires_grad_(True)        i1_out = self.single_inv_net1(i1)        i2_out = self.single_inv_net2(i2)        i3_out = self.single_inv_net1(i3)        i4_out = self.single_inv_net2(i4)        inv_out = torch.cat((i1_out, i2_out, i3_out, i4_out))        self.psi_model = self.wx2(inv_out)        self.potential_constants = torch.tensor(flatten(flatten(            self.state_dict().values())), dtype=self.dtype, device=self.device).view(            self.invariants_count, self.term_count * self.invariants_count)        self.psi_model.backward(retain_graph=True, create_graph=True)        dpsi_dI1 = i1.grad        dpsi_dI2 = i2.grad        dpsi_dI3 = i3.grad        dpsi_dI4 = i4.grad        if exp_type in [1.5, 0.5, ("Compression",), ("Tensile",)]:            return self.stress_TC((dpsi_dI1, dpsi_dI2, dpsi_dI3, dpsi_dI4, lam))        elif exp_type in [1.0, ("Shear",)]:            return self.stress_SS((dpsi_dI1, dpsi_dI2, lam))        else:            raise TypeError("Wrong type of exp_type!")    def get_potential(self, p=6):        if self.potential_constants is None:            self.potential_constants = torch.tensor(flatten(flatten(                self.state_dict().values()))).view(self.invariants_count, self.term_count * self.invariants_count)        w = self.potential_constants        psi = f" {w[1, 0]                * w[0, 0]:.{p}f} * (I1 - 3) \\\\\               + {w[1, 1]:.{p}f} * (e^{{  {w[0, 1]:.{p}f} * (I1 - 3)}} - 1)\\\\\               + {w[1, 2]                * w[0, 2]:.{p}f} * (I1 - 3) ^ 2 \\\\\               + {w[1, 3]:.{p}f} * (e^{{  {w[0, 3]:.{p}f} * (I1 - 3) ^ 2}} - 1)\\\\\                \               + {w[1, 4]                * w[0, 4]:.{p}f} * (I2 - 3) \\\\\               + {w[1, 5]:.{p}f} * (e^{{  {w[0, 5]:.{p}f} * (I2 - 3))}} - 1)\\\\\               + {w[1, 6]                * w[0, 6]:.{p}f} * (I2 - 3) ^ 2 \\\\\               + {w[1, 7]:.{p}f} * (e^{{  {w[0, 7]:.{p}f} * (I2 - 3) ^ 2)}} - 1)\\\\\ "        return psi    def calc_regularization(self, l=2):        """        :param l: power        :return: sum of potential coefficients to the power of p        """        return torch.sum(self.potential_constants ** l, dtype=self.dtype)    def calc_l1(self):        return torch.sum(torch.abs(self.potential_constants), dtype=self.dtype)    def _init_weights(self, module):        if isinstance(module, nn.Linear):            torch.nn.init.uniform_(module.weight, a=0.0, b=self.initial_weight)            module.weight.data = torch.clamp(module.weight.data, min=0)    def clamp_weights(self):        for module in self.modules():            if "SingleInvNet" in module._get_name():                module.clamp_weights()        # self.single_inv_net1.clamp_weights()        # self.single_inv_net2.clamp_weights()        with torch.no_grad():            self.wx2.weight.clamp_(min=0)if __name__ == "__main__":    model = StrainEnergyCANN_Ani()    print(model.invariants_count)    model.clamp_weights()"""TODO:"""