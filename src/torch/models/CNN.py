# import torch.optim as optim# from torch.utils.data import DataLoader, TensorDataset# # import tqdm# from dataload import ExcelDataset# from torchmetrics.regression import MeanSquaredError# from torchviz import make_dotimport torchimport torch.nn as nnimport osimport mathfrom math import expimport numpy as npimport sympy as sp# Гиперпараметры# input_size = 2  # Размерность входных данныхoutput_size = 1  # Размерность выходных данныхhidden_size = 270  # Новое количество нейронов на слоеlearning_rate = 0.001epochs = 100def flatten(l):    return [item for sublist in l for item in sublist]def myGradient(a, b):    return torch.autograd.grad(a, b, torch.ones_like(a), create_graph=True)[0]def Stress_calc_TC(inputs):    dPsidI1, dPsidI2, Stretch = inputs    one = torch.tensor(1.0, dtype=torch.float32)    two = torch.tensor(2.0, dtype=torch.float32)    minus  = two * (dPsidI1 * 1 / (Stretch**2) + dPsidI2 * 1 / (Stretch**3))    stress = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus    return stressdef Stress_calc_SS(inputs):    dPsidI1, dPsidI2, gamma = inputs    two = torch.tensor(2.0, dtype=torch.float32)    # Shear stress    stress = two * gamma * (dPsidI1 + dPsidI2)    return stressdef Stress_calc_inv(inputs):    """    :param inputs:    :return:    """    dPsidI1, dPsidI2, C = inputs    I2 = 0.5 * (torch.trace(C)**2 - torch.trace(C @ C))    dPsidI3 = torch.tensor(1.0, dtype=torch.float32)    two = torch.tensor(2.0, dtype=torch.float32)    dI1dC = torch.eye(3)    dI2dC = I2 * torch.inverse(C)    dI3dC = torch.tensor(0, dtype=torch.float32)    # Shear stress    stress = two * (dPsidI1 * dI1dC + dPsidI2 * dI2dC + dPsidI3 * dI3dC)    return stress# Self defined activation functions for expdef activation_Exp(x):    return 1.0*(torch.exp(x) -1.0)def activation_ln(x):    return -1.0*torch.log(1.0 - x)class SingleInvNet1(nn.Module):    def __init__(self, input_size, idi, device, l2=0.001):        """        y=xA^T+b (b=0)        :param input_size: input data size        :param idi: index of neuron        :param l2: L2 regularization coefficient        """        super().__init__()        self.l2 = l2        self.idi = idi        self.w11 = nn.Linear(input_size, 1, bias=False).to(device)    def forward(self, i: torch.Tensor) -> torch.Tensor:        i_ref = i - 3.0        w11_out = self.w11(i_ref)        out = torch.cat(w11_out)        return outclass SingleInvNet2(nn.Module):    def __init__(self, input_size, idi, device, l2=0.001):        """        y=xA^T+b (b=0)        :param input_size: input data size        :param idi: index of neuron        :param l2: L2 regularization coefficient        """        super().__init__()        self.l2 = l2        self.idi = idi        self.w11_linear = nn.Linear(input_size, 1, bias=False).to(device)        self.w21_square = nn.Linear(input_size, 1, bias=False).to(device)    def forward(self, i: torch.Tensor) -> torch.Tensor:        i_ref = i - 3.0        w11_out = self.w11_linear(i_ref)        i_sqr = torch.mul(i_ref, i_ref)        w21_out = self.w21_square(i_sqr)        out = torch.cat((w11_out, w21_out))        return outclass SingleInvNet4(nn.Module):    def __init__(self, input_size, idi, device, l2=0.001):        """        y=xA^T+b (b=0)        :param input_size: input data size        :param idi: index of neuron        :param l2: L2 regularization coefficient        """        super().__init__()        self.l2 = l2        self.idi = idi        self.w11 = nn.Linear(input_size, 1, bias=False).to(device)        self.w21 = nn.Linear(input_size, 1, bias=False).to(device)        self.w31 = nn.Linear(input_size, 1, bias=False).to(device)        self.w41 = nn.Linear(input_size, 1, bias=False).to(device)        self.activation_Exp = activation_Exp    def forward(self, i: torch.Tensor) -> torch.Tensor:        i_ref = i - 3.0        w11_out = self.w11(i_ref)        # w11_out = self.dropout1(w11_out)        w21_out = self.activation_Exp(self.w21(i_ref))        # w21_out = self.dropout2(w21_out)        i_sqr = torch.mul(i_ref, i_ref)        w31_out = self.w31(i_sqr)        # w31_out = self.dropout3(w31_out)        w41_out = self.activation_Exp(self.w41(i_sqr))        # w41_out = self.dropout4(w41_out)        # out = torch.cat((w11_out, w21_out, w31_out, w41_out), dim=1)        out = torch.cat((w11_out, w21_out, w31_out, w41_out))        return outclass SingleInvNet6(nn.Module):    def __init__(self, input_size, idi, device, l2=0.001):        """        y=xA^T+b (b=0)        :param input_size: input data size        :param idi: index of neuron        :param l2: L2 regularization coefficient        """        super().__init__()        self.l2 = l2        self.idi = idi        self.w11 = nn.Linear(input_size, 1, bias=False).to(device)        self.w21 = nn.Linear(input_size, 1, bias=False).to(device)        self.w31 = nn.Linear(input_size, 1, bias=False).to(device)        self.w41 = nn.Linear(input_size, 1, bias=False).to(device)        self.w51 = nn.Linear(input_size, 1, bias=False).to(device)        self.w61 = nn.Linear(input_size, 1, bias=False).to(device)        self.activation_Exp = activation_Exp        self.activation_ln = activation_ln    def forward(self, i: torch.Tensor) -> torch.Tensor:        i_ref = i - 3.0        w11_out = self.w11(i_ref)        w21_out = self.activation_Exp(self.w21(i_ref))        w31_out = self.activation_ln(self.w31(i_ref))        i_sqr = torch.mul(i_ref, i_ref)        w41_out = self.w31(i_sqr)        w51_out = self.activation_Exp(self.w41(i_sqr))        w61_out = self.activation_ln(self.w61(i_ref))        # out = torch.cat((w11_out, w21_out, w31_out, w41_out), dim=1)        out = torch.cat((w11_out, w21_out, w31_out, w41_out, w51_out, w61_out))        return outclass StrainEnergyCANN(nn.Module):    def __init__(self, batch_size, device):        super().__init__()        self.potential_constants = np.zeros(16)        self.device = device        self.batch_size = batch_size        self.single_inv_net1 = SingleInvNet4(batch_size, 0, device)        self.single_inv_net2 = SingleInvNet4(batch_size, 4, device)        self.wx2 = nn.Linear(8, 1, bias=False)        self.P = Stress_calc_inv    # def forward(self, i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:    def forward(self, invariants: torch.Tensor) -> torch.Tensor:        if self.batch_size == 1:            invariants = invariants[0].clone().detach().to(self.device)            i1 = invariants[0]            i2 = invariants[1]            i1_out = self.single_inv_net1(i1.unsqueeze(0))            i2_out = self.single_inv_net2(i2.unsqueeze(0))        else:            i1 = invariants[:, 0]            i2 = invariants[:, 1]            i1_out = self.single_inv_net1(i1.unsqueeze(1).unsqueeze(0))            i2_out = self.single_inv_net2(i2.unsqueeze(1).unsqueeze(0))        psi_model = torch.cat((i1_out, i2_out))        # out = torch.cat((i1_out, i2_out), dim=1)        # out = out.view(-1, 8)  # Изменение формы перед применением линейного слоя        psi_model = self.wx2(psi_model)        stress_model = self.P()        return psi_model    def get_potential(self):        """        :return: [weights of model]        """        params = []        for id, weights in enumerate(self.parameters()):            print(f"id = {id}, weight = {weights}")            if id == 12:                weights = weights.tolist()                for weight_last_layer in weights[0]:                    # print(weight_last_layer)                    params.append(weight_last_layer)                break            weight = weights.detach().to('cpu').numpy().copy()            params.append(weight[0, 0].item())            # params.append(param.data())        # psi = w2[0] * w1[0] * (I1 - 3) + w2[1] * (sp.exp(w1[1] * (I1 - 3)) - 1) + w2[2] * w1[2] * (I1 - 3) ** 2 + w2[        #     3] * (sp.exp(w1[3] * (I1 - 3) ** 2) - 1) + w2[4] * w1[4] * (I2 - 3) + w2[5] * (        #                   sp.exp(w1[5] * (I2 - 3)) - 1) + w2[6] * w1[6] * (I2 - 3) ** 2 + w2[7] * (        #                   sp.exp(w1[7] * (I2 - 3) ** 2) - 1)        return paramsclass StrainEnergyCANN_C(nn.Module):    def __init__(self,                 batch_size=1,                 device="cpu",                 stress_calc=Stress_calc_inv,                 dtype=torch.float32,                 term_count=6,                 invariants_count=2                 ):        super().__init__()        self.dtype = dtype        self.device = device        self.batch_size = batch_size        self.potential_constants = None        self.single_inv_net1 = SingleInvNet6(batch_size, 0, device)        self.single_inv_net2 = SingleInvNet6(batch_size, 6, device)        self.wx2 = nn.Linear(term_count * invariants_count, 1, bias=False).requires_grad_(True)        self.psi_model = torch.zeros((3, 3))    # def forward(self, i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:    def forward(self, F, invariants) -> torch.Tensor:        # F, invariants = inputs        if type(invariants) == torch.Tensor:            invariants = invariants.squeeze()        i1, i2 = invariants        # print(C)        i1 = i1.requires_grad_(True)        i2 = i2.requires_grad_(True)        i1_out = self.single_inv_net1(i1.unsqueeze(0))        i2_out = self.single_inv_net2(i2.unsqueeze(0))        inv_out = torch.cat((i1_out, i2_out))        # out = torch.cat((i1_out, i2_out), dim=1)        # out = out.view(-1, 8)  # Изменение формы перед применением линейного слоя        self.psi_model = self.wx2(inv_out)        self.potential_constants = torch.tensor(flatten(flatten(self.state_dict().values())),                                                dtype=self.dtype, device=self.device).view(2, 12)        stress_model = self.calculate_stress((i1, i2, F))        return stress_model    def get_potential(self, p=2):        if self.potential_constants is None:            self.potential_constants = torch.tensor(flatten(flatten(self.state_dict().values()))).view(2, 12)        w = self.potential_constants        # I1, I2 = sp.symbols('I1 I2')        # # Ваше выражение        # psi = f" {w[1, 0]      *             w[0, 0]:.3f} * (I1 - 3) \\\\\        #        + {w[1, 1]:.3f} * e^(    {w[0, 1]:.3f} * (I1 - 3)) \\\\\        #        - {w[1, 2]:.3f} * ln(1 -  {w[0, 2]:.3f} * (I1 - 3)) \\\\\        #        + {w[1, 3]      *             w[0, 3]:.3f} * (I1 - 3) ** 2 \\\\\        #        + {w[1, 4]:.3f} * e^(    {w[0, 4]:.3f} * (I1 - 3) ** 2) \\\\\        #        - {w[1, 5]:.3f} * ln(1 -  {w[0, 5]:.3f} * (I1 - 3) ** 2) \\\\\        #         \        #        + {w[1, 6]      *         w[0, 6]:.3f} * (I2 - 3) \\\\\        #        + {w[1, 8]:.3f} * e^(   {w[0, 7]:.3f} * (I2 - 3)) \\\\\        #        - {w[1, 10]:.3f}* ln(1 - {w[0, 8]:.3f} *  (I2 - 3)) \\\\\        #        + {w[1, 7]:.3f} *        {w[0, 9]:.3f} * (I2 - 3) ** 2 \\\\\        #        + {w[1, 9]:.3f} * e^(   {w[0, 10]:.3f} *(I2 - 3) ** 2) \\\\\        #        - {w[1, 11]:.3f}* ln(1 - {w[0, 11]:.3f} * (I2 - 3) ** 2)\\\\"        psi = f" {w[1, 0]      *             w[0, 0]:.{p}f} * (I1 - 3) \\\\\               + {w[1, 1]:.{p}f} * (e^{{    {w[0, 1]:.{p}f} * (I1 - 3)}} - 1)\\\\\               - {w[1, 2]:.{p}f} * ln(1 -  {w[0, 2]:.{p}f} * (I1 - 3)) \\\\\               + {w[1, 3]      *             w[0, 3]:.{p}f} * (I1 - 3) ^ 2 \\\\\               + {w[1, 4]:.{p}f} * (e^{{    {w[0, 4]:.{p}f} * (I1 - 3) ^ 2}} - 1)\\\\\               - {w[1, 5]:.{p}f} * ln(1 -  {w[0, 5]:.{p}f} * (I1 - 3) ^ 2) \\\\\                \               + {w[1, 6]      *         w[0, 6]:.{p}f} * (I2 - 3) \\\\\               + {w[1, 8]:.{p}f} * (e^{{   {w[0, 7]:.{p}f} * (I2 - 3))}} - 1)\\\\\               - {w[1, 10]:.{p}f}* ln(1 - {w[0, 8]:.{p}f} *  (I2 - 3)) \\\\\               + {w[1, 7]:.{p}f} *        {w[0, 9]:.{p}f} * (I2 - 3) ^ 2 \\\\\               + {w[1, 9]:.{p}f} * (e^{{   {w[0, 10]:.{p}f} *(I2 - 3) ^ 2)}} - 1)\\\\\               - {w[1, 11]:.{p}f}* ln(1 - {w[0, 11]:.{p}f} * (I2 - 3) ^ 2)\\\\"        # psi_tex = sp.latex(psi)        # print(psi_tex)        return psi    def calc_regularization(self, l=2):        """        :param l: power        :return: sum of potential coefficients to the power of p        """        return torch.sum(self.potential_constants**l, dtype=self.dtype)    def calc_l1(self):        return torch.sum(torch.abs(self.potential_constants), dtype=self.dtype)    def calculate_stress(self, inputs):        w = self.potential_constants.reshape(2, 12)        I1, I2, F = inputs        dI_dF_1 = 2 * F        dI_dF_2 = 2 * (I1 * F - torch.matmul(torch.matmul(F, F.t()), F))        P = (dI_dF_1 * (      w[1, 0] * w[0, 0] + w[1, 1] * w[0, 1] *   torch.exp(w[0, 1] * (I1 - 3))                                                + w[1, 2] * w[0, 2] /       (1 -  w[0, 2] * (I1 - 3)))            + 2 * (I1 - 3) * (w[1, 3] * w[0, 3] + w[1, 4] * w[0, 4] *   torch.exp(w[0, 4] * (I1 - 3)**2))                                                + w[1, 5] * w[0, 5] /   (1 -      w[0, 5] * (I1 - 3)**2)            + dI_dF_2 * (     w[1, 6] * w[0, 6] + w[1, 7] * w[0, 7] *   torch.exp(w[0, 7] * (I2 - 3))                                                + w[1, 8] * w[0, 8] /   (1 -      w[0, 8] * (I2 - 3))            + 2 * (I2 - 3) * (w[1, 9] * w[0, 9] + w[1, 10] * w[0, 10] * torch.exp(w[0, 10] *(I2 - 3)**2))                                                + w[1, 11] * w[0, 11] / (1 -      w[0, 11] *(I2 - 3)**2)))        return Pif __name__ == "__main__":    I1, I2 = sp.symbols('I1 I2')    w1 = sp.symbols('w11:19')  # w1, w2, ..., w8    w2 = sp.symbols('w21:29')  # w21, w22, ..., w28    torch.manual_seed(42)    trained_model = StrainEnergyCANN_C(batch_size=1, device="cpu")    trained_model.load_state_dict(torch.load(r'C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\src\torch\pretrained_models\CNN_brain_6term_C_cuda\20240418_182131_10.pth'))    print(trained_model.get_potential())    print(trained_model.calc_regularization())"""TODO:"""