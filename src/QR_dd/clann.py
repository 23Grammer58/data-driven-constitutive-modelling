import torch
import torch.nn as nn
import torch.nn.functional as F


class CLANN(nn.Module):
    """
    Convex Laplace Approximation Neural Network - выпуклая нейронная сеть 
    для аппроксимации функции отклика в задаче деформации Лапласа.
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3, C11=1.0, C12=0.5, C22=1.0, direct_stress=False):
        super(CLANN, self).__init__()
        
        # Входной слой
        self.fc_input_weight = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.fc_input_bias = nn.Parameter(torch.randn(hidden_dim))
        
        # Выходной слой
        self.fc_output_weight = nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.fc_output_bias = nn.Parameter(torch.randn(output_dim))
        
        # Функция softplus для обеспечения неотрицательности весов
        self.softplus = nn.Softplus()
        
        # Параметры для вычисления напряжений
        self.C11 = C11
        self.C12 = C12
        self.C22 = C22
        
        # Флаг, указывающий, обучается ли модель напрямую предсказывать напряжения
        self.direct_stress = direct_stress
    
    def forward(self, x, return_stress=False, return_piola=False, return_r=False):
        """
        Прямой проход через нейронную сеть.
        
        Args:
            x (torch.Tensor): Входные данные (xi)
            return_stress (bool): Возвращать ли напряжения Коши S
            return_piola (bool): Возвращать ли напряжения Пиолы-Кирхгофа P
            return_r (bool): Возвращать ли функции отклика r (если модель обучается напрямую по напряжениям)
            
        Returns:
            torch.Tensor или tuple: Предсказанные значения r/S или кортеж с различными комбинациями (r, S, P)
        """
        # Входной слой с ReLU активацией
        weight = self.softplus(self.fc_input_weight)
        hidden = F.linear(x, weight, self.fc_input_bias)
        hidden = F.relu(hidden)
        
        # Выходной слой
        weight = self.softplus(self.fc_output_weight)
        output = F.linear(hidden, weight, self.fc_output_bias)
        
        # Если модель обучается напрямую предсказывать напряжения S
        if self.direct_stress:
            S = output  # Выход модели - это напряжения S
            
            # Вычисляем r из S (обратное преобразование)
            r = self.compute_r_from_stress(S)
            
            if return_r and not return_piola:
                return S, r
            
            if not return_r and not return_piola:
                return S
        else:
            r = output  # Выход модели - это функции отклика r
            
            if not return_stress and not return_piola:
                return r
            
            # Вычисляем напряжения S на основе r
            S = self.compute_stress(r)
            
            if return_stress and not return_piola:
                return r, S
        
        # Вычисляем тензор напряжений Пиолы-Кирхгофа P = S * F^(-T)
        P = self.compute_piola_stress(S, x)
        
        if self.direct_stress and return_r:
            return S, r, P
        else:
            return r, S, P
    
    def compute_stress(self, r):
        """
        Вычисляет напряжения S на основе предсказанных значений r.
        
        S11 = r1
        S22 = r2
        S12 = 1/C11 * (sqrt(C22*C11) - C12) * r3
        
        Args:
            r (torch.Tensor): Тензор предсказанных значений r размера [batch_size, 3]
            
        Returns:
            torch.Tensor: Тензор напряжений S размера [batch_size, 3]
        """
        batch_size = r.shape[0]
        S = torch.zeros(batch_size, 3, device=r.device)
        
        # S11 = r1
        S[:, 0] = r[:, 0]
        
        # S22 = r2
        S[:, 1] = r[:, 1]
        
        # S12 = 1/C11 * (sqrt(C22*C11) - C12) * r3
        S[:, 2] = (1.0 / self.C11) * (torch.sqrt(self.C22 * self.C11) - self.C12) * r[:, 2]
        
        return S
    
    def compute_r_from_stress(self, S):
        """
        Вычисляет функции отклика r на основе напряжений S (обратное преобразование).
        
        r1 = S11
        r2 = S22
        r3 = S12 * C11 / (sqrt(C22*C11) - C12)
        
        Args:
            S (torch.Tensor): Тензор напряжений S размера [batch_size, 3]
            
        Returns:
            torch.Tensor: Тензор функций отклика r размера [batch_size, 3]
        """
        batch_size = S.shape[0]
        r = torch.zeros(batch_size, 3, device=S.device)
        
        # r1 = S11
        r[:, 0] = S[:, 0]
        
        # r2 = S22
        r[:, 1] = S[:, 1]
        
        # r3 = S12 * C11 / (sqrt(C22*C11) - C12)
        r[:, 2] = S[:, 2] * self.C11 / (torch.sqrt(self.C22 * self.C11) - self.C12)
        
        return r
    
    def compute_piola_stress(self, S, xi):
        """
        Вычисляет тензор напряжений Пиолы-Кирхгофа P = S * F^(-T).
        
        Args:
            S (torch.Tensor): Тензор напряжений Коши размера [batch_size, 3]
            xi (torch.Tensor): Входные данные (xi) размера [batch_size, 3]
            
        Returns:
            torch.Tensor: Тензор напряжений Пиолы-Кирхгофа P размера [batch_size, 3]
        """
        batch_size = S.shape[0]
        P = torch.zeros(batch_size, 3, device=S.device)
        
        # Вычисляем тензор C = F^T * F из xi
        # xi1 = C11, xi2 = C22, xi3 = C12 = C21
        C11 = xi[:, 0]
        C22 = xi[:, 1]
        C12 = xi[:, 2]
        
        # Вычисляем F из C
        # Для 2D случая, F можно вычислить через разложение Холецкого C = L * L^T
        # где L - нижнетреугольная матрица
        # F11 = sqrt(C11)
        # F12 = C12 / F11
        # F21 = 0
        # F22 = sqrt(C22 - F12^2)
        
        F11 = torch.sqrt(C11)
        F12 = C12 / F11
        F21 = torch.zeros_like(F11)
        F22 = torch.sqrt(C22 - F12**2)
        
        # Вычисляем F^(-T)
        det_F = F11 * F22 - F12 * F21
        F_inv_T = torch.zeros(batch_size, 2, 2, device=S.device)
        F_inv_T[:, 0, 0] = F22 / det_F
        F_inv_T[:, 0, 1] = -F12 / det_F
        F_inv_T[:, 1, 0] = -F21 / det_F
        F_inv_T[:, 1, 1] = F11 / det_F
        
        # Преобразуем S из векторной формы [S11, S22, S12] в матричную форму
        S_mat = torch.zeros(batch_size, 2, 2, device=S.device)
        S_mat[:, 0, 0] = S[:, 0]  # S11
        S_mat[:, 1, 1] = S[:, 1]  # S22
        S_mat[:, 0, 1] = S[:, 2]  # S12
        S_mat[:, 1, 0] = S[:, 2]  # S21 = S12
        
        # Вычисляем P = S * F^(-T)
        P_mat = torch.bmm(S_mat, F_inv_T)
        
        # Преобразуем P из матричной формы в векторную форму [P11, P22, P12]
        P[:, 0] = P_mat[:, 0, 0]  # P11
        P[:, 1] = P_mat[:, 1, 1]  # P22
        P[:, 2] = P_mat[:, 0, 1]  # P12
        
        return P
    
    def predict_stress(self, x):
        """
        Предсказывает напряжения S для входных данных x.
        
        Args:
            x (torch.Tensor): Входные данные
            
        Returns:
            torch.Tensor: Предсказанные напряжения S
        """
        self.eval()
        with torch.no_grad():
            if self.direct_stress:
                S = self.forward(x)
            else:
                _, S = self.forward(x, return_stress=True)
        return S
    
    def predict_r(self, x):
        """
        Предсказывает функции отклика r для входных данных x.
        
        Args:
            x (torch.Tensor): Входные данные
            
        Returns:
            torch.Tensor: Предсказанные функции отклика r
        """
        self.eval()
        with torch.no_grad():
            if self.direct_stress:
                _, r = self.forward(x, return_r=True)
            else:
                r = self.forward(x)
        return r
    
    def predict_piola_stress(self, x):
        """
        Предсказывает напряжения Пиолы-Кирхгофа P для входных данных x.
        
        Args:
            x (torch.Tensor): Входные данные
            
        Returns:
            torch.Tensor: Предсказанные напряжения Пиолы-Кирхгофа P
        """
        self.eval()
        with torch.no_grad():
            if self.direct_stress:
                _, _, P = self.forward(x, return_r=True, return_piola=True)
            else:
                _, _, P = self.forward(x, return_stress=True, return_piola=True)
        return P 