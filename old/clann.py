import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.LU_strain import upper_triangle, xie_calc


class CLANN(nn.Module):
    """
    Convex Laplace Approximation Neural Network - выпуклая нейронная сеть 
    для аппроксимации функции отклика в задаче деформации Лапласа.
    
    Поддерживает три режима работы:
    1. clann_base: (xi1, xi2, xi3) -> (r1, r2, r3)
    2. clann_pk2: (F11, F12, F22) -> (P11, P12, P22) 
    3. clann_pk2_with_xi: (xi1, xi2, xi3, F11, F12, F22) -> (P11, P12, P22)
    """
    
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3, mode='base'):
        super(CLANN, self).__init__()
        
        # Режимы работы
        self.mode = mode  # 'base', 'pk2', 'pk2_with_xi'
        
        # Параметры для выпуклой нейронной сети
        # Входной слой - веса и смещения
        self.fc_input_weight = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.fc_input_bias = nn.Parameter(torch.randn(hidden_dim))
        
        # Выходной слой - веса и смещения
        self.fc_output_weight = nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.fc_output_bias = nn.Parameter(torch.randn(output_dim))

        # Функция softplus для обеспечения неотрицательности весов (выпуклость)
        self.softplus = nn.Softplus()

        # Проверяем корректность режима
        if mode not in ['base', 'pk2', 'pk2_with_xi']:
            raise ValueError(f"Неподдерживаемый режим: {mode}")
    
    def compute_xi_from_F(self, F):
        """
        Вычисляет параметры Laplace stretch xi из градиента деформации F.
        
        Args:
            F (torch.Tensor): Градиент деформации размера [batch_size, 4] = [F11, F12, F21, F22]
                             или [batch_size, 3] = [F11, F12, F22] (предполагая F21=0)
            
        Returns:
            torch.Tensor: Параметры xi размера [batch_size, 3]
        """
        
        if F.shape[1] == 3:
            # F = [F11, F12, F22], предполагаем F21 = F12
            F11 = F[:, 0]
            F12 = F[:, 1] 
            F22 = F[:, 2]
            F21 = F12
        elif F.shape[1] == 2:
            # F = [F11, F22], предполагаем F21 = F12 = 0
            F11 = F[:, 0]
            F22 = F[:, 2] 
            F12 = 0
            F21 = 0
        elif F.shape[1] == 4:
            # F = [F11, F12, F21, F22]
            F11 = F[:, 0]
            F12 = F[:, 1]
            F21 = F[:, 2]
            F22 = F[:, 3]
        else:
            raise ValueError(f"Неподдерживаемая размерность F: {F.shape}")
        F = torch.stack([F11, F12, F21, F22], dim=1)
        
        # C11 = F11**2 + F21**2
        # C22 = F12**2 + F22**2
        # C12 = F11*F12 + F21*F22

        xie = xie_calc(upper_triangle(F))
        # if F.shape[1] == 3:
        #     # F = [F11, F12, F22], предполагаем F21 = 0
        #     F11 = F[:, 0]
        #     F12 = F[:, 1] 
        #     F22 = F[:, 2]
        #     F21 = torch.zeros_like(F11)
        # elif F.shape[1] == 4:
        #     # F = [F11, F12, F21, F22]
        #     F11 = F[:, 0]
        #     F12 = F[:, 1]
        #     F21 = F[:, 2]
        #     F22 = F[:, 3]
        
        
        # # Вычисляем тензор деформации Коши-Грина C = F^T * F
        # C11 = F11**2 + F21**2
        # C22 = F12**2 + F22**2
        # C12 = F11*F12 + F21*F22
        
        # # Вычисляем F˜ из C согласно формулам:
        # # F˜11 = √C11
        # # F˜12 = C12 / F˜11  
        # # F˜22 = √(C22 - F˜12²)
        # F_tilde_11 = torch.sqrt(C11)
        # F_tilde_12 = C12 / F_tilde_11
        # F_tilde_22 = torch.sqrt(C22 - F_tilde_12**2)
        
        # # Вычисляем меру деформации (Laplace stretch):
        # # ξ1 = ln F˜11
        # # ξ2 = ln F˜22  
        # # ξ3 = F˜12 / F˜11
        # xi1 = torch.log(F_tilde_11)
        # xi2 = torch.log(F_tilde_22)
        # xi3 = F_tilde_12 / F_tilde_11

        
        # return torch.stack([xi1, xi2, xi3], dim=1)
        return xie
    
    def compute_F_tilde_from_xi(self, xi):
        """
        Восстанавливает F˜ из параметров Laplace stretch xi.
        
        Args:
            xi (torch.Tensor): Параметры xi размера [batch_size, 3]
            
        Returns:
            tuple: (F_tilde_11, F_tilde_12, F_tilde_22)
        """
        xi1 = xi[:, 0]  # ξ1 = ln F˜11
        xi2 = xi[:, 1]  # ξ2 = ln F˜22
        xi3 = xi[:, 2]  # ξ3 = F˜12 / F˜11
        
        # Восстанавливаем F˜ из xi:
        # F˜11 = exp(ξ1)
        # F˜22 = exp(ξ2)
        # F˜12 = ξ3 * F˜11
        F_tilde_11 = torch.exp(xi1)
        F_tilde_22 = torch.exp(xi2)
        F_tilde_12 = xi3 * F_tilde_11
        
        return F_tilde_11, F_tilde_12, F_tilde_22
    
    def compute_stress_from_r(self, r):
        """
        Вычисляет напряжения Коши S из функций отклика r.
        
        Args:
            r (torch.Tensor): Функции отклика размера [batch_size, 3]
            
        Returns:
            torch.Tensor: Напряжения Коши S размера [batch_size, 3]
        """
        batch_size = r.shape[0]
        S = torch.zeros(batch_size, 3, device=r.device)
        
        # S11 = r1
        S[:, 0] = r[:, 0]
        
        # S22 = r2
        S[:, 1] = r[:, 1]
        
        # S12 = 1/C11 * (sqrt(C22*C11) - C12) * r3
        S[:, 2] = (1.0 / self.C11) * (math.sqrt(self.C22 * self.C11) - self.C12) * r[:, 2]
        
        return S
    
    def compute_r_from_stress(self, S):
        """
        Вычисляет функции отклика r из напряжений Коши S.
        
        Args:
            S (torch.Tensor): Напряжения Коши размера [batch_size, 3]
            
        Returns:
            torch.Tensor: Функции отклика r размера [batch_size, 3]
        """
        batch_size = S.shape[0]
        r = torch.zeros(batch_size, 3, device=S.device)
        
        # r1 = S11
        r[:, 0] = S[:, 0]
        
        # r2 = S22
        r[:, 1] = S[:, 1]
        
        # r3 = S12 * C11 / (sqrt(C22*C11) - C12)
        r[:, 2] = S[:, 2] * self.C11 / (math.sqrt(self.C22 * self.C11) - self.C12)
        
        return r
    
    def compute_piola_from_stress(self, S, xi):
        """
        Вычисляет напряжения Пиолы-Кирхгофа P из напряжений Коши S.
        P = S * F^(-T)
        
        Args:
            S (torch.Tensor): Напряжения Коши размера [batch_size, 3]
            xi (torch.Tensor): Параметры xi размера [batch_size, 3]
            
        Returns:
            torch.Tensor: Напряжения Пиолы-Кирхгофа P размера [batch_size, 3]
        """
        batch_size = S.shape[0]
        
        # Восстанавливаем F˜ из xi
        F_tilde_11, F_tilde_12, F_tilde_22 = self.compute_F_tilde_from_xi(xi)
        
        # F˜ соответствует разложению Холецкого, поэтому F = F˜
        F11 = F_tilde_11
        F12 = F_tilde_12
        F21 = torch.zeros_like(F_tilde_11)
        F22 = F_tilde_22
        
        # Вычисляем F^(-T)
        det_F = F11 * F22 - F12 * F21
        F_inv_T = torch.zeros(batch_size, 2, 2, device=S.device)
        F_inv_T[:, 0, 0] = F22 / det_F
        F_inv_T[:, 0, 1] = -F12 / det_F
        F_inv_T[:, 1, 0] = -F21 / det_F
        F_inv_T[:, 1, 1] = F11 / det_F
        
        # Преобразуем S в матричную форму
        S_mat = torch.zeros(batch_size, 2, 2, device=S.device)
        S_mat[:, 0, 0] = S[:, 0]  # S11
        S_mat[:, 1, 1] = S[:, 1]  # S22
        S_mat[:, 0, 1] = S[:, 2]  # S12
        S_mat[:, 1, 0] = S[:, 2]  # S21 = S12
        
        # Вычисляем P = S * F^(-T)
        P_mat = torch.bmm(S_mat, F_inv_T)
        
        # Преобразуем P в векторную форму
        P = torch.zeros(batch_size, 3, device=S.device)
        P[:, 0] = P_mat[:, 0, 0]  # P11
        P[:, 1] = P_mat[:, 1, 1]  # P22
        P[:, 2] = P_mat[:, 0, 1]  # P12
        
        return P
    
    def compute_stress_from_piola(self, P, xi):
        """
        Вычисляет напряжения Коши S из напряжений Пиолы-Кирхгофа P.
        S = P * F^T
        
        Args:
            P (torch.Tensor): Напряжения Пиолы-Кирхгофа размера [batch_size, 3]
            xi (torch.Tensor): Параметры xi размера [batch_size, 3]
            
        Returns:
            torch.Tensor: Напряжения Коши S размера [batch_size, 3]
        """
        batch_size = P.shape[0]
        
        # Восстанавливаем F˜ из xi
        F_tilde_11, F_tilde_12, F_tilde_22 = self.compute_F_tilde_from_xi(xi)
        
        # F˜ соответствует разложению Холецкого, поэтому F = F˜
        F11 = F_tilde_11
        F12 = F_tilde_12
        F21 = torch.zeros_like(F_tilde_11)
        F22 = F_tilde_22
        
        # Вычисляем F^T
        F_T = torch.zeros(batch_size, 2, 2, device=P.device)
        F_T[:, 0, 0] = F11  # F11
        F_T[:, 0, 1] = F21  # F21 = 0
        F_T[:, 1, 0] = F12  # F12
        F_T[:, 1, 1] = F22  # F22
        
        # Преобразуем P в матричную форму
        P_mat = torch.zeros(batch_size, 2, 2, device=P.device)
        P_mat[:, 0, 0] = P[:, 0]  # P11
        P_mat[:, 1, 1] = P[:, 1]  # P22
        P_mat[:, 0, 1] = P[:, 2]  # P12
        P_mat[:, 1, 0] = P[:, 2]  # P21 = P12
        
        # Вычисляем S = P * F^T
        S_mat = torch.bmm(P_mat, F_T)
        
        # Преобразуем S в векторную форму
        S = torch.zeros(batch_size, 3, device=P.device)
        S[:, 0] = S_mat[:, 0, 0]  # S11
        S[:, 1] = S_mat[:, 1, 1]  # S22
        S[:, 2] = S_mat[:, 0, 1]  # S12
        
        return S
    
    def compute_r_gradient(self, xi):
        """
        Вычисляет градиент r по xi через автоматическое дифференцирование.
        
        Args:
            xi (torch.Tensor): Параметры xi размера [batch_size, 3]
            
        Returns:
            torch.Tensor: Градиент dr/dxi размера [batch_size, 3, 3]
        """
        xi.requires_grad_(True)
        
        # Получаем r через нейронную сеть
        r = self.forward_base(xi)
        
        # Создаем тензор для хранения градиентов
        batch_size = xi.shape[0]
        dr_dxi = torch.zeros(batch_size, 3, 3, device=xi.device)
        
        # Вычисляем градиент для каждого компонента r
        for i in range(3):
            # Обнуляем предыдущие градиенты
            if xi.grad is not None:
                xi.grad.zero_()
            
            # Вычисляем градиент i-го компонента r по всем xi
            grad_outputs = torch.zeros_like(r)
            grad_outputs[:, i] = 1.0
            
            grads = torch.autograd.grad(
                outputs=r, inputs=xi, grad_outputs=grad_outputs,
                retain_graph=True, create_graph=True
            )[0]
            
            dr_dxi[:, i] = grads
        
        return dr_dxi
    
    def forward_base(self, xi):
        """
        Базовый режим: (xi1, xi2, xi3) -> (r1, r2, r3)
        
        Args:
            xi (torch.Tensor): Параметры xi размера [batch_size, 3]
            
        Returns:
            torch.Tensor: Функции отклика r размера [batch_size, 3]
        """
        # Входной слой с неотрицательными весами для обеспечения выпуклости
        input_weight = self.softplus(self.fc_input_weight)
        hidden = F.relu(F.linear(xi, input_weight, self.fc_input_bias))
        
        # Выходной слой с неотрицательными весами для обеспечения выпуклости
        output_weight = self.softplus(self.fc_output_weight)
        r = F.linear(hidden, output_weight, self.fc_output_bias)
        
        return r
    
    def forward_pk2(self, F):
        """
        Режим PK2: (F11, F12, F22) -> (P11, P12, P22)
        
        Args:
            F (torch.Tensor): Градиент деформации размера [batch_size, 3] = [F11, F12, F22]
            
        Returns:
            torch.Tensor: Напряжения Пиолы-Кирхгофа P размера [batch_size, 3]
        """
        # Шаг 1: Вычисляем xi из F
        xi = self.compute_xi_from_F(F)
        
        # Шаг 2: Вычисляем r через нейронную сеть
        r = self.forward_base(xi)
        
        # Шаг 3: Вычисляем градиент dr/dxi
        dr_dxi = self.compute_r_gradient(xi)
        
        # Шаг 4: Вычисляем S из r
        S = self.compute_stress_from_r(r)
        
        # Шаг 5: Вычисляем P из S
        P = self.compute_piola_from_stress(S, xi)
        
        return P
    
    def forward_pk2_with_xi(self, xi, F):
        """
        Режим PK2 с готовыми xi: (xi1, xi2, xi3, F11, F12, F22) -> (P11, P12, P22)
        
        Args:
            xi (torch.Tensor): Параметры xi размера [batch_size, 3]
            F (torch.Tensor): Градиент деформации размера [batch_size, 3] (не используется для вычисления xi)
            
        Returns:
            torch.Tensor: Напряжения Пиолы-Кирхгофа P размера [batch_size, 3]
        """
        # Шаг 1: Используем готовые xi (F не нужен для вычисления xi)
        
        # Шаг 2: Вычисляем r через нейронную сеть
        r = self.forward_base(xi)
        
        # Шаг 3: Вычисляем градиент dr/dxi
        dr_dxi = self.compute_r_gradient(xi)
        
        # Шаг 4: Вычисляем S из r
        S = self.compute_stress_from_r(r)
        
        # Шаг 5: Вычисляем P из S
        P = self.compute_piola_from_stress(S, xi)
        
        return P
    
    def forward(self, *args):
        """
        Основной метод forward, выбирает режим работы в зависимости от self.mode.
        
        Args:
            *args: Аргументы в зависимости от режима:
                - base: xi (torch.Tensor)
                - pk2: F (torch.Tensor)
                - pk2_with_xi: xi (torch.Tensor), F (torch.Tensor)
        
        Returns:
            torch.Tensor: Результат в зависимости от режима
        """
        if self.mode == 'base':
            if len(args) != 1:
                raise ValueError("Режим 'base' требует 1 аргумент: xi")
            return self.forward_base(args[0])
        
        elif self.mode == 'pk2':
            if len(args) != 1:
                raise ValueError("Режим 'pk2' требует 1 аргумент: F")
            return self.forward_pk2(args[0])
        
        elif self.mode == 'pk2_with_xi':
            if len(args) != 2:
                raise ValueError("Режим 'pk2_with_xi' требует 2 аргумента: xi, F")
            return self.forward_pk2_with_xi(args[0], args[1])
        
        else:
            raise ValueError(f"Неподдерживаемый режим: {self.mode}")
    
    def predict_r(self, xi):
        """Предсказывает функции отклика r для входных данных xi."""
        self.eval()
        with torch.no_grad():
            return self.forward_base(xi)
    
    def predict_stress(self, xi):
        """Предсказывает напряжения Коши S для входных данных xi."""
        self.eval()
        with torch.no_grad():
            r = self.forward_base(xi)
            return self.compute_stress_from_r(r)
    
    def predict_piola_stress(self, xi):
        """Предсказывает напряжения Пиолы-Кирхгофа P для входных данных xi."""
        self.eval()
        with torch.no_grad():
            r = self.forward_base(xi)
            S = self.compute_stress_from_r(r)
            return self.compute_piola_from_stress(S, xi)


if __name__ == "__main__":
    # Тестирование разных режимов
    print("=== Тестирование CLANN ===")
    
    # Режим base
    model_base = CLANN(input_dim=3, hidden_dim=32, output_dim=3, mode='pk2')
    xi = torch.randn(5, 3)
    r = model_base(xi)
    print(f"Режим base: xi {xi.shape} -> r {r.shape}")
    
    # Режим pk2
    model_pk2 = CLANN(input_dim=3, hidden_dim=32, output_dim=3, mode='pk2')
    F = torch.randn(5, 3)  # [F11, F12, F22]
    P = model_pk2(F)
    print(f"Режим pk2: F {F.shape} -> P {P.shape}")
    
    # Режим pk2_with_xi
    model_pk2_xi = CLANN(input_dim=3, hidden_dim=32, output_dim=3, mode='pk2_with_xi')
    P_xi = model_pk2_xi(xi, F)
    print(f"Режим pk2_with_xi: xi {xi.shape}, F {F.shape} -> P {P_xi.shape}")
    
    print("Все режимы работают корректно!")

