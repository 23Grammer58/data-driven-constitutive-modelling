import math
import numpy as np
from numpy.linalg import inv, det
import torch

def calculate_xi_from_cauchy_green(C):
    """
    Расчет параметров xi1, xi2, xi3 из тензора деформации Коши-Грина.
    
    Параметры:
    C - тензор деформации Коши-Грина (массив 2x2 или 3x3)
    
    Возвращает:
    xi1, xi2, xi3 - параметры деформации
    """
    # Извлекаем компоненты тензора
    C11 = C[0, 0]
    C12 = C[0, 1]
    C22 = C[1, 1]
    
    # Расчет параметров xi согласно формулам
    xi1 = 0.5 * math.log(C11)
    xi2 = 0.5 * math.log(C22 - (C12**2 / C11))
    xi3 = C12 / C11
    
    return xi1, xi2, xi3

# Примеры функций энергии деформации

def mooney_rivlin_energy(I1, I2, I3, params):
    """
    Функция энергии деформации для модели Муни-Ривлина.
    
    Параметры:
    I1, I2, I3 - инварианты тензора деформации
    params - словарь параметров модели (C10, C01)
    
    Возвращает:
    W - энергия деформации
    """
    C10 = params.get('C10', 0.0)
    C01 = params.get('C01', 0.0)
    
    return C10 * (I1 - 3) + C01 * (I2 - 3)

def yeoh_energy(I1, I2, I3, params):
    """
    Функция энергии деформации для модели Йео.
    
    Параметры:
    I1, I2, I3 - инварианты тензора деформации
    params - словарь параметров модели (C10, C20, C30)
    
    Возвращает:
    W - энергия деформации
    """
    C10 = params.get('C10', 0.0)
    C20 = params.get('C20', 0.0)
    C30 = params.get('C30', 0.0)
    
    return C10 * (I1 - 3) + C20 * (I1 - 3)**2 + C30 * (I1 - 3)**3

def polynomial_energy(I1, I2, I3, params):
    """
    Функция энергии деформации для полиномиальной модели.
    
    Параметры:
    I1, I2, I3 - инварианты тензора деформации
    params - словарь параметров модели (C10, C01, C11, C20, C02)
    
    Возвращает:
    W - энергия деформации
    """
    C10 = params.get('C10', 0.0)
    C01 = params.get('C01', 0.0)
    C11 = params.get('C11', 0.0)
    C20 = params.get('C20', 0.0)
    C02 = params.get('C02', 0.0)
    
    return (C10 * (I1 - 3) + C01 * (I2 - 3) +
            C11 * (I1 - 3) * (I2 - 3) +
            C20 * (I1 - 3)**2 + C02 * (I2 - 3)**2)

def calculate_response_functions(C, energy_function, params):
    """
    Расчет функций отклика r1, r2, r3 для случая W = W(xi(C(F))).
    
    Параметры:
    C - тензор деформации Коши-Грина
    energy_function - функция энергии деформации
    params - словарь параметров модели
    
    Возвращает:
    r1, r2, r3 - функции отклика
    """
    # Шаг 1: Вычисляем параметры xi из тензора C
    xi1, xi2, xi3 = calculate_xi_from_cauchy_green(C)
    
    # Шаг 2: Создаем тензоры для параметров xi с требованием вычисления градиента
    xi1_tensor = torch.tensor(xi1, requires_grad=True)
    xi2_tensor = torch.tensor(xi2, requires_grad=True)
    xi3_tensor = torch.tensor(xi3, requires_grad=True)
    
    # Шаг 3: Вычисляем инварианты из параметров xi
    # Здесь lambda1, lambda2 - главные растяжения, lambda3 - параметр сдвига
    lambda1 = torch.exp(xi1_tensor)
    lambda2 = torch.exp(xi2_tensor)
    lambda3 = xi3_tensor
    
    # Шаг 4: Вычисляем инварианты деформации из параметров xi
    I1_xi = lambda1**2 + lambda2**2 + lambda3**2
    I2_xi = lambda1**2 * lambda2**2 + lambda2**2 * lambda3**2 + lambda3**2 * lambda1**2
    I3_xi = lambda1**2 * lambda2**2 * lambda3**2
    
    # Шаг 5: Вычисляем энергию деформации через параметры xi
    # W = W(xi(C(F)))
    W_xi = energy_function(I1_xi, I2_xi, I3_xi, params)
    
    # Шаг 6: Вычисляем градиенты энергии деформации по параметрам xi
    # dW/dxi = (dW/dI) * (dI/dxi)
    xi_inputs = [xi1_tensor, xi2_tensor, xi3_tensor]
    xi_gradients = torch.autograd.grad(W_xi, xi_inputs, create_graph=False, retain_graph=False, allow_unused=True)
    
    # Шаг 7: Извлекаем значения градиентов
    dWdxi1 = xi_gradients[0].item() if xi_gradients[0] is not None else 0.0
    dWdxi2 = xi_gradients[1].item() if xi_gradients[1] is not None else 0.0
    dWdxi3 = xi_gradients[2].item() if xi_gradients[2] is not None else 0.0
    
    # Шаг 8: Вычисляем функции отклика
    # r1 = dW/dxi1 - отклик на растяжение в направлении 1
    r1 = dWdxi1
    
    # r2 = dW/dxi2 - отклик на растяжение в направлении 2
    r2 = dWdxi2
    
    # r3 = sqrt(C22*C11 - C12^2) * dW/dxi3 - отклик на сдвиг
    # Множитель sqrt(C22*C11 - C12^2) связан с геометрией деформации
    C11 = C[0, 0]
    C12 = C[0, 1]
    C22 = C[1, 1]
    r3 = math.sqrt(C22 * C11 - C12**2) * dWdxi3
    
    return r1, r2, r3

def calculate_response_functions_analytical(C, energy_function, params):
    """
    Расчет функций отклика r1, r2, r3 для случая W = W(xi(C(F))) с использованием аналитических формул.
    
    Параметры:
    C - тензор деформации Коши-Грина
    energy_function - функция энергии деформации
    params - словарь параметров модели
    
    Возвращает:
    r1, r2, r3 - функции отклика
    """
    # Шаг 1: Вычисляем параметры xi из тензора C
    xi1, xi2, xi3 = calculate_xi_from_cauchy_green(C)
    
    # Шаг 2: Вычисляем главные растяжения из параметров xi
    lambda1 = math.exp(xi1)
    lambda2 = math.exp(xi2)
    lambda3 = xi3
    
    # Шаг 3: Вычисляем инварианты деформации из параметров xi
    I1 = lambda1**2 + lambda2**2 + lambda3**2
    I2 = lambda1**2 * lambda2**2 + lambda2**2 * lambda3**2 + lambda3**2 * lambda1**2
    I3 = lambda1**2 * lambda2**2 * lambda3**2
    
    # Шаг 4: Вычисляем производные инвариантов по параметрам xi
    # dI1/dxi1 = 2 * lambda1^2
    dI1_dxi1 = 2 * lambda1**2
    # dI1/dxi2 = 2 * lambda2^2
    dI1_dxi2 = 2 * lambda2**2
    # dI1/dxi3 = 2 * lambda3
    dI1_dxi3 = 2 * lambda3
    
    # dI2/dxi1 = 2 * lambda1^2 * (lambda2^2 + lambda3^2)
    dI2_dxi1 = 2 * lambda1**2 * (lambda2**2 + lambda3**2)
    # dI2/dxi2 = 2 * lambda2^2 * (lambda1^2 + lambda3^2)
    dI2_dxi2 = 2 * lambda2**2 * (lambda1**2 + lambda3**2)
    # dI2/dxi3 = 2 * lambda3 * (lambda1^2 + lambda2^2)
    dI2_dxi3 = 2 * lambda3 * (lambda1**2 + lambda2**2)
    
    # dI3/dxi1 = 2 * lambda1^2 * lambda2^2 * lambda3^2
    dI3_dxi1 = 2 * lambda1**2 * lambda2**2 * lambda3**2
    # dI3/dxi2 = 2 * lambda1^2 * lambda2^2 * lambda3^2
    dI3_dxi2 = 2 * lambda1**2 * lambda2**2 * lambda3**2
    # dI3/dxi3 = lambda1^2 * lambda2^2 * lambda3
    dI3_dxi3 = lambda1**2 * lambda2**2 * lambda3
    
    # Шаг 5: Вычисляем производные энергии деформации по инвариантам
    # Для этого используем PyTorch и автоматическое дифференцирование
    I1_tensor = torch.tensor(I1, requires_grad=True)
    I2_tensor = torch.tensor(I2, requires_grad=True)
    I3_tensor = torch.tensor(I3, requires_grad=True)
    
    # Вычисляем энергию деформации
    W = energy_function(I1_tensor, I2_tensor, I3_tensor, params)
    
    # Вычисляем градиенты энергии деформации по инвариантам
    inputs = [I1_tensor, I2_tensor, I3_tensor]
    gradients = torch.autograd.grad(W, inputs, create_graph=False, retain_graph=False, allow_unused=True)
    
    # Извлекаем значения градиентов
    dW_dI1 = gradients[0].item() if gradients[0] is not None else 0.0
    dW_dI2 = gradients[1].item() if gradients[1] is not None else 0.0
    dW_dI3 = gradients[2].item() if gradients[2] is not None else 0.0
    
    # Шаг 6: Вычисляем производные энергии деформации по параметрам xi
    # используя правило цепи: dW/dxi = (dW/dI1) * (dI1/dxi) + (dW/dI2) * (dI2/dxi) + (dW/dI3) * (dI3/dxi)
    dW_dxi1 = dW_dI1 * dI1_dxi1 + dW_dI2 * dI2_dxi1 + dW_dI3 * dI3_dxi1
    dW_dxi2 = dW_dI1 * dI1_dxi2 + dW_dI2 * dI2_dxi2 + dW_dI3 * dI3_dxi2
    dW_dxi3 = dW_dI1 * dI1_dxi3 + dW_dI2 * dI2_dxi3 + dW_dI3 * dI3_dxi3
    
    # Шаг 7: Вычисляем функции отклика
    # r1 = dW/dxi1 - отклик на растяжение в направлении 1
    r1 = dW_dxi1
    
    # r2 = dW/dxi2 - отклик на растяжение в направлении 2
    r2 = dW_dxi2
    
    # r3 = sqrt(C22*C11 - C12^2) * dW/dxi3 - отклик на сдвиг
    C11 = C[0, 0]
    C12 = C[0, 1]
    C22 = C[1, 1]
    r3 = math.sqrt(C22 * C11 - C12**2) * dW_dxi3
    
    return r1, r2, r3

def calculate_hyperelastic_stress(F, energy_function=polynomial_energy, params=None, compressible=False, calculate_response=False, use_analytical=False):
    """
    Вычисление тензоров напряжений для гиперупругой модели материала.
    
    Параметры:
    F - градиент деформации
    energy_function - функция энергии деформации (по умолчанию polynomial_energy)
    params - словарь параметров модели
    compressible - флаг, указывающий, является ли материал сжимаемым
    calculate_response - флаг, указывающий, нужно ли вычислять функции отклика r1, r2, r3
    use_analytical - флаг, указывающий, использовать ли аналитические формулы для расчета функций отклика
    
    Возвращает:
    P - тензор напряжений Пиолы-Кирхгофа
    S - тензор напряжений Коши
    r - функции отклика [r1, r2, r3] (если calculate_response=True)
    """
    # Если параметры не указаны, используем значения по умолчанию
    if params is None:
        params = {
            'C10': 0.0,
            'C01': 0.172056,
            'C11': 0.0,
            'C20': 0.0,
            'C02': 0.05493547
        }
    
    # Преобразуем numpy массив в тензор PyTorch с требованием вычисления градиента
    F_torch = torch.tensor(F, dtype=torch.float64, requires_grad=True)
    
    # Вычисляем тензор Коши-Грина C = F^T * F
    C_torch = F_torch.t() @ F_torch
    
    # Вычисляем инварианты деформации
    I1 = torch.trace(C_torch)
    I2 = 0.5 * (I1**2 - torch.trace(C_torch @ C_torch))
    I3 = torch.det(C_torch)
    
    # Создаем тензоры для инвариантов с требованием вычисления градиента
    I1_tensor = torch.tensor(I1.item(), requires_grad=True)
    I2_tensor = torch.tensor(I2.item(), requires_grad=True)
    I3_tensor = torch.tensor(I3.item(), requires_grad=True)
    
    # Вычисление энергии деформации с использованием переданной функции
    W = energy_function(I1_tensor, I2_tensor, I3_tensor, params)
    
    # Создаем список входных тензоров для автоматического дифференцирования
    inputs = [I1_tensor, I2_tensor, I3_tensor]
    
    # Вычисляем градиенты энергии деформации по всем инвариантам за один проход
    # Добавляем параметр allow_unused=True для обработки неиспользуемых тензоров
    gradients = torch.autograd.grad(W, inputs, create_graph=False, retain_graph=False, allow_unused=True)
    
    # Извлекаем значения градиентов
    dWdI1 = gradients[0].item() if gradients[0] is not None else 0.0
    dWdI2 = gradients[1].item() if gradients[1] is not None else 0.0
    dWdI3 = gradients[2].item() if gradients[2] is not None else 0.0
    
    # Преобразуем обратно в numpy для дальнейших вычислений
    F_np = F_torch.detach().numpy()
    C_np = F_np.T @ F_np
    I1_np = I1.item()
    I2_np = I2.item()
    I3_np = I3.item()
    
    # Вычисление производных инвариантов по F
    dI1dF = 2 * F_np  # dI1/dF = 2F
    
    # dI2/dF = 2(I1*F - F*F^T*F)
    FFT = F_np @ F_np.T
    dI2dF = 2 * (I1_np * F_np - F_np @ FFT)
    
    # dI3/dF = 2*I3*F^(-T)
    F_inv_T = inv(F_np).T
    dI3dF = 2 * I3_np * F_inv_T
    
    # Вычисление тензора напряжений Пиолы-Кирхгофа
    P = dWdI1 * dI1dF + dWdI2 * dI2dF + dWdI3 * dI3dF
    
    if compressible:
        # Вычисление тензора напряжений Коши (S = 1/det(F) * F * P * F^T)
        J = det(F_np)
        S = (1.0 / J) * F_np @ P @ F_np.T
    else:
        S = F_np @ P @ F_np.T
    
    # Если требуется расчет функций отклика
    if calculate_response:
        # Выбираем метод расчета функций отклика
        if use_analytical:
            # Используем аналитические формулы
            r1, r2, r3 = calculate_response_functions_analytical(C_np, energy_function, params)
        else:
            # Используем автоматическое дифференцирование
            r1, r2, r3 = calculate_response_functions(C_np, energy_function, params)
        
        return P, S, [r1, r2, r3]
    
    return P, S

# Пример использования функций расчета отклика
if __name__ == "__main__":
    # Создаем тестовый градиент деформации
    F = np.array([
        [1.2, 0.1],
        [0.0, 1.1]
    ])
    
    # Параметры модели Муни-Ривлина
    params = {
        'C10': 0.5,
        'C01': 0.2
    }
    
    # Вычисляем тензоры напряжений и функции отклика
    # с использованием автоматического дифференцирования
    P, S, r_auto = calculate_hyperelastic_stress(
        F, 
        energy_function=mooney_rivlin_energy, 
        params=params, 
        calculate_response=True,
        use_analytical=False
    )
    
    # Вычисляем тензоры напряжений и функции отклика
    # с использованием аналитических формул
    P, S, r_analytical = calculate_hyperelastic_stress(
        F, 
        energy_function=mooney_rivlin_energy, 
        params=params, 
        calculate_response=True,
        use_analytical=True
    )
    
    # Выводим результаты
    print("Тензор напряжений Пиолы-Кирхгофа:")
    print(P)
    print("\nТензор напряжений Коши:")
    print(S)
    
    print("\nФункции отклика (автоматическое дифференцирование):")
    print(f"r1 = {r_auto[0]}")
    print(f"r2 = {r_auto[1]}")
    print(f"r3 = {r_auto[2]}")
    
    print("\nФункции отклика (аналитические формулы):")
    print(f"r1 = {r_analytical[0]}")
    print(f"r2 = {r_analytical[1]}")
    print(f"r3 = {r_analytical[2]}")
    
    # Проверка разницы между методами
    diff_r1 = abs(r_auto[0] - r_analytical[0])
    diff_r2 = abs(r_auto[1] - r_analytical[1])
    diff_r3 = abs(r_auto[2] - r_analytical[2])
    
    print("\nРазница между методами:")
    print(f"Разница r1: {diff_r1}")
    print(f"Разница r2: {diff_r2}")
    print(f"Разница r3: {diff_r3}")
    
    # Вычисляем тензор Коши-Грина
    C = F.T @ F
    
    # Вычисляем параметры xi
    xi1, xi2, xi3 = calculate_xi_from_cauchy_green(C)
    
    print("\nПараметры xi:")
    print(f"xi1 = {xi1}")
    print(f"xi2 = {xi2}")
    print(f"xi3 = {xi3}")
    
    # Вычисляем главные растяжения
    lambda1 = math.exp(xi1)
    lambda2 = math.exp(xi2)
    
    print("\nГлавные растяжения:")
    print(f"lambda1 = {lambda1}")
    print(f"lambda2 = {lambda2}")
    print(f"lambda3 (сдвиг) = {xi3}")
    
    # Вычисляем инварианты
    I1 = lambda1**2 + lambda2**2 + xi3**2
    I2 = lambda1**2 * lambda2**2 + lambda2**2 * xi3**2 + xi3**2 * lambda1**2
    I3 = lambda1**2 * lambda2**2 * xi3**2
    
    print("\nИнварианты деформации:")
    print(f"I1 = {I1}")
    print(f"I2 = {I2}")
    print(f"I3 = {I3}")
    
    

    