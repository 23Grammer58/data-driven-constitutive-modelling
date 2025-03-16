import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from clann import CLANN


def train_and_evaluate(hidden_dim, train_X, train_y, val_X, val_y, epochs=5000, lr=0.01, 
                      C11=1.0, C12=0.5, C22=1.0, direct_stress=False):
    """
    Обучает модель CLANN с заданным количеством нейронов на скрытом слое
    и вычисляет ошибку на валидационной выборке.
    
    Args:
        hidden_dim (int): Количество нейронов на скрытом слое
        train_X (torch.Tensor): Входные данные для обучения
        train_y (torch.Tensor): Целевые данные для обучения (r или S в зависимости от direct_stress)
        val_X (torch.Tensor): Входные данные для валидации
        val_y (torch.Tensor): Целевые данные для валидации (r или S в зависимости от direct_stress)
        epochs (int): Количество эпох обучения
        lr (float): Скорость обучения
        C11, C12, C22: Параметры для вычисления напряжений
        direct_stress (bool): Обучать ли модель напрямую предсказывать напряжения S
        
    Returns:
        tuple: (обученная модель, ошибка на валидационной выборке)
    """
    model = CLANN(input_dim=train_X.shape[1], hidden_dim=hidden_dim, output_dim=train_y.shape[1], 
                 C11=C11, C12=C12, C22=C22, direct_stress=direct_stress)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Обучение модели
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        
        # Опционально: вывод прогресса обучения
        if (epoch + 1) % 1000 == 0:
            print(f"Эпоха {epoch+1}/{epochs}, Потери: {loss.item():.6f}")

    # Вычисление ошибки на валидации
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_X)
        val_loss = criterion(val_outputs, val_y).item()

    return model, val_loss


def cross_validate(hidden_dim, X, y, n_splits=5, epochs=5000, lr=0.01, 
                  C11=1.0, C12=0.5, C22=1.0, direct_stress=False):
    """
    Выполняет кросс-валидацию модели CLANN с заданным количеством нейронов.
    
    Args:
        hidden_dim (int): Количество нейронов на скрытом слое
        X (torch.Tensor): Входные данные
        y (torch.Tensor): Целевые данные (r или S в зависимости от direct_stress)
        n_splits (int): Количество разбиений для кросс-валидации
        epochs (int): Количество эпох обучения
        lr (float): Скорость обучения
        C11, C12, C22: Параметры для вычисления напряжений
        direct_stress (bool): Обучать ли модель напрямую предсказывать напряжения S
        
    Returns:
        tuple: (лучшая модель, средняя ошибка, стандартное отклонение ошибки)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_losses = []
    best_model = None
    best_loss = float('inf')
    
    print(f"\nНачинаем кросс-валидацию для {hidden_dim} нейронов ({n_splits} разбиений)...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nРазбиение {fold+1}/{n_splits}")
        
        # Подготовка данных для текущего разбиения
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Обучение модели
        model, val_loss = train_and_evaluate(hidden_dim, X_train, y_train, X_val, y_val, 
                                            epochs, lr, C11, C12, C22, direct_stress)
        fold_losses.append(val_loss)
        
        print(f"Разбиение {fold+1}: Ошибка валидации = {val_loss:.6f}")
        
        # Сохраняем лучшую модель
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
    
    # Вычисляем среднюю ошибку и стандартное отклонение
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    
    print(f"\nРезультаты кросс-валидации для {hidden_dim} нейронов:")
    print(f"Средняя ошибка: {mean_loss:.6f} ± {std_loss:.6f}")
    
    return best_model, mean_loss, std_loss


def find_optimal_neurons(train_X, train_y, val_X, val_y, 
                         start_neurons=32, max_neurons=512, 
                         epochs=5000, lr=0.01, C11=1.0, C12=0.5, C22=1.0,
                         direct_stress=False):
    """
    Находит оптимальное количество нейронов на скрытом слое
    с помощью бинарного поиска.
    
    Args:
        train_X, train_y, val_X, val_y: Данные для обучения и валидации
        start_neurons (int): Начальное количество нейронов
        max_neurons (int): Максимальное количество нейронов
        epochs (int): Количество эпох обучения
        lr (float): Скорость обучения
        C11, C12, C22: Параметры для вычисления напряжений
        direct_stress (bool): Обучать ли модель напрямую предсказывать напряжения S
        
    Returns:
        tuple: (лучшая модель, оптимальное количество нейронов, лучшая ошибка)
    """
    print(f"Начинаем поиск оптимального количества нейронов...")
    
    # Начальная конфигурация
    current_neurons = start_neurons
    best_model, best_loss = train_and_evaluate(current_neurons, train_X, train_y, val_X, val_y, 
                                              epochs, lr, C11, C12, C22, direct_stress)
    best_neurons = current_neurons
    
    print(f"Начальная конфигурация: {current_neurons} нейронов - валидационная ошибка: {best_loss:.6f}")
    
    # Список размеров для проверки (степени двойки)
    neuron_sizes = [2**i for i in range(int(torch.log2(torch.tensor(start_neurons))), 
                                        int(torch.log2(torch.tensor(max_neurons)))+1)]
    
    # Проверяем каждый размер
    for neurons in neuron_sizes[1:]:  # Пропускаем начальный размер, который уже проверили
        print(f"\nПроверяем {neurons} нейронов...")
        model, loss = train_and_evaluate(neurons, train_X, train_y, val_X, val_y, 
                                        epochs, lr, C11, C12, C22, direct_stress)
        
        print(f"Результат: {neurons} нейронов - валидационная ошибка: {loss:.6f}")
        
        if loss < best_loss:
            best_loss = loss
            best_neurons = neurons
            best_model = model
            print(f"Улучшение! Новая лучшая конфигурация: {best_neurons} нейронов, ошибка = {best_loss:.6f}")
    
    print(f"\nОптимальная конфигурация: {best_neurons} нейронов на скрытом слое с валидационной ошибкой: {best_loss:.6f}")
    
    return best_model, best_neurons, best_loss


def evaluate_model_performance(model, X, y_r, C11=1.0, C12=0.5, C22=1.0):
    """
    Оценивает точность предсказания модели для r, S и P.
    
    Args:
        model (CLANN): Обученная модель
        X (torch.Tensor): Входные данные
        y_r (torch.Tensor): Целевые значения r
        C11, C12, C22: Параметры для вычисления напряжений
        
    Returns:
        dict: Словарь с метриками оценки
    """
    model.eval()
    with torch.no_grad():
        # Получаем предсказанные значения r
        if model.direct_stress:
            S_pred, r_pred = model.forward(X, return_r=True)
        else:
            r_pred = model(X)
        
        # Вычисляем истинные напряжения S на основе истинных значений r
        S_true = torch.zeros_like(y_r)
        S_true[:, 0] = y_r[:, 0]  # S11 = r1
        S_true[:, 1] = y_r[:, 1]  # S22 = r2
        S_true[:, 2] = (1.0 / C11) * (torch.sqrt(C22 * C11) - C12) * y_r[:, 2]  # S12
        
        # Получаем предсказанные напряжения S
        if model.direct_stress:
            S_pred = model(X)
        else:
            S_pred = model.compute_stress(r_pred)
        
        # Получаем предсказанные напряжения P
        P_pred = model.compute_piola_stress(S_pred, X)
        
        # Вычисляем истинные напряжения P на основе истинных значений S
        P_true = model.compute_piola_stress(S_true, X)
        
        # Вычисляем ошибки
        mse_r = nn.MSELoss()(r_pred, y_r).item()
        mse_S = nn.MSELoss()(S_pred, S_true).item()
        mse_P = nn.MSELoss()(P_pred, P_true).item()
        
        # Вычисляем относительные ошибки для каждой компоненты S
        rel_error_S11 = torch.mean(torch.abs((S_pred[:, 0] - S_true[:, 0]) / (S_true[:, 0] + 1e-8))).item()
        rel_error_S22 = torch.mean(torch.abs((S_pred[:, 1] - S_true[:, 1]) / (S_true[:, 1] + 1e-8))).item()
        rel_error_S12 = torch.mean(torch.abs((S_pred[:, 2] - S_true[:, 2]) / (S_true[:, 2] + 1e-8))).item()
        
        # Вычисляем относительные ошибки для каждой компоненты P
        rel_error_P11 = torch.mean(torch.abs((P_pred[:, 0] - P_true[:, 0]) / (P_true[:, 0] + 1e-8))).item()
        rel_error_P22 = torch.mean(torch.abs((P_pred[:, 1] - P_true[:, 1]) / (P_true[:, 1] + 1e-8))).item()
        rel_error_P12 = torch.mean(torch.abs((P_pred[:, 2] - P_true[:, 2]) / (P_true[:, 2] + 1e-8))).item()
        
    return {
        'mse_r': mse_r,
        'mse_S': mse_S,
        'mse_P': mse_P,
        'rel_error_S11': rel_error_S11,
        'rel_error_S22': rel_error_S22,
        'rel_error_S12': rel_error_S12,
        'rel_error_P11': rel_error_P11,
        'rel_error_P22': rel_error_P22,
        'rel_error_P12': rel_error_P12
    }


def prepare_stress_data(X, y_r, C11=1.0, C12=0.5, C22=1.0):
    """
    Подготавливает данные для обучения модели по напряжениям.
    
    Args:
        X (torch.Tensor): Входные данные
        y_r (torch.Tensor): Целевые значения r
        C11, C12, C22: Параметры для вычисления напряжений
        
    Returns:
        torch.Tensor: Целевые значения S
    """
    # Вычисляем истинные напряжения S на основе истинных значений r
    S = torch.zeros_like(y_r)
    S[:, 0] = y_r[:, 0]  # S11 = r1
    S[:, 1] = y_r[:, 1]  # S22 = r2
    S[:, 2] = (1.0 / C11) * (torch.sqrt(C22 * C11) - C12) * y_r[:, 2]  # S12
    
    return S


def main():
    # Загрузка данных
    print("Загрузка данных...")
    df = pd.read_csv("Filtered_ConstEqXiDeNoise.csv", header=0)
    X = torch.Tensor(df[["xi1", "xi2", "xi3"]].values)
    y_r = torch.Tensor(df[["r1", "r2", "r3"]].values)
    
    # Параметры для вычисления напряжений
    C11 = 1.0
    C12 = 0.5
    C22 = 1.0
    
    # Флаг, указывающий, обучать ли модель напрямую предсказывать напряжения
    direct_stress = True
    
    # Подготавливаем данные в зависимости от режима обучения
    if direct_stress:
        print("Подготовка данных для обучения по напряжениям...")
        y = prepare_stress_data(X, y_r, C11, C12, C22)
        print("Модель будет обучаться напрямую предсказывать напряжения S")
    else:
        y = y_r
        print("Модель будет обучаться предсказывать функции отклика r")
    
    # Разбиваем данные на обучающую и валидационную выборки
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Размер обучающей выборки: {train_X.shape[0]}, размер валидационной выборки: {val_X.shape[0]}")
    
    # Поиск оптимального количества нейронов
    _, best_neurons, _ = find_optimal_neurons(
        train_X, train_y, val_X, val_y, 
        start_neurons=32,  # Начинаем с 32 нейронов
        max_neurons=512,   # Максимум 512 нейронов
        epochs=5000,       # 5000 эпох для каждой конфигурации
        lr=0.01,           # Скорость обучения
        C11=C11, C12=C12, C22=C22,  # Параметры для вычисления напряжений
        direct_stress=direct_stress  # Режим обучения
    )
    
    # Проводим кросс-валидацию с оптимальным количеством нейронов
    print("\n" + "="*50)
    print(f"Проводим кросс-валидацию для оптимального количества нейронов: {best_neurons}")
    print("="*50)
    
    # Объединяем обучающую и валидационную выборки для кросс-валидации
    final_model, mean_loss, std_loss = cross_validate(
        best_neurons, 
        X,  # Используем все данные для кросс-валидации
        y, 
        n_splits=5,  # 5 разбиений
        epochs=5000, 
        lr=0.01,
        C11=C11, C12=C12, C22=C22,  # Параметры для вычисления напряжений
        direct_stress=direct_stress  # Режим обучения
    )
    
    # Оцениваем точность предсказания модели
    print("\n" + "="*50)
    print("Оценка точности предсказания модели")
    print("="*50)
    
    metrics = evaluate_model_performance(final_model, X, y_r, C11, C12, C22)
    
    print(f"MSE для r: {metrics['mse_r']:.6f}")
    print(f"MSE для S: {metrics['mse_S']:.6f}")
    print(f"MSE для P: {metrics['mse_P']:.6f}")
    print("\nОтносительные ошибки для S:")
    print(f"S11: {metrics['rel_error_S11']:.6f}")
    print(f"S22: {metrics['rel_error_S22']:.6f}")
    print(f"S12: {metrics['rel_error_S12']:.6f}")
    print("\nОтносительные ошибки для P:")
    print(f"P11: {metrics['rel_error_P11']:.6f}")
    print(f"P22: {metrics['rel_error_P22']:.6f}")
    print(f"P12: {metrics['rel_error_P12']:.6f}")
    
    # Экспортируем лучшую модель в формат ONNX
    print("\nЭкспорт лучшей модели в ONNX формат...")
    final_model.eval()
    dummy_input = torch.randn(1, X.shape[1])
    
    # Определяем суффикс для имени файла в зависимости от режима обучения
    mode_suffix = "direct" if direct_stress else "indirect"
    
    # Экспортируем модель с выходом r
    class CLANNResponseWrapper(nn.Module):
        def __init__(self, model):
            super(CLANNResponseWrapper, self).__init__()
            self.model = model
            
        def forward(self, x):
            return self.model.predict_r(x)
    
    response_model = CLANNResponseWrapper(final_model)
    torch.onnx.export(
        response_model, dummy_input, f"clann_r_{best_neurons}_{mode_suffix}.onnx",
        input_names=["input"], output_names=["output_r"],
        opset_version=11
    )
    print(f"Модель для предсказания функций отклика r экспортирована в файл clann_r_{best_neurons}_{mode_suffix}.onnx")
    
    # Экспортируем модель с выходом S
    class CLANNStressWrapper(nn.Module):
        def __init__(self, model):
            super(CLANNStressWrapper, self).__init__()
            self.model = model
            
        def forward(self, x):
            return self.model.predict_stress(x)
    
    stress_model = CLANNStressWrapper(final_model)
    torch.onnx.export(
        stress_model, dummy_input, f"clann_stress_{best_neurons}_{mode_suffix}.onnx",
        input_names=["input"], output_names=["output_stress"],
        opset_version=11
    )
    print(f"Модель для предсказания напряжений S экспортирована в файл clann_stress_{best_neurons}_{mode_suffix}.onnx")
    
    # Экспортируем модель с выходом P
    class CLANNPiolaWrapper(nn.Module):
        def __init__(self, model):
            super(CLANNPiolaWrapper, self).__init__()
            self.model = model
            
        def forward(self, x):
            return self.model.predict_piola_stress(x)
    
    piola_model = CLANNPiolaWrapper(final_model)
    torch.onnx.export(
        piola_model, dummy_input, f"clann_piola_{best_neurons}_{mode_suffix}.onnx",
        input_names=["input"], output_names=["output_piola"],
        opset_version=11
    )
    print(f"Модель для предсказания напряжений Пиолы-Кирхгофа P экспортирована в файл clann_piola_{best_neurons}_{mode_suffix}.onnx")
    
    # Выводим информацию о средней ошибке
    if direct_stress:
        print(f"Средняя ошибка кросс-валидации для S: {mean_loss:.6f} ± {std_loss:.6f}")
    else:
        print(f"Средняя ошибка кросс-валидации для r: {mean_loss:.6f} ± {std_loss:.6f}")


if __name__ == "__main__":
    main() 