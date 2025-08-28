import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, KFold
from clann import CLANN


def train_and_evaluate(hidden_dim, train_X, train_y, val_X, val_y, epochs=5000, lr=0.01, 
                      C11=1.0, C12=0.5, C22=1.0, direct_stress=False, direct_piola=False):
    """
    Обучает модель CLANN с заданным количеством нейронов на скрытом слое
    и вычисляет ошибку на валидационной выборке.
    
    Args:
        hidden_dim (int): Количество нейронов на скрытом слое
        train_X (torch.Tensor): Входные данные для обучения
        train_y (torch.Tensor): Целевые данные для обучения (r, S или P в зависимости от режима)
        val_X (torch.Tensor): Входные данные для валидации
        val_y (torch.Tensor): Целевые данные для валидации (r, S или P в зависимости от режима)
        epochs (int): Количество эпох обучения
        lr (float): Скорость обучения
        C11, C12, C22: Параметры для вычисления напряжений
        direct_stress (bool): Обучать ли модель напрямую предсказывать напряжения S
        direct_piola (bool): Обучать ли модель напрямую предсказывать напряжения P
        
    Returns:
        tuple: (обученная модель, ошибка на валидационной выборке)
    """
    model = CLANN(input_dim=train_X.shape[1], hidden_dim=hidden_dim, output_dim=train_y.shape[1], 
                 C11=C11, C12=C12, C22=C22, direct_stress=direct_stress, direct_piola=direct_piola)
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
                  C11=1.0, C12=0.5, C22=1.0, direct_stress=False, direct_piola=False):
    """
    Выполняет кросс-валидацию модели CLANN с заданным количеством нейронов.
    
    Args:
        hidden_dim (int): Количество нейронов на скрытом слое
        X (torch.Tensor): Входные данные
        y (torch.Tensor): Целевые данные (r, S или P в зависимости от режима)
        n_splits (int): Количество разбиений для кросс-валидации
        epochs (int): Количество эпох обучения
        lr (float): Скорость обучения
        C11, C12, C22: Параметры для вычисления напряжений
        direct_stress (bool): Обучать ли модель напрямую предсказывать напряжения S
        direct_piola (bool): Обучать ли модель напрямую предсказывать напряжения P
        
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
                                            epochs, lr, C11, C12, C22, direct_stress, direct_piola)
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
                         direct_stress=False, direct_piola=False):
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
        direct_piola (bool): Обучать ли модель напрямую предсказывать напряжения P
        
    Returns:
        tuple: (лучшая модель, оптимальное количество нейронов, лучшая ошибка)
    """
    print(f"Начинаем поиск оптимального количества нейронов...")
    
    # Начальная конфигурация
    current_neurons = start_neurons
    best_model, best_loss = train_and_evaluate(current_neurons, train_X, train_y, val_X, val_y, 
                                              epochs, lr, C11, C12, C22, direct_stress, direct_piola)
    best_neurons = current_neurons
    
    print(f"Начальная конфигурация: {current_neurons} нейронов - валидационная ошибка: {best_loss:.6f}")
    
    # Список размеров для проверки (степени двойки)
    neuron_sizes = [2**i for i in range(int(torch.log2(torch.tensor(start_neurons))), 
                                        int(torch.log2(torch.tensor(max_neurons)))+1)]
    
    # Проверяем каждый размер
    for neurons in neuron_sizes[1:]:  # Пропускаем начальный размер, который уже проверили
        print(f"\nПроверяем {neurons} нейронов...")
        model, loss = train_and_evaluate(neurons, train_X, train_y, val_X, val_y, 
                                        epochs, lr, C11, C12, C22, direct_stress, direct_piola)
        
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
    S[:, 2] = (1.0 / C11) * (math.sqrt(C22 * C11) - C12) * y_r[:, 2]  # S12
    
    return S


def prepare_piola_stress_data(X, y_r, C11=1.0, C12=0.5, C22=1.0):
    """
    Подготавливает данные для обучения модели по напряжениям Пиолы-Кирхгофа.
    
    Args:
        X (torch.Tensor): Входные данные (xi)
        y_r (torch.Tensor): Целевые значения r
        C11, C12, C22: Параметры для вычисления напряжений
        
    Returns:
        torch.Tensor: Целевые значения P (напряжения Пиолы-Кирхгофа)
    """
    # Сначала вычисляем напряжения S
    S = prepare_stress_data(X, y_r, C11, C12, C22)
    
    # Создаем временную модель для использования функции compute_piola_stress
    temp_model = CLANN(input_dim=3, hidden_dim=1, output_dim=3, 
                      C11=C11, C12=C12, C22=C22, direct_stress=False)
    
    # Вычисляем напряжения Пиолы-Кирхгофа P = S * F^(-T)
    P = temp_model.compute_piola_stress(S, X)
    
    return P


def load_square_ksies_data(filepath="dd_tables/square_ksies.csv"):
    """
    Загружает данные из файла square_ksies.csv и подготавливает их для обучения.
    
    Args:
        filepath (str): Путь к файлу с данными
        
    Returns:
        tuple: (X, y_r, y_P, F) где:
            X - входные данные xi размера [N, 3]
            y_r - функции отклика r размера [N, 3] 
            y_P - напряжения Пиолы-Кирхгофа P размера [N, 3]
            F - градиент деформации размера [N, 3] = [F11, F12, F22]
    """
    print(f"Загрузка данных из {filepath}...")
    df = pd.read_csv(filepath)
    
    # Используем готовые xi из файла
    X = torch.tensor(df[["xi_1", "xi_2", "xi_3"]].values, dtype=torch.float32)
    
    # Извлекаем градиент деформации F (предполагаем F21 = 0)
    F_xx = torch.tensor(df["F_xx"].values, dtype=torch.float32)
    F_yy = torch.tensor(df["F_yy"].values, dtype=torch.float32)
    F_xy = torch.tensor(df["F_xy"].values, dtype=torch.float32)
    
    F = torch.stack([F_xx, F_xy, F_yy], dim=1)  # [F11, F12, F22]
    
    # Извлекаем напряжения Пиолы-Кирхгофа (предполагаем, что PX=P11, PY=P22, P12=0)
    P11 = torch.tensor(df["PX"].values, dtype=torch.float32)
    P22 = torch.tensor(df["PY"].values, dtype=torch.float32)
    P12 = torch.zeros_like(P11)  # Предполагаем, что P12 = 0 для данной задачи
    
    y_P = torch.stack([P11, P22, P12], dim=1)
    
    # Вычисляем напряжения Коши S = P * F^T
    batch_size = X.shape[0]
    S = torch.zeros(batch_size, 3, dtype=torch.float32)
    
    # Создаем матрицы F^T для каждого образца
    F_T = torch.zeros(batch_size, 2, 2, dtype=torch.float32)
    F_T[:, 0, 0] = F_xx  # F11
    F_T[:, 0, 1] = torch.zeros_like(F_xx)  # F21 = 0
    F_T[:, 1, 0] = F_xy  # F12
    F_T[:, 1, 1] = F_yy  # F22
    
    # Создаем матрицы P для каждого образца
    P_mat = torch.zeros(batch_size, 2, 2, dtype=torch.float32)
    P_mat[:, 0, 0] = P11  # P11
    P_mat[:, 1, 1] = P22  # P22
    P_mat[:, 0, 1] = P12  # P12
    P_mat[:, 1, 0] = P12  # P21 = P12
    
    # Вычисляем S = P * F^T
    S_mat = torch.bmm(P_mat, F_T)
    
    # Извлекаем компоненты S
    S[:, 0] = S_mat[:, 0, 0]  # S11
    S[:, 1] = S_mat[:, 1, 1]  # S22
    S[:, 2] = S_mat[:, 0, 1]  # S12
    
    # Вычисляем функции отклика r из S
    # Используем фиксированные параметры материала
    C11_param = 1.0
    C12_param = 0.5
    C22_param = 1.0
    
    y_r = torch.zeros_like(S)
    y_r[:, 0] = S[:, 0]  # r1 = S11
    y_r[:, 1] = S[:, 1]  # r2 = S22
    y_r[:, 2] = S[:, 2] * C11_param / (math.sqrt(C22_param * C11_param) - C12_param)  # r3
    
    print(f"Загружено {X.shape[0]} образцов")
    print(f"Диапазон xi: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Диапазон P: [{y_P.min():.4f}, {y_P.max():.4f}]")
    print(f"Диапазон r: [{y_r.min():.4f}, {y_r.max():.4f}]")
    print(f"Диапазон F: [{F.min():.4f}, {F.max():.4f}]")
    
    return X, y_r, y_P, F


def main():
    # Загрузка данных из нового файла
    print("Загрузка данных из square_ksies.csv...")
    X, y_r, y_P, F = load_square_ksies_data("dd_tables/square_ksies.csv")
    
    # Параметры для вычисления напряжений (фиксированные)
    C11 = 1.0
    C12 = 0.5
    C22 = 1.0
    
    # Выбираем режим работы CLANN
    mode = 'pk2_with_xi'  # Используем режим с готовыми xi и F
    
    print(f"Режим работы CLANN: {mode}")
    print("Модель будет обучаться предсказывать напряжения Пиолы-Кирхгофа P")
    
    # Подготавливаем данные для обучения
    y = y_P  # Целевые данные - напряжения P
    
    # Разбиваем данные на обучающую и валидационную выборки
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)
    train_F, val_F, _, _ = train_test_split(F, y, test_size=0.2, random_state=42)
    
    print(f"Размер обучающей выборки: {train_X.shape[0]}, размер валидационной выборки: {val_X.shape[0]}")
    
    # Создаем модель
    model = CLANN(input_dim=3, hidden_dim=64, output_dim=3, mode=mode, C11=C11, C12=C12, C22=C22)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Обучение модели
    epochs = 1000
    print(f"\nНачинаем обучение на {epochs} эпох...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass в зависимости от режима
        if mode == 'base':
            outputs = model(train_X)
        elif mode == 'pk2':
            outputs = model(train_F)
        elif mode == 'pk2_with_xi':
            outputs = model(train_X, train_F)
        
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        
        # Вывод прогресса обучения
        if (epoch + 1) % 100 == 0:
            print(f"Эпоха {epoch+1}/{epochs}, Потери: {loss.item():.6f}")
    
    # Оценка на валидационной выборке
    model.eval()
    with torch.no_grad():
        if mode == 'base':
            val_outputs = model(val_X)
        elif mode == 'pk2':
            val_outputs = model(val_F)
        elif mode == 'pk2_with_xi':
            val_outputs = model(val_X, val_F)
        
        val_loss = criterion(val_outputs, val_y).item()
        print(f"\nОшибка на валидационной выборке: {val_loss:.6f}")
    
    # Оценка точности предсказания модели
    print("\n" + "="*50)
    print("Оценка точности предсказания модели")
    print("="*50)
    
    # Предсказания на всех данных
    model.eval()
    with torch.no_grad():
        if mode == 'base':
            P_pred = model(X)
            r_pred = model.predict_r(X)
            S_pred = model.predict_stress(X)
        elif mode == 'pk2':
            P_pred = model(F)
            # Для режима pk2 нужно вычислить xi из F для получения r и S
            xi_from_F = model.compute_xi_from_F(F)
            r_pred = model.predict_r(xi_from_F)
            S_pred = model.predict_stress(xi_from_F)
        elif mode == 'pk2_with_xi':
            P_pred = model(X, F)
            r_pred = model.predict_r(X)
            S_pred = model.predict_stress(X)
    
    # Вычисляем ошибки
    mse_P = nn.MSELoss()(P_pred, y_P).item()
    mse_r = nn.MSELoss()(r_pred, y_r).item()
    
    # Вычисляем истинные S из r для сравнения
    S_true = model.compute_stress_from_r(y_r)
    mse_S = nn.MSELoss()(S_pred, S_true).item()
    
    print(f"MSE для P: {mse_P:.6f}")
    print(f"MSE для r: {mse_r:.6f}")
    print(f"MSE для S: {mse_S:.6f}")
    
    # Вычисляем относительные ошибки
    rel_error_P11 = torch.mean(torch.abs((P_pred[:, 0] - y_P[:, 0]) / (y_P[:, 0] + 1e-8))).item()
    rel_error_P22 = torch.mean(torch.abs((P_pred[:, 1] - y_P[:, 1]) / (y_P[:, 1] + 1e-8))).item()
    rel_error_P12 = torch.mean(torch.abs((P_pred[:, 2] - y_P[:, 2]) / (y_P[:, 2] + 1e-8))).item()
    
    print("\nОтносительные ошибки для P:")
    print(f"P11: {rel_error_P11:.6f}")
    print(f"P22: {rel_error_P22:.6f}")
    print(f"P12: {rel_error_P12:.6f}")
    
    # Экспортируем модель в формат ONNX
    print("\nЭкспорт модели в ONNX формат...")
    model.eval()
    
    if mode == 'base':
        dummy_input = torch.randn(1, 3)
        torch.onnx.export(
            model, dummy_input, f"clann_{mode}_64.onnx",
            input_names=["xi"], output_names=["r"],
            opset_version=11
        )
    elif mode == 'pk2':
        dummy_input = torch.randn(1, 3)
        torch.onnx.export(
            model, dummy_input, f"clann_{mode}_64.onnx",
            input_names=["F"], output_names=["P"],
            opset_version=11
        )
    elif mode == 'pk2_with_xi':
        # Для режима с двумя входами создаем wrapper
        class CLANNWrapper(nn.Module):
            def __init__(self, model):
                super(CLANNWrapper, self).__init__()
                self.model = model
                
            def forward(self, xi, F):
                return self.model(xi, F)
        
        wrapper = CLANNWrapper(model)
        dummy_xi = torch.randn(1, 3)
        dummy_F = torch.randn(1, 3)
        torch.onnx.export(
            wrapper, (dummy_xi, dummy_F), f"clann_{mode}_64.onnx",
            input_names=["xi", "F"], output_names=["P"],
            opset_version=11
        )
    
    print(f"Модель экспортирована в файл clann_{mode}_64.onnx")
    print(f"Обучение завершено! Финальная ошибка валидации: {val_loss:.6f}")


if __name__ == "__main__":
    main() 