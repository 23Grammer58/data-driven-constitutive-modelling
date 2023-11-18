import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from dataload import ExcelDataset
from torchmetrics.regression import MeanSquaredError

import os

# Гиперпараметры
input_size = 2  # Размерность входных данных
output_size = 1  # Размерность выходных данных
hidden_size = 270  # Новое количество нейронов на слое
learning_rate = 0.001
epochs = 100

# Создание однослойного перцептрона
class SingleLayerPerceptron(nn.Module):
    def __init__(self):
        super(SingleLayerPerceptron, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


# Нормализация данных
# def normalize_data(data):
#     mean = data.mean()
#     std = data.std()
#     return (data - mean) / std
#
# # file_path = "../../data"
# # excel_files = os.listdir(file_path)
# # excel_files = [os.path.join(file_path, file) for file in excel_files]
# # data = ExcelDataset("one_data_names.txt")
# data = ExcelDataset()
# print("data len =", len(data))
# labels = normalize_data(data.target)
#
# # Нормализация входных данных
# normalized_data = normalize_data(data.features)
#
# # Создание DataLoader
# dataset = TensorDataset(normalized_data, labels)
#
# train_size = int(0.9 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# # Инициализация модели, функции потерь и оптимизатора
# model = SingleLayerPerceptron()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # Обучение модели
# for epoch in range(epochs):
#     for inputs, targets in train_dataloader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
#
# # Пример использования обученной модели для инференса
# # test_data = torch.randn(10, input_size)  # Замени это на свои тестовые данные
# # normalized_test_data = normalize_data(test_dataset)
# mean_squared_error = MeanSquaredError()
# # mean_squared_error(preds, target)
# print(test_dataset)
# print("размер тестовой выборки", len(test_dataset))
#
# model.eval()
# all_predictions = []
# all_targets = []
#
# with torch.no_grad():
#     for inputs, targets in test_dataloader:
#         predictions = model(inputs)
#         all_predictions.append(predictions)
#         all_targets.append(targets)
#
# all_predictions = torch.cat(all_predictions, dim=0)
# all_targets = torch.cat(all_targets, dim=0)
#
# mse = criterion(all_predictions, all_targets)
# print("Mean Squared Error on Test Data:", mse.item())
#
# # if __name__ == "__main__":
