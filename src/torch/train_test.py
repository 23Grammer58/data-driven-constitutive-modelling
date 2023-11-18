import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dataload import ExcelDataset
from model import SingleLayerPerceptron
from torchmetrics.regression import MeanSquaredError

import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Selected device: {device}")

# Гиперпараметры
input_size = 2  # Размерность входных данных
output_size = 1  # Размерность выходных данных
hidden_size = 270  # Новое количество нейронов на слое
learning_rate = 0.01
epochs = 5


def normalize_data(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


def load_data():
    data = ExcelDataset()
    print("data len =", len(data))
    labels = normalize_data(data.target)

    # Нормализация входных данных
    normalized_data = normalize_data(data.features)

    # Создание DataLoader
    dataset = TensorDataset(normalized_data, labels)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(test_dataset)
    print("размер тестовой выборки", len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    return train_dataloader, test_dataloader


def train(train_dataloader, experiment_name="model"):
    # Инициализация модели, функции потерь и оптимизатора
    model = SingleLayerPerceptron()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Обучение модели
    for epoch in range(epochs):
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.5f}')
        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    if experiment_name is not None:
        path_to_save_weights = os.path.join("pretrained_models", experiment_name + ".pth")
        torch.save(model.state_dict(), path_to_save_weights)
        print(f"Saved PyTorch Model State to {experiment_name}.pth")

    return model


def test(model, test_dataloader):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            all_predictions.append(predictions)
            all_targets.append(targets)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    criterion = nn.MSELoss()
    mse = criterion(all_predictions, all_targets)
    print("Mean Squared Error on Test Data:", mse.item())


if __name__ == "__main__":
    print("loading data...")
    train_dataloader, test_dataloader = load_data()
    model = train(train_dataloader)
    print("test data...")
    test(model, test_dataloader)
