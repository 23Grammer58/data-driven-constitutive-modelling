import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm

# Гиперпараметры
input_size = 2  # Размерность входных данных
output_size = 2  # Размерность выходных данных
learning_rate = 0.001
epochs = 100

# Создание однослойного перцептрона
class SingleLayerPerceptron(nn.Module):
    def __init__(self):
        super(SingleLayerPerceptron, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x

# Нормализация данных
def normalize_data(data):
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    return (data - mean) / std

# Генерация фиктивных данных
# Замени этот блок на загрузку своих данных
data = torch.randn(100, input_size)
labels = torch.randn(100, output_size)

# Нормализация входных данных
normalized_data = normalize_data(data)

# Создание DataLoader
dataset = TensorDataset(normalized_data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Инициализация модели, функции потерь и оптимизатора
model = SingleLayerPerceptron()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Пример использования обученной модели для инференса
test_data = torch.randn(10, input_size)  # Замени это на свои тестовые данные
normalized_test_data = normalize_data(test_data)
with torch.no_grad():
    model.eval()
    predictions = model(normalized_test_data)
    model.train()

print("Predictions:", predictions)
