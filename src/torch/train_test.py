import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from dataload import ExcelDataset
from src.torch.models.CNN import StrainEnergyCANN

import os
import matplotlib.pyplot as plt

from models.CNN import StrainEnergyCANN

print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Selected device: {device}")

# Гиперпараметры
input_size = 2  # Размерность входных данных
output_size = 1  # Размерность выходных данных
hidden_size = 270  # Новое количество нейронов на слое
learning_rate = 0.1
epochs = 1

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

def normalize_data(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


def load_data(path_to_exp_names, batch_size):
    dataset = ExcelDataset(path=path_to_exp_names, transform=None)
    print("data len =", len(dataset))
    # Нормализация входных данных
    dataset.features = normalize_data(dataset.features)
    dataset.target = normalize_data(dataset.target)
    # labels = data.target
    # normalized_data = data.features

    # Создание DataLoader
    # dataset = TensorDataset(normalized_fetures, labels)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # print(test_dataset)
    print("размер тестовой выборки", len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return train_dataloader, test_dataloader


def train(dataloader, experiment_name, plot_loss=False):
    batch_size = dataloader.batch_size
    # Инициализация модели, функции потерь и оптимизатора
    model = StrainEnergyCANN(batch_size, device=device).to(device)
    # model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Обучение модели
    losses = []
    it = 0
    running_loss = 0.

    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            # if inputs.shape != (batch_size, 2):
            #     it += 1
            #     print(it)
            #     continue
            optimizer.zero_grad()
            # i1_inputs=inputs[:, :1]
            # i2_inputs = inputs[:, 1:]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = 1 * len(dataloader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.


            # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.5f}')
            # if train_dataloader == [inputs, targets]:
            #     losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
            losses.append(loss.item)

    print(it)
    if experiment_name is not None:
        path_to_save_weights = os.path.join("pretrained_models", experiment_name + ".pth")
        torch.save(model.state_dict(), path_to_save_weights)
        print(f"Saved PyTorch Model State to {experiment_name}.pth")

    if plot_loss:
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
    return model


def test(model, dataloader, plot_err=False):
    batch_size = dataloader.batch_size

    model.eval()
    all_predictions = []
    all_targets = []

    inputss = []
    it = 0
    err = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs, targets = inputs.to(device).transpose(), targets.to(device).transpose()
            # if inputs.shape != (batch_size, 2) or targets.shape != (1, 2):
            #     it += 1
            #     continue
            # i1_inputs=inputs[:, :1]
            # i2_inputs = inputs[:, 1:]

            # try:
            predictions = model(inputs).to(device)
            # except:
            #     predictions = model(inputs.transpose())
            #     print(model.weights)
            all_predictions.append(predictions)
            all_targets.append(targets)
            err.append((targets - predictions) ** 2)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    criterion = nn.MSELoss()
    mse = criterion(all_predictions, all_targets)
    print("Mean Squared Error on Test Data:", mse.item())

def jit(model, x):
    traced_net = torch.jit.trace(model, x)
    if plot_err:
        try:
            plt.plot(err.cpu())
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.show()
            plt.savefig("Errs")
        except:
             print("can't plot")


def jit(model, x, experiment_name):
    # traced_net = torch.jit.trace(model, x)
    directory = "pretrained_models/"

    torch.jit.save(traced_net, directory + 'MLP.pt')


if __name__ == "__main__":

    experiment = "CNN2"
    print("loading data...")
    # train_dataloader, test_dataloader = load_data("full_data_names.txt", batch_size=1)
    # print(train_dataloader.dataset)
    # trained_model = train(train_dataloader, plot_loss=True, experiment_name="experimental")
    print("test data...")
    trained_model = StrainEnergyCANN(batch_size=1, device=device)
    trained_model.load_state_dict(torch.load('pretrained_models/experimental.pth'))
    # test(trained_model, test_dataloader, plot_err=True)
    # print(trained_model.parameters())
    for param in trained_model.parameters():
        print(param, type(param), param.size())

    


    # x1 = torch.randn(1).to(device)
    # x2 = torch.randn(1).to(device)
    # # print(x1)
    # jit(trained_model, (x1, x2), experiment_name=experiment)

"""
TODO:
1) поработать с временем загрузки данных
2) обернуть весь пайплайн в класс 
"""