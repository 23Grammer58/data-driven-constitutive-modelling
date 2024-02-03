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

# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Selected device: {device}")
device = "cpu"
# Гиперпараметры
input_size = 2  # Размерность входных данных
output_size = 1  # Размерность выходных данных
hidden_size = 270  # Новое количество нейронов на слое
learning_rate = 0.001
EPOCHS = 100

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))


def normalize_data(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


def load_data(path_to_exp_names, batch_size):
    dataset = ExcelDataset(path=path_to_exp_names, transform=normalize_data)
    # print("data len =", len(dataset))
    # Нормализация входных данных
    # dataset.features = normalize_data(dataset.features)
    # dataset.target = normalize_data(dataset.target)
    # labels = data.target
    # normalized_data = data.features

    # Создание DataLoader
    # dataset_t = TensorDataset(normalized_fetures, labels)
    dataset_t = TensorDataset(dataset.features, dataset.target)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # train_size = int(0.9 * len(dataset_t))
    # test_size = len(dataset_t) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset_t, [train_size, test_size])

    # print(test_dataset)
    # print("размер тестовой выборки", len(test_dataset))

    dataset_loader = DataLoader(dataset_t, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return dataset_loader
    # return train_dataloader, test_dataloader


def train(train_loader, test_loader, experiment_name, plot_loss=False):

    batch_size = train_loader.batch_size
    # Инициализация модели, функции потерь и оптимизатора
    model = StrainEnergyCANN(batch_size, device=device).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train_one_epoch(epoch_index, tb_writer):

        losses = []
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(train_loader):
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
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = 1 * len(train_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    epoch_number = 0
    best_vloss = 1_000_000.
    losses = []

    # Обучение модели
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and avg
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'valid': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        losses.append(best_vloss)
        epoch_number += 1

        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

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
            predictions = model(inputs)
            # except:
            #     predictions = model(inputs.transpose())
            #     print(model.weights)
            all_predictions.append(predictions)
            all_targets.append(targets)
            err.append(((targets - predictions) ** 2).values())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    criterion = nn.MSELoss()
    mse = criterion(all_predictions, all_targets)
    print("Mean Squared Error on Test Data:", mse.item())

    if plot_err:
        try:
            plt.plot(err)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.show()
            plt.savefig("Errs")
        except:
             print("can't plot")


def jit(model,  experiment_name, x=None):
    # traced_net = torch.jit.trace(model, x)
    traced_net = torch.jit.script(model)
    directory = "pretrained_models/"

    torch.jit.save(traced_net, directory + experiment_name + '.pt')


# def get_potential(model):


if __name__ == "__main__":
    torch.manual_seed(42)

    experiment = "CNN3"

    print("loading data...")
    # train_dataloader, test_dataloader = load_data("full_data_names.txt", batch_size=1)
    train_dataloader = load_data("without_one.txt", batch_size=1)
    test_dataloader = load_data("one_data_names.txt", batch_size=1)


    trained_model = train(
                        train_dataloader,
                        test_dataloader,
                        plot_loss=True,
                        experiment_name=experiment)

    print("test data...")
    # trained_model = StrainEnergyCANN(batch_size=1, device=device)
    # trained_model.load_state_dict(torch.load('pretrained_models/experimental.pth'))
    # test(trained_model, test_dataloader, plot_err=True)

    weights = trained_model.get_potential()
    print(weights)
    # for param in trained_model.parameters():
    #     print(param, type(param), param.size())

    


    # x1 = torch.randn(1).to(device)
    # x2 = torch.randn(1).to(device)
    # # print(x1)
    jit(trained_model,  experiment_name=experiment)

"""
TODO:
1) поработать с временем загрузки данных
2) обернуть весь пайплайн в класс 
3) обучать с нормализацией и без
"""