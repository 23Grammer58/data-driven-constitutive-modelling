from math import exp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils.dataload import ExcelDataset

import os
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from models.CNN import StrainEnergyCANN, StrainEnergyCANN_C

from tqdm import tqdm

# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# print(f"Selected device: {device}")
device = "cpu"

# Гиперпараметры
input_size = 2  # Размерность входных данных
output_size = 1  # Размерность выходных данных
hidden_size = 270  # Новое количество нейронов на слое
learning_rate = 0.1
EPOCHS = 1000
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))


def normalize_data(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


def load_data(path_to_exp_names, batch_size, transform=normalize_data, device=device):
    dataset = ExcelDataset(path=path_to_exp_names, transform=transform, device=device)
    # print("data len =", len(dataset))
    # Нормализация входных данных
    # dataset.features = normalize_data(dataset.features)
    # dataset.target = normalize_data(dataset.target)
    # labels = data.target
    # normalized_data = data.features

    # Создание DataLoader
    # dataset_t = TensorDataset(normalized_fetures, labels)

    # dataset_t = TensorDataset(dataset.features, dataset.target)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # train_size = int(0.9 * len(dataset_t))
    # test_size = len(dataset_t) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset_t, [train_size, test_size])

    # print(test_dataset)
    # print("размер тестовой выборки", len(test_dataset))

    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return dataset_loader
    # return train_dataloader, test_dataloader


def train(train_loader, test_loader, experiment_name, plot_loss=False):
    """
    Train the model on the given dataset.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the validation dataset.
        experiment_name (str): Name of the experiment.
        plot_loss (bool, optional): Whether to plot the training and validation loss. Defaults to False.

    Returns:
        model (nn.Module): The trained model.
    """
    batch_size = train_loader.batch_size

    # Инициализация модели, функции потерь и оптимизатор
    # а
    model = StrainEnergyCANN_C(batch_size, device=device).to(device)

    if experiment_name is not None:
        path_to_save_weights = os.path.join("pretrained_models", experiment_name)
        if not os.path.exists(path_to_save_weights):
            os.makedirs(path_to_save_weights)
            print(f"Директория {path_to_save_weights} успешно создана")
        else:
            print(f"Директория {path_to_save_weights} уже существует")

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train_one_epoch(epoch_index, tb_writer, l2_reg_coeff=0.01):

        running_loss = 0.
        last_loss = 0.

        last_data = len(train_loader)
        for i, data in enumerate(train_loader):

            lam, C, invariants, targets = data

            C = C.reshape(-1, 3)
            inputs = (lam, C, invariants)
            targets = targets.reshape(-1, 3)

            # inputs, targets = inputs.to(device), targets.to(device)
            # if inputs.shape != (batch_size, 2):
            #     it += 1
            #     print(it)
            #     continue
            optimizer.zero_grad()
            # i1_inputs=inputs[:, :1]
            # i2_inputs = inputs[:, 1:]
            stress_model = model(inputs)

            loss = loss_fn(stress_model, targets)

            # coefs = model.get_potential()

            # l2_reg = None
            # for param in coefs:
            #     # print(param)
            #     if l2_reg is None:
            #         l2_reg = param ** 2
            #     else:
            #         l2_reg = l2_reg + param ** 2
            #     # print("l2 reg = ", l2_reg)
            #

            l2_reg = model.calc_regularization()
            # # Добавляем L2 регуляризацию к функции потерь
            if l2_reg is not None:
                loss = loss + l2_reg_coeff * l2_reg

            loss.backward()
            optimizer.step()

            # print('experiment {} loss: {}'.format(i + 1, loss.item()))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', loss.item(), tb_x)
            last_loss = loss.item()
            running_loss += loss.item()
            # if i % last_data == last_data - 1:
            #     last_loss = running_loss / last_data  # loss per experiment
            #     print('experiment {} loss: {}'.format(i + 1, last_loss))
            #     tb_x = epoch_index * len(train_loader) + i + 1
            #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            #     running_loss = 0.
            # if i == (last_data-1):
            #     print(last_data)
        return running_loss / last_data

    epoch_number = 0
    best_vloss = torch.inf
    elosses = []
    vlosses = []

    # Обучение модели
    for epoch in tqdm(range(EPOCHS)):
        # print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        #
        # # Disable gradient computation and reduce memory consumption.
        # with torch.no_grad():
        #     for i, vdata in enumerate(test_loader):
        #         vinputs, vlabels = vdata
        #         vinputs = vinputs.reshape(-1, 3)
        #         vlabels = vlabels.reshape(-1, 3)
        #
        #         voutputs = model(vinputs).to(device)
        #         vloss = loss_fn(voutputs, vlabels)
        #         running_vloss += vloss
        #         vlosses.append(vloss)
        #
        # avg_vloss = running_vloss / (i + 1)
        # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        #
        # # Log the running loss averaged per batch
        # # for both training and avg
        # writer.add_scalars('Training vs. Validation Loss',
        #                    {'Training': avg_loss, 'valid': avg_vloss},
        #                    epoch_number + 1)
        # writer.flush()
        #
        # # Track best performance, and save the model's state
        model_path = '{}_{}'.format(timestamp, epoch_number)
        path_to_save_weights = os.path.join("pretrained_models", experiment_name)
        path_to_save_weights = os.path.join(path_to_save_weights, model_path + ".pth")

        if avg_loss < best_vloss:
            best_vloss = avg_loss
            print(f"Saved PyTorch Model State to {path_to_save_weights}")
            torch.save(model.state_dict(), model_path)

        elif epoch % 10 == 0:
            torch.save(model.state_dict(), path_to_save_weights)
            print(f"Saved PyTorch Model State to {path_to_save_weights}")

        elosses.append(avg_loss)
        epoch_number += 1
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss.item():.4f}')

    plt.plot(vlosses)
    plt.show()
    if plot_loss:
        plt.plot(elosses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
    return model


def test(model, dataloader, plot_err=False):
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


def jit(model, experiment_name, x=None):
    # traced_net = torch.jit.trace(model, x)
    traced_net = torch.jit.script(model)
    directory = "pretrained_models/"

    torch.jit.save(traced_net, directory + experiment_name + '.pt')


def get_potential_formula_2term(model):
    raw_params = model.get_potential()
    mid = len(raw_params) // 2
    first_half = raw_params[:mid]
    second_half = raw_params[mid:]
    coefficients = [first_half[i] * second_half[i] for i in range(mid)]

    # Инициализация символов SymPy
    I1, I2 = sp.symbols('I1 I2')

    # Проверка на количество коэффициентов
    # if len(coefficients) < 4:
    #     raise ValueError("Для формулы требуется как минимум 4 коэффициента")

    # Подставляем коэффициенты в формулу
    psi = (coefficients[0] * (I1 - 3) + coefficients[2] * (I2 - 3) +
           coefficients[1] * (I1 - 3) ** 2 + coefficients[3] * (I2 - 3) ** 2)

    # Вывод формулы
    sp.pprint(psi)


def get_potential_formula_4term(model):
    raw_params = model.get_potential()
    # print(raw_params)
    mid = len(raw_params) // 2
    first_half = raw_params[:mid]
    second_half = raw_params[mid:]
    coefficients = [first_half[i] * second_half[i] for i in range(mid)]

    # Инициализация символов SymPy
    I1, I2 = sp.symbols('I1 I2')

    # Проверка на количество коэффициентов
    # if len(coefficients) < 4:
    #     raise ValueError("Для формулы требуется как минимум 4 коэффициента")

    # Подставляем коэффициенты в формулу
    psi = (coefficients[0] * (I1 - 3)
           + coefficients[2] * (I2 - 3)
           + coefficients[1] * (I1 - 3) ** 2
           + coefficients[3] * (I2 - 3) ** 2
           + coefficients[4] * sp.exp(I1 - 3)
           + coefficients[6] * sp.exp(I2 - 3)
           + coefficients[5] * sp.exp(I1 - 3) ** 2
           + coefficients[7] * sp.exp(I2 - 3) ** 2)

    # Вывод формулы
    sp.pprint(psi)


def get_potential_formula_6term(model):
    coefficients = model.get_potential()

    # Инициализация символов SymPy
    I1, I2 = sp.symbols('I1 I2')

    # Подставляем коэффициенты в формулу
    psi = (  coefficients[0] * (I1 - 3)
           + coefficients[2] * (I2 - 3)
           + coefficients[1] * (I1 - 3) ** 2
           + coefficients[3] * (I2 - 3) ** 2
           + coefficients[4] * sp.exp(I1 - 3)
           + coefficients[6] * sp.exp(I2 - 3)
           + coefficients[5] * sp.exp(I1 - 3) ** 2
           + coefficients[7] * sp.exp(I2 - 3) ** 2
           + coefficients[9] * sp.exp(I1 - 3) ** 2
           + coefficients[11] * sp.exp(I2 - 3) ** 2

    )

    # Вывод формулы
    sp.pprint(psi)


if __name__ == "__main__":
    torch.manual_seed(42)

    experiment = "CNN_brain_6term_C_cuda/"
    # experiment = None
    # print("loading data...")
    #
    # train_dataloader, test_dataloader = load_data(
    #     "full_data_names.txt",
    #     batch_size=1)
    data_path = r"C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\data\brain_bade\CANNsBRAINdata.xlsx"

    # train_dataloader= load_data(
    #     "one_data_names.txt", batch_size=1)
    # test_dataloader = load_data(
    #     "another_one_data_name.txt", batch_size=1)

    train_dataloader = load_data(
        data_path,
        batch_size=1,
        transform=None,
        device=device
    )

    trained_model = train(
        train_dataloader,
        train_dataloader,
        plot_loss=True,
        experiment_name=experiment)

    print("test data...")
    # trained_model = StrainEnergyCANN_C(batch_size=1, device=device)
    potential_files = os.listdir("pretrained_models/" + experiment)
    for epoch_number, file in enumerate(potential_files):
        # trained_model.load_state_dict(torch.load('pretrained_models/CNN_MR_2term_2/20240326_210215_' + str(epoch_number) + ".pth"))
        trained_model.load_state_dict(torch.load("pretrained_models/" + experiment + file))
        # test(trained_model, test_dataloader, plot_err=True)
        print(f"Epoch {epoch_number}: {trained_model.get_potential()}")
        print("-------------------------------------------------------------------------------------------------------")
        # get_potential_formula(trained_model)
        print(trained_model.get_potential())
        for id, param in enumerate(trained_model.parameters()):
            print(f"num param = {id}, {param}")
    print(trained_model)

    # trained_model.load_state_dict(
    #     torch.load('pretrained_models/CNN_MR_full_2term_l2/20240313_131752_7.pth'))

    # # Разделение параметров и вычисление коэффициентов
    # raw_params = trained_model.get_potential()
    # mid = len(raw_params) // 2
    # first_half = raw_params[:mid]
    # second_half = raw_params[mid:]
    # coefficients = [first_half[i] * second_half[i] for i in range(mid)]

    # Инициализация символов SymPy
    # I1, I2 = sp.symbols('I1 I2')
    #
    # # Проверка на количество коэффициентов
    # if len(coefficients) < 4:
    #     raise ValueError("Для формулы требуется как минимум 4 коэффициента")
    #
    # # Подставляем коэффициенты в формулу
    # psi = (coefficients[0] * (I1 - 3) + coefficients[2] * (I2 - 3) +
    #        coefficients[1] * (I1 - 3) ** 2 + coefficients[3] * (I2 - 3) ** 2)
    #
    # # Вывод формулы
    # sp.pprint(psi)
    #
    # psi_evaluated = psi.subs({I1: 4, I2: 4}).evalf()
    # print(psi_evaluated)
    # x1 = torch.randn(1).to(device)
    # x2 = torch.randn(1).to(device)
    # # print(x1)
    # jit(trained_model, experiment_name=experiment)

"""
TODO:
1) поработать с временем загрузки данных
2) обернуть весь пайплайн в класс 
3) обучать с нормализацией и без
"""
