import numpy as np
from scipy.signal import argrelextrema

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

import itertools
import copy
import pathlib
from datetime import datetime
import os
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.metrics import r2_score
# from models.CNN import *
from CANN_torch.models import ModelArchitecture_I2
from CANN_torch.utils import _filter_data_by_protocol
# from models.CNN import StrainEnergyCANN, StrainEnergyCANN_C, StrainEnergyCANN_polinomial3
import seaborn as sns
import pandas as pd
from CANN_torch.core import Trainer


# hyperparameters and paths
path_to_data = r"../../../data/latex/16_03/"
experiment_mod = "biaxial"
batch_size = 16
# path_to_results = r"C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\src\CANN_torch\pretrained_models"


def get_list_of_paths_to_experiments_type(experiment="uniaxial"):
    l = []
    for file in os.listdir(path_to_data):
        l.append(os.path.join(path_to_data, file))
    return l


def load_and_extract(file_path, experiment_type):
    df = pd.read_csv(file_path, header=0)
    if experiment_type == "Unia":
        df['experiment_type'] = 1000
    else:
        df['experiment_type'] = df['cycle_number']
    return df[['Xlam', 'Ylam', 'stress_x_mpa', 'stress_y_mpa', 'experiment_type']]


I1_bx = lambda lam1, lam2: lam1 ** 2 + lam2 ** 2 + 1 / (lam1 * lam2) ** 2
I2_bx = lambda lam1, lam2: 1 / lam1 ** 2 + 1 / lam2 ** 2 + (lam1 * lam2) ** 2
F_bx = lambda lam1, lam2: ([lam1, 0, 0], [0, lam2, 0], [0, 0, 1 / (lam1 * lam2)])


# Механические переменные и их функции расчёта
mechanical_variables = {
    "I1": I1_bx,
    "I2": I2_bx,
    "F": F_bx
}


def preproc_data(experiment_protocols: list = None):
    experiments_path = get_list_of_paths_to_experiments_type()
    data_frames = [load_and_extract(file, file[-8:-4]) for file in experiments_path] # поменять для друго даты
    # print(data_frames[0][])

    # Объединяем все DataFrame в один
    data_frames = pd.concat(data_frames).reset_index(drop=True, inplace=False)

    if not experiment_protocols:
        experiment_protocols = data_frames['experiment_type'].unique()

    data_frames_thin = []
    # Итерируемся по каждому уникальному experiment_type
    for experiment_type in experiment_protocols:

        # Фильтруем данные по текущему experiment_type
        data_frame = data_frames[data_frames['experiment_type'] == experiment_type]

    combined_data = data_frames
    combined_data.dropna(inplace=True)

    combined_data.columns = ['lamx', 'lamy', 'Px', 'Py', 'experiment_type']

    return combined_data


class SimpleDataset(Dataset):
    """
        Самый простой датасет.
        Структура данных такова (x1, x2, y1, y2, ...), x - признак, y - целевое значение.
    """
    def __init__(self, dataframe):
        self.data = dataframe
        # self.features = [dataframe[0],dataframe[2], dataframe[3], dataframe[4], dataframe[5]]
        # self.targets  = dataframe[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = copy.deepcopy([*self.data.iloc[idx]])
        # for f in features:
        #     if type(f) is str:
        #         print(f)
        target1 = features.pop(2)
        target2 = features.pop(2)
        # features.pop(-1)
        target = torch.tensor((target1.item(), target2.item()))

        return features, target

    def to_tensor(self):
        for column in self.data.columns:
            # if column != "experiment_type":
            self.data[column] = self.data[column].apply(
                lambda x: torch.tensor(x, dtype=torch.float32)).copy()

    # def to_tensor(self):
    #     for column in self.data.columns:
    #         if column != "experiment_type":
    #             self.data[column] = self.data[column].apply(
    #                 lambda x: torch.tensor(x, dtype=torch.float32) if isinstance(x, (list, tuple)) else x
    #             )


def init_loaders(
        all_data: pd.DataFrame = None,
        experiment_protocol_train: Optional[str] or Optional[list] = None,
        experiment_protocol_test: Optional[str] or Optional[list] = None
):
    """
        Инициализация train_data_loader, test_data_loader.

        Parameters:
        - all_data: данные, из которых будут взяты подтипы и инициализируется test_data_loader,
        - experiment_protocol_train: подтипы, из которых инициализируется train_data_loader

        return train_data_loader, test_data_loader
    """

    # global experiment
    global batch_size

    if all_data is None:
        all_data = preproc_data(None)

    # Обработка обучающих данных
    train_dataframe = _filter_data_by_protocol(all_data, experiment_protocol_train)

    # Обработка тестовых данных
    test_dataframe = _filter_data_by_protocol(all_data, experiment_protocol_test)

    train_dataset = SimpleDataset(train_dataframe)
    test_dataset = SimpleDataset(test_dataframe)

    print(f"Original DataFrame size: {len(train_dataframe)}")
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check data splitting and preprocessing steps.")

    train_dataset.to_tensor()
    test_dataset.to_tensor()
    # f, t = train_dataset[10]
    # print(train_dataset[10])
    # print("f:", f)
    # print("t:", t)
    train_data_loader = DataLoader(
        train_dataset,
        shuffle=True,
        # num_workers=1,
        pin_memory=False,
        batch_size=batch_size
    )
    test_data_loader = DataLoader(
        test_dataset,
        shuffle=False,
        # num_workers=1,
        pin_memory=False
    )

    return train_data_loader, test_data_loader


def plot_results_by_experiment_type_(data: pd.DataFrame, path_to_save: str, plot_name_prefix: str = "plot"):
    """
     Visualize dataset and predictions.
    """

    data.columns = ['lambda_x', 'lambda_y', 'stress_x', 'stress_y', 'experiment_type', 'P11', 'P22']

    # Получим уникальные типы экспериментов
    experiment_types = data['experiment_type'].unique()
    r2s = dict.fromkeys(experiment_types)
    # r2s = pd.DataFrame.from_dict(r2s)

    for experiment_type in experiment_types:
        subset = data[data['experiment_type'] == experiment_type]

        # Создадим первый график (lambda_x, P11) и скаттер на нем stress_x для данного типа эксперимента
        fig, ax1 = plt.subplots(figsize=(10, 6))

        sns.lineplot(data=subset, x='lambda_x', y='P11', ax=ax1, label='P11')
        sns.scatterplot(data=subset, x='lambda_x', y='stress_x', ax=ax1, color='red', label='Stress_x')

        # Рассчитаем R² для P11
        r2_p11 = r2_score(subset['stress_x'], subset['P11'])

        ax1.set_title(f'{experiment_type}: P11 and Stress_x\nR² = {r2_p11:.2f}')
        ax1.set_xlabel('Lambda_x')
        ax1.set_ylabel('P11 / Stress_x (MPa)')
        plt.legend()
        if path_to_save:
            plt.savefig(os.path.join(path_to_save, f"{plot_name_prefix}_{experiment_type}_x.png"))
        # plt.show()

        # Создадим второй график (lambda_y, P22) и скаттер на нем stress_y для данного типа эксперимента
        fig, ax2 = plt.subplots(figsize=(10, 6))

        sns.lineplot(data=subset, x='lambda_y', y='P22', ax=ax2, label='P22')
        sns.scatterplot(data=subset, x='lambda_y', y='stress_y', ax=ax2, color='blue', label='Stress_y')

        # Рассчитаем R² для P22
        r2_p22 = r2_score(subset['stress_y'], subset['P22'])

        ax2.set_title(f'{experiment_type}: P22 and Stress_y\nR² = {r2_p22:.2f}')
        ax2.set_xlabel('Lambda_y')
        ax2.set_ylabel('P22 / Stress_y (MPa)')
        plt.legend()
        plt.savefig(os.path.join(path_to_save, f"{plot_name_prefix}_{experiment_type}_y.png"))
        r2s[experiment_type] = (r2_p11, r2_p22)

    plt.show()
    r2s = pd.DataFrame.from_dict(r2s, orient='index', columns=['PX', 'PY'])
    r2s.reset_index(inplace=True)
    r2s.rename(columns={'index': 'Category'}, inplace=True)
    # Вычисление среднего значения по всем значениям
    mean_values = r2s[['PX', 'PY']].mean()
    mean_row = pd.DataFrame([['Mean', mean_values['PX'], mean_values['PY']]], columns=r2s.columns)
    # Добавление строки со средними значениями в DataFrame
    r2s = pd.concat([r2s, mean_row], ignore_index=True)
    return r2s


def plot_results_by_experiment_type__(data: pd.DataFrame, path_to_save: str, plot_name_prefix: str = "plot"):
    """
    Visualize dataset and predictions.
    """

    data.columns = ['lambda_x', 'lambda_y', 'stress_x', 'stress_y', 'experiment_type', 'P11', 'P22']

    # Получим уникальные типы экспериментов
    experiment_types = data['experiment_type'].unique()
    r2s = dict.fromkeys(experiment_types)

    # Создаем списки для хранения данных для объединенных графиков
    combined_lambda_x_data = []
    combined_lambda_y_data = []

    for experiment_type in experiment_types:
        subset = data[data['experiment_type'] == experiment_type]

        # Создадим первый график (lambda_x, P11) и скаттер на нем stress_x для данного типа эксперимента
        fig, ax1 = plt.subplots(figsize=(10, 6))

        sns.lineplot(data=subset, x='lambda_x', y='P11', ax=ax1, label='P11')
        sns.scatterplot(data=subset, x='lambda_x', y='stress_x', ax=ax1, color='red', label='Stress_x')

        # Рассчитаем R² для P11
        r2_p11 = r2_score(subset['stress_x'], subset['P11'])

        ax1.set_title(f'{experiment_type}: P11 and Stress_x\nR² = {r2_p11:.2f}')
        ax1.set_xlabel('Lambda_x')
        ax1.set_ylabel('P11 / Stress_x (MPa)')
        plt.legend()
        if path_to_save:
            plt.savefig(os.path.join(path_to_save, f"{plot_name_prefix}_{experiment_type}_x.png"))
        plt.close(fig)  # Закрываем фигуру после сохранения

        # Сохраним данные для объединенного графика по x
        combined_lambda_x_data.append((subset['lambda_x'], subset['P11'], experiment_type))

        # Создадим второй график (lambda_y, P22) и скаттер на нем stress_y для данного типа эксперимента
        fig, ax2 = plt.subplots(figsize=(10, 6))

        sns.lineplot(data=subset, x='lambda_y', y='P22', ax=ax2, label='P22')
        sns.scatterplot(data=subset, x='lambda_y', y='stress_y', ax=ax2, color='blue', label='Stress_y')

        # Рассчитаем R² для P22
        r2_p22 = r2_score(subset['stress_y'], subset['P22'])

        ax2.set_title(f'{experiment_type}: P22 and Stress_y\nR² = {r2_p22:.2f}')
        ax2.set_xlabel('Lambda_y')
        ax2.set_ylabel('P22 / Stress_y (MPa)')
        plt.legend()
        if path_to_save:
            plt.savefig(os.path.join(path_to_save, f"{plot_name_prefix}_{experiment_type}_y.png"))
        plt.close(fig)  # Закрываем фигуру после сохранения

        # Сохраним данные для объединенного графика по y
        combined_lambda_y_data.append((subset['lambda_y'], subset['P22'], experiment_type))

        r2s[experiment_type] = (r2_p11, r2_p22)

    # Объединенный график по lambda_x
    fig, ax_combined_x = plt.subplots(figsize=(10, 6))
    for lambda_x, P11, experiment_type in combined_lambda_x_data:
        sns.lineplot(x=lambda_x, y=P11, ax=ax_combined_x, label=experiment_type)
    ax_combined_x.set_title('Combined P11 vs Lambda_x for All Experiment Types')
    ax_combined_x.set_xlabel('Lambda_x')
    ax_combined_x.set_ylabel('P11')
    plt.legend()
    if path_to_save:
        plt.savefig(os.path.join(path_to_save, f"{plot_name_prefix}_combined_x.png"))
    plt.close(fig)  # Закрываем фигуру после сохранения

    # Объединенный график по lambda_y
    fig, ax_combined_y = plt.subplots(figsize=(10, 6))
    for lambda_y, P22, experiment_type in combined_lambda_y_data:
        sns.lineplot(x=lambda_y, y=P22, ax=ax_combined_y, label=experiment_type)
    ax_combined_y.set_title('Combined P22 vs Lambda_y for All Experiment Types')
    ax_combined_y.set_xlabel('Lambda_y')
    ax_combined_y.set_ylabel('P22')
    plt.legend()
    if path_to_save:
        plt.savefig(os.path.join(path_to_save, f"{plot_name_prefix}_combined_y.png"))
        plt.close(fig)  # Закрываем фигуру после сохранения

    r2s = pd.DataFrame.from_dict(r2s, orient='index', columns=['PX', 'PY'])
    r2s.reset_index(inplace=True)
    r2s.rename(columns={'index': 'Category'}, inplace=True)

    # Вычисление среднего значения по всем значениям
    mean_values = r2s[['PX', 'PY']].mean()
    mean_row = pd.DataFrame([['Mean', mean_values['PX'], mean_values['PY']]], columns=r2s.columns)

    # Добавление строки со средними значениями в DataFrame
    r2s = pd.concat([r2s, mean_row], ignore_index=True)

    return r2s


def plot_results_by_experiment_type(data: pd.DataFrame, path_to_save: str, plot_name_prefix: str = "plot_test_all"):
    """
    Visualize dataset and predictions.
    """

    data.columns = ['lambda_x', 'lambda_y', 'stress_x', 'stress_y', 'experiment_type', 'P11', 'P22']

    # Получим уникальные типы экспериментов
    experiment_types = data['experiment_type'].unique()
    r2s = dict.fromkeys(experiment_types)

    # Создаем фигуры для общего графика по x и y
    plt.figure(figsize=(12, 6))

    # График для всех типов экспериментов по lambda_x
    for experiment_type in experiment_types:
        subset = data[data['experiment_type'] == experiment_type]

        # Линия P11
        sns.lineplot(data=subset, x='lambda_x', y='P11', label=f'P11 - {experiment_type}', ci=None)
        # Точки stress_x
        sns.scatterplot(data=subset, x='lambda_x', y='stress_x', color='red', label=f'Stress_x - {experiment_type}',
                        marker='o')

        # Рассчитаем R² для P11
        r2_p11 = r2_score(subset['stress_x'], subset['P11'])
        if r2_p11 < 0:
            r2_p11 = 0.
        r2s[experiment_type] = (r2_p11, None)  # Сохраняем R² для P11

    plt.title('P11 and Stress_x for all Experiment Types')
    plt.xlabel('Lambda_x')
    plt.ylabel('P11 / Stress_x (MPa)')
    plt.legend()
    if path_to_save:
        plt.savefig(os.path.join(path_to_save, f"{plot_name_prefix}_all_x.png"))
        plt.close()

    plt.show()

    # Создаем фигуру для общего графика по lambda_y
    plt.figure(figsize=(12, 6))

    for experiment_type in experiment_types:
        subset = data[data['experiment_type'] == experiment_type]

        # Линия P22
        # sns.lineplot(data=subset, x='lambda_y', y='P22', label=f'P22 - {experiment_type}', ci=None)
        sns.lineplot(data=subset, x='lambda_y', y='P22', label=f'P22 - {experiment_type}')
        # Точки stress_y
        sns.scatterplot(data=subset, x='lambda_y', y='stress_y', color='blue', label=f'Stress_y - {experiment_type}',
                        marker='o')

        # Рассчитаем R² для P22
        r2_p22 = r2_score(subset['stress_y'], subset['P22'])
        if r2_p22 < 0:
            r2_p22 = 0.
        r2s[experiment_type] = (r2s[experiment_type][0], r2_p22)  # Сохраняем R² для P22

    plt.title('P22 and Stress_y for all Experiment Types')
    plt.xlabel('Lambda_y')
    plt.ylabel('P22 / Stress_y (MPa)')
    plt.legend()
    if path_to_save:
        plt.savefig(os.path.join(path_to_save, f"{plot_name_prefix}_all_y.png"))

    plt.show()

    # Конвертация R² в DataFrame и добавление среднего значения
    r2s = pd.DataFrame.from_dict(r2s, orient='index', columns=['PX', 'PY'])
    r2s.reset_index(inplace=True)
    r2s.rename(columns={'index': 'Category'}, inplace=True)

    # Вычисление среднего значения по всем значениям
    mean_values = r2s[['PX', 'PY']].mean()
    mean_row = pd.DataFrame([['Mean', mean_values['PX'], mean_values['PY']]], columns=r2s.columns)

    # Добавление строки со средними значениями в DataFrame
    r2s = pd.concat([r2s, mean_row], ignore_index=True)

    return r2s

def main():

    dataframe_all = preproc_data()
    models = [ModelArchitecture_I2]

    experiments_all = ["all"]
    r2_mean = []
    summary_table = []
    experiments_test = ["all"]

    for model in models:
        for idx, experiments in enumerate(list(itertools.product(experiments_all, experiments_test))):
            experiment, experiment_test = experiments
            if experiment and experiment_test != "all":
                dataframe = dataframe_all[dataframe_all['experiment_type'].isin(experiment_test) or dataframe_all['experiment_type'].isin(experiment)]
            else:
                dataframe = dataframe_all

            train_data_loader, test_data_loader = init_loaders(dataframe, experiment, experiment_test)

            # for el in train_data_loader:
            #     print(el)

            name = "GoreTex_DIC_3_16_l201_" + str(experiment) + "_" + str(model.__name__)
            name = "test"
            print("----------------------------------------------------------------------")
            print(experiment)
            test_train = Trainer(
                                plot_valid=False,
                                epochs=10,
                                experiment_name=name,
                                l2_reg_coeff=0.01,
                                # l1_reg_coeff=0.001,
                                learning_rate=0.001,
                                # checkpoint=path,
                                model=model,
                                # SingleInvNet=SingleInvNet4,
                                batch_size=batch_size
                                )

            trained_model = test_train.train(train_data_loader, None, weighting_data=False)

            trained_model.eval()
            vpredictions = []
            for data in test_data_loader:
                features, target = data
                vpredictions.append(trained_model(features).detach().squeeze().numpy())
            vpredictions = np.array(vpredictions)
            # print(test_train.path_to_save_weights)

            dataframe["P11_model"] = vpredictions.transpose()[0]
            dataframe["P22_model"] = vpredictions.transpose()[1]
            dataframe.to_csv(os.path.join(test_train.path_to_save_weights, "data.csv"))
            # for column in dataframe:
            #     if column not in ["experiment_type", "P11_model", "P22_model"]:
            #     # if column not in ["P11_model", "P22_model"]:
            #         try:
            #             dataframe[column] = dataframe[column].apply(lambda tensor: tensor.numpy())
            #         except:
            #             print(column)

            metrics = plot_results_by_experiment_type(dataframe, test_train.path_to_save_weights)
            print(metrics)
            dataframe.pop("P11")
            dataframe.pop("P22")
            # r2_mean.append(metrics[metrics["Category"] == "Mean"])
            # metrics.to_csv(os.path.join(test_train.path_to_save_weights, "metrics.csv"))

            # weights_dict[experiment] = trained_model.potential_constants

            r2_mean.append(metrics[metrics["Category"] == "Mean"])
            metrics.to_csv(os.path.join(test_train.path_to_save_weights, "metrics.csv"))

            blocks = trained_model.extract_weights_as_blocks()
            row = {"Experiment Type": experiment}
            row.update(blocks)
            mean_values = metrics.loc[metrics['Category'] == 'Mean', ['PX', 'PY']].values.flatten()
            row.update({'Mean': (mean_values[0], mean_values[1])})
            summary_table.append(row)
            # Преобразование в DataFrame

            # Сохранение таблицы
        # print(summary_table)



            # trained_model.path_to_best_weights

        plt.show()

        summary_df = pd.DataFrame(summary_table)
        print(f"Сводная таблица: \n {summary_table}")

        name = "weights_blocks_summary.csv"
        output_path = os.path.join("../results", str(experiments_all))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        summary_df.to_csv(os.path.join(output_path, name), index=False)

if __name__ == "__main__":
    main()
    # name = "GoreTex_DIC_3_16_" + str(experiment) + "_" + ModelArchitecture_I5

    # path = r"C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\src\CANN_torch\pretrained_models\GoreTex_DIC_3_16_030_100_ModelArchitecture_I5"
    # # dir = r"\GoreTex_DIC_3_16_100_500_ModelArchitecture_I5"
    # pth = r"\20241203_0051_4477" + ".pth"
    # path = path + pth
    # test_train = Trainer(
    #     checkpoint=path,
    #     model=ModelArchitecture_I5,
    #     SingleInvNet=SingleInvNet4
    #
    # )
    # test_train.model.load_state_dict(torch.load(path))
    # test_train.model.get_weights()
    # print(test_train.model.potential_constants)
    # print(test_train.model.get_potential(p=3))
    # print(test_train.model)
    # trained_model = ModelArchitecture_I5()
    # #
    # ws = trained_model.get_weights()
    #
    # print(choose_experiment("Equi"))
    # data = preproc_data("Equi")
    # print(data)
    # plt.scatter(data['lamx'], data['Px'])
    # plt.show()
    # print(data["Px"])