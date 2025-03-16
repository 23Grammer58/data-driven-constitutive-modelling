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
from models.CNN import *
from models.CANN_gpt import ModelArchitecture_I5, SingleInvNet6, SingleInvNet4
# from models.CNN import StrainEnergyCANN, StrainEnergyCANN_C, StrainEnergyCANN_polinomial3
from utils.dataload import ExcelDataset, normalize_data
from utils.visualisation import *
import seaborn as sns
import pandas as pd
from trainer import Trainer

# hyperparameters and paths
path_to_data = r"../../../data/GoreTex/2/DIC/3"
experiment_mod = "biaxial"
batch_size = 16
# path_to_results = r"C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\src\CANN_torch\pretrained_models"

def get_list_of_paths_to_experiments_type(experiment="uniaxial"):
    l = []
    for file in os.listdir(path_to_data):
        l.append(os.path.join(path_to_data, file))
    return l


def load_and_extract(file_path, experiment_type):
    df = pd.read_csv(file_path)
    if experiment_type != "Comp":
        df['experiment_type'] = experiment_type
    return df[["# Xlam", "PX", "Ylam", "PY", 'experiment_type']]


I1_bx = lambda lam1, lam2: lam1 ** 2 + lam2 ** 2 + 1 / (lam1 * lam2) ** 2
I2_bx = lambda lam1, lam2: 1 / lam1 ** 2 + 1 / lam2 ** 2 + (lam1 * lam2) ** 2
I4_bx = lambda lam1, lam2: lam1 ** 2 + math.cos(math.pi / 4) ** 2 + lam2 * math.sin(torch.pi / 4) ** 2
I5_bx = lambda lam1, lam2: lam1 ** 2 + math.cos(math.pi / 4) ** 4 + lam2 * math.sin(torch.pi / 4) ** 4
F_bx = lambda lam1, lam2: ([lam1, 0, 0], [0, lam2, 0], [0, 0, 1 / (lam1 * lam2)])

# Механические переменные и их функции расчёта
mechanical_variables = {
    "I1": I1_bx,
    "I2": I2_bx,
    "I4": I4_bx,
    "I5": I5_bx,
    "F": F_bx
}


def find_local_minima(data, window_size=2, threshold=0.0005):
    # Находим локальные минимумы
    local_min_indices = argrelextrema(data, np.less, order=window_size)[0]

    # Фильтруем по порогу
    local_minima = []
    for index in local_min_indices:
        if index > 0 and index < len(data) - 1:
            if data[index] < data[index - 1] - threshold and data[index] < data[index + 1] - threshold:
                local_minima.append(data[index])

    return local_minima

lamx = "# Xlam"
def preproc_data(experiment_protocols: list = None):
    experiments_path = get_list_of_paths_to_experiments_type()
    data_frames = [load_and_extract(file, file[-11:-4]) for file in experiments_path] # поменять для друго даты

    num_points = 10
    # _ = []

    # data_frames = _
    # разобьем комплексный эксперимент по циклам
    # data_frame_complex = data_frames[0]

    # data_frame_complex['Xlam_diff'] = data_frame_complex['# Xlam'].diff(periods=2)
    # data_frame_complex['Ylam_diff'] = data_frame_complex['Ylam'].diff(periods=2)
    #
    # # Найдем локальные минимумы в Xlam и Ylam (когда направление меняется с уменьшения на увеличение)
    # xlam_min_points = (data_frame_complex['Xlam_diff'].shift(-1) > 0) & (data_frame_complex['Xlam_diff'] < 0)
    # ylam_min_points = (data_frame_complex['Ylam_diff'].shift(-1) > 0) & (data_frame_complex['Ylam_diff'] < 0)
    #
    #
    # # Объединяем информацию о пиках деформации
    # min_points = (xlam_min_points | ylam_min_points).cumsum()
    #
    # # Обновляем столбец с номерами циклов
    # data_frame_complex['experiment_type'] = min_points

    # protocol_mapping = {
    #     0.0: "100_100",
    #     2.0: "100_075",
    #     4.0: "075_100",
    #     6.0: "100_050",
    #     8.0: "050_100",
    #     10.0: "100_030",
    #     12.0: "030_100"
    # }
    #
    # # Заменяем значения experiment_type на протоколы
    # data_frame_complex['experiment_type'] = data_frame_complex['experiment_type'].map(protocol_mapping)

    # Удаляем вспомогательные столбцы для дифференцирования
    # data_frames[0] = data_frame_complex.drop(columns=['Xlam_diff', 'Ylam_diff'])

    # Объединяем все DataFrame в один
    data_frames = pd.concat(data_frames).reset_index(drop=True, inplace=False)

    if not experiment_protocols:
        experiment_protocols = data_frames['experiment_type'].unique()

    data_frames_thin = []
    # Итерируемся по каждому уникальному experiment_type
    for experiment_type in experiment_protocols:

        # Фильтруем данные по текущему experiment_type
        data_frame = data_frames[data_frames['experiment_type'] == experiment_type]


        # first_id = data_frame.index[0]
        # last_id = data_frame.index[-1]
        #
        # # Проверяем условие HolX или работаем с lamx
        # if experiment_type == "HolX":
        #     lam_max = data_frame["Ylam"].idxmax()
        # else:
        #     lam_max = data_frame["# Xlam"].idxmax()  # Предполагаем, что lamx — это Xlam
        #
        # # Обрабатываем значения для каждого столбца P
        # for P in ["PX", "PY"]:
        #     P_before_max = data_frame[P].iloc[:lam_max - first_id].to_numpy()
        #     P_after_max = data_frame[P].iloc[lam_max - first_id +  1:].to_numpy()
        #     print(data_frame[P].iloc[lam_max - first_id])
        #
        #     len_before_max = len(P_before_max)
        #     len_after_max = len(P_after_max)
        #
        #     # Выровняем длины массивов, чтобы можно было их сложить
        #     if len_before_max > len_after_max:
        #         P_after_max = np.append(P_after_max, [0] * (len_before_max - len_after_max))
        #     elif len_before_max < len_after_max:
        #         P_before_max = np.append(P_before_max, [0] * (len_after_max - len_before_max))
        #
        #     # Вычисляем среднее значение и добавим в конец пиковое значение напряжения
        #     P_mean = np.append((P_before_max + P_after_max[::-1]) / 2, data_frame[P].iloc[lam_max - first_id])
        #
        #     # Убедимся, что длина P_mean совпадает с длиной исходных данных
        #     P_mean_series = pd.Series(P_mean, index=data_frame.index[:len(P_mean)])
        #
        #     # Добавляем результат в data_frame
        #     data_frames.loc[data_frame.index[:len(P_mean)], P + "_mean"] = P_mean_series

        # Для каждой механической переменной вычисляем значения

        # for variable, func_calc in mechanical_variables.items():
        #     # Создаем столбец с lambdas из значений lamx (предположим Xlam) и Ylam
        #     data_frame.loc[:, 'lambdas'] = list(zip(data_frame["# Xlam"], data_frame["Ylam"]))
        #
        #     # Применяем функцию расчета для каждой пары lambda
        #     # data_frames.loc[:, variable] = func_calc(data_frame["# Xlam"], data_frame["Ylam"])
        #     data_frames.loc[data_frame.index, variable] = data_frame['lambdas'].apply(
        #         lambda lambdas: func_calc(lambdas[0], lambdas[1]))

        # indices = np.linspace(1, len(data_frame) - 1, num_points, dtype=int)
        # data_frames_thin.append(pd.DataFrame(data_frame.iloc[indices].copy()))

    # combined_data = pd.concat(data_frames_thin).reset_index(drop=True, inplace=False)
    combined_data = data_frames
    combined_data.dropna(inplace=True)
    # print(combined_data.columns)

    combined_data = combined_data[[lamx, 'Ylam', 'PX', 'PY', 'experiment_type']]
    combined_data.columns = ['lamx', 'lamy', 'Px', 'Py', 'experiment_type']
    # combined_data = combined_data[[lamx, 'Ylam', 'PX_mean', 'PY_mean', 'I1', 'I2',
    #    'I4', 'I5', 'F', 'experiment_type']]
    # combined_data.columns = ['lamx', 'lamy', 'Px', 'Py', 'I1', 'I2',
    #    'I4', 'I5', 'F', 'experiment_type']

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
        features.pop(-1)
        target = torch.tensor((target1.item(), target2.item()))

        return features, target

    def to_tensor(self):
        for column in self.data.columns:
            if column != "experiment_type":
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

    # переписать___________
    # if experiment_protocol_train == "all":
    #     train_dataframe = all_data
    if type(experiment_protocol_train) is str and experiment_protocol_train != "all":
        experiment_protocol_train = [experiment_protocol_train]
    if type(experiment_protocol_train) is list or tuple:
        train_dataframe = pd.concat([all_data[all_data["experiment_type"] == experiment] for experiment in
                        experiment_protocol_train ]).reset_index(drop=True, inplace=False)
    else:
        train_dataframe = all_data

    # if experiment_protocol_test == "all":
    #     test_dataframe = all_data
    if type(experiment_protocol_test) is str and experiment_protocol_test != "all":
        experiment_protocol_test = [experiment_protocol_test]
    if type(experiment_protocol_test) is list:
        test_dataframe = pd.concat([all_data[all_data["experiment_type"] == experiment] for experiment in
                        experiment_protocol_test]).reset_index(drop=True, inplace=False)
    else:
        test_dataframe = all_data
    #_______________

    train_dataset = SimpleDataset(train_dataframe)
    test_dataset = SimpleDataset(all_data)

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
    # dataframe = dataframe_all
    models = [ModelArchitecture_I5]
    #models = [ModelArchitecture_I5_exp, ModelArchitecture_I5_log]

    # checkpoint_path = r"C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\src\CANN_torch\pretrained_models\GoreTex_DIC_2_OffX_ModelArchitecture_I5\20241030_1926_9999.pth"
    path = r"C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\src\CANN_torch\pretrained_models\GoreTex_DIC_3_log_16_['030_100', '100_100', '050_100', '100_075']_ModelArchitecture_I5\20241112_2220_2990.pth"
    # experiments_all = [["075_100"]]
    experiments_all = ["100_100", '100_050', "100_075", '050_100', "075_100", "100_030", "030_100", ["100_100", '100_050', "100_075", '050_100', "075_100", "100_030", "030_100"]]
    # experiments_all = ["all"]
    # experiments_all = [["100_100"], ['100_050'], ["100_075"], ["100_030"], ['050_100'], ["075_100"], ["030_100"]]
    # experiments_all = [["100_100", '100_050', "100_075", "100_030", '050_100', "075_100", "030_100", "HolY", "OffX", "OffY"]]

    # experiments_all = [["030_100", "100_100", '050_100', "100_075", "HolY", "OffX"], ["030_100", "100_100", '050_100', "100_075"],
    #                    ["100_100", '100_050', "100_075", "100_030", '050_100', "075_100", "030_100", "HolY", "OffX", "OffY"]]
    # experiments_all = [["030_100", "100_100", '050_100', "100_075"], ["030_100"], ["100_100"], ['050_100'], ["100_075"]]
    # experiments_all = [["030_100", "100_100", '050_100', "100_075"]]
    # dataframe = dataframe_all[dataframe_all['experiment_type'].isin(["030_100", "100_100", "050_100", "100_075"])]
    # experiments_all = [list(itertools.combinations(experiments_all, r)) for r in range(1, len(experiments_all) + 1)]
    r2_mean = []
    summary_table = []
    experiments_test = ["all"]
    # experiments_test = [["100_100", '100_050', "100_075", '050_100', "075_100"]]

    # weights_dict = dict.fromkeys(experiments_all)
    # experiments_test = [["100_100", '100_050', "100_075", "100_030", '050_100', "075_100", "030_100"]]
    for model in models:
        for idx, experiments in enumerate(list(itertools.product(experiments_all, experiments_test))):
            experiment, experiment_test = experiments
            if experiment and experiment_test != "all":
                dataframe = dataframe_all[dataframe_all['experiment_type'].isin(experiment_test) or dataframe_all['experiment_type'].isin(experiment)]
            else:
                dataframe = dataframe_all
            train_data_loader, test_data_loader = init_loaders(dataframe, experiment, experiment_test)
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
                                SingleInvNet=SingleInvNet4,
                                batch_size=batch_size
                                )

            trained_model = test_train.train(train_data_loader, None, weighting_data=False)

            trained_model.eval()
            vpredictions = []
            for data in test_data_loader:
                features, target = data
                vpredictions.append(trained_model(features).detach().squeeze().numpy())
            vpredictions = np.array(vpredictions)
            print(test_train.path_to_save_weights)
            # print(vpredictions)
            # print(vpredictions.transpose()[0])
            # dataframe[dataframe['experiment_type'] == experiment_type]
            dataframe["P11_model"] = vpredictions.transpose()[0]
            dataframe["P22_model"] = vpredictions.transpose()[1]
            dataframe.to_csv(os.path.join(test_train.path_to_save_weights, "data.csv"))
            for column in dataframe:
                if column not in ["experiment_type", "P11_model", "P22_model"]:
                    dataframe[column] = dataframe[column].apply(lambda tensor: tensor.numpy())

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