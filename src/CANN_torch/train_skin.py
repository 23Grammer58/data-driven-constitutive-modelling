import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

import copy
from datetime import datetime
import os
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.metrics import r2_score
from models.CNN import *
from models.CANN_gpt import ModelArchitecture_I5
# from models.CNN import StrainEnergyCANN, StrainEnergyCANN_C, StrainEnergyCANN_polinomial3
from utils.dataload import ExcelDataset, normalize_data
from utils.visualisation import *
import seaborn as sns
import pandas as pd
from trainer import Trainer

# hyperparameters and paths
path_to_data = r"..\..\data\GoreTex"
experiment_mod = "biaxial"
num_points = 32
batch_size = 2
# path_to_results = r"C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\src\CANN_torch\pretrained_models"


def get_list_of_paths_to_experiments_type(experiment="uniaxial"):
    experiment_type_path = os.path.join(path_to_data, experiment)
    l = []
    for file in os.listdir(experiment_type_path):
        l.append(os.path.join(experiment_type_path, file))
    return l


def load_and_extract(file_path, experiment_type):
    df = pd.read_csv(file_path)
    df['experiment_type'] = experiment_type
    return df[['lambda_clamps_X', 'lambda_clamps_Y', 'mean_stress_x_mpa', 'mean_stress_y_mpa', 'experiment_type']]


I1_bx = lambda lam1, lam2: lam1 ** 2 + lam2 ** 2 + 1 / (lam1 * lam2) ** 2
I2_bx = lambda lam1, lam2: 1 / lam1 ** 2 + 1 / lam2 ** 2 + (lam1 * lam2) ** 2
I4_bx = lambda lam1, lam2: lam1 ** 2 + math.cos(math.pi / 4) ** 2 + lam2 * math.sin(torch.pi / 4) ** 2
I5_bx = lambda lam1, lam2: lam1 ** 2 + math.cos(math.pi / 4) ** 4 + lam2 * math.sin(torch.pi / 4) ** 4
F_bx = lambda lam1, lam2: ([lam1, 0, 0], [0, lam2, 0], [0, 0, 1 / (lam1 * lam2)])


def preproc_data(experiment_mod="biaxial"):
    experiments_path = get_list_of_paths_to_experiments_type(experiment_mod)
    data_frames = [load_and_extract(file, file[-11:-4]) for file in experiments_path]

    # print(data_frames)

    all_data = pd.concat(data_frames, ignore_index=True)
    thinned_data_frames = []
    df_list = []

    for df in data_frames:
        if num_points != -1:
            indices = np.linspace(1, len(df) - 1, num_points, dtype=int)
            # df[1] = df[1] / 10**6
            df = pd.DataFrame(df.iloc[indices].copy())
        # print(type(sampled_df))
        df['lambdas'] = list(zip(df['lambda_clamps_X'], df['lambda_clamps_Y']))
        df['stresses'] = list(zip(df['mean_stress_x_mpa'], df['mean_stress_y_mpa']))
        df_list.append(df[:num_points // 2])

    data_frames = df_list

    mechanical_variables = {
        "I1": I1_bx,
        "I2": I2_bx,
        "I4": I4_bx,
        "I5": I5_bx,

        "F": F_bx,
        # "exp_type": [(lambda x: 1), (lambda x: 0)] # 1 - torsion&compression, 0 - shear
        # "torsion_compression": (lambda x: 1)
    }

    # calculate I1, I2, F from lambda (bi-axial)
    for variable in mechanical_variables.keys():
        func_calc = mechanical_variables.get(variable)

        for data_frame in data_frames:
            data_frame[variable] = data_frame['lambdas'].apply(lambda lambdas: func_calc(lambdas[0], lambdas[0]))

    combined_data = pd.concat(data_frames).reset_index(drop=True, inplace=False)
    # combined_data.columns = ['lambda1', 'P_experimental', 'I1', 'I2', 'F', 'experiment_type']
    combined_data.pop("lambdas")
    combined_data.pop("stresses")
    experiment_mod = combined_data.pop("experiment_type")
    combined_data["experiment_type"] = experiment_mod
    combined_data["mean_stress_y_mpa"][combined_data["mean_stress_x_mpa"] < 0] = 1e-5
    combined_data["mean_stress_y_mpa"][combined_data["mean_stress_y_mpa"] < 0] = 1e-5

    # combined_data["mean_stress_x_mpa"].loc[combined_data["mean_stress_x_mpa"] < 0] = 1e-5
    # combined_data["mean_stress_y_mpa"].loc[combined_data["mean_stress_y_mpa"] < 0] = 1e-5
    # combined_data.loc[combined_data["mean_stress_x_mpa"] < 0, :] = 1e-5
    # combined_data.loc[combined_data["mean_stress_y_mpa"] < 0, :] = 1e-5

    return combined_data


class SimpleDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        # self.features = [dataframe[0],dataframe[2], dataframe[3], dataframe[4], dataframe[5]]
        # self.targets  = dataframe[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = copy.deepcopy([*self.data.iloc[idx]])
        target1 = features.pop(2)
        target2 = features.pop(2)
        target = torch.tensor((target1.item(), target2.item()))
        return features, target

    def to_tensor(self):
        for column in self.data.columns:
            if column != "experiment_type":
                self.data[column] = self.data[column].apply(
                    lambda x: torch.tensor(x, dtype=torch.float32)).copy()


def init_loaders(all_data=None, experiment_type: Optional[str] or Optional[list] = None):
    global experiment
    global batch_size

    if all_data is None:
        all_data = preproc_data("biaxial")

    if type(experiment_type) is str:
        experiment_type = [experiment_type]
    elif type(experiment_type) is list:
        train_dataframe = pd.concat([all_data[all_data["experiment_type"] == experiment] for experiment in
                        experiment_type]).reset_index(drop=True, inplace=False)
    else:
        train_dataframe = all_data

    train_dataset = SimpleDataset(train_dataframe)
    test_dataset = SimpleDataset(all_data)

    train_dataset.to_tensor()
    test_dataset.to_tensor()

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


def plot_results_by_experiment_type(data: pd.DataFrame, path_to_save: str, plot_name_prefix: str = "plot"):
    """
     Visualize dataset and predictions.

     Parameters:
     - experiment_col (str): The column name for the experiment identifier.
     - x_col (int or str): The column name or index for the x-axis data.
     - y_col (int or str): The column name or index for the y-axis data.
     """

    data.columns = ['lambda_x', 'lambda_y', 'stress_x', 'stress_y', 'I1', 'I2', 'I4', 'I5', 'F',
                    'experiment_type', 'P11', 'P22']

    # Получим уникальные типы экспериментов
    experiment_types = data['experiment_type'].unique()
    r2s = dict.fromkeys(experiment_types)

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
        plt.savefig(os.path.join(path_to_save, f"{plot_name_prefix}_{experiment_type}_x.png"))
        plt.show()

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
        plt.show()
        r2s[experiment_type] = (r2_p11, r2_p22)
    return r2s

def main():

    # experiments = [["100_100", '100_050', "100_075", "100_033"]]
    experiments =\
        [
            ["100_050"],
            ["100_100", '100_050', "100_075", "100_033"],
            ["100_100", '075_100', "100_075", "050_100"],
            ["100_100", '050_100', "075_100", "033_100"],
            ["100_100", '100_050', "033_100"]
        ]
    dataframe = preproc_data(experiment_mod=experiment_mod)
    models = [ModelArchitecture_I5]

    for model in models:

        for idx, experiment in enumerate(experiments):
            train_data_loader, test_data_loader = init_loaders(dataframe, experiment)
            name = "GoreTex_" + str(experiment) + "_" + str(model.__name__)
            print("----------------------------------------------------------------------")
            print(experiment)
            test_train = Trainer(
                                plot_valid=False,
                                epochs=10000,
                                experiment_name=name,
                                l2_reg_coeff=0.0001,
                                l1_reg_coeff=0.0001,
                                learning_rate=0.001,
                                checkpoint=None,
                                model=model,
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
            dataframe["P11_model"] = vpredictions.transpose()[0]
            dataframe["P22_model"] = vpredictions.transpose()[1]
            dataframe.to_csv(os.path.join(test_train.path_to_save_weights, "data.csv"))
            for column in dataframe:
                if column not in ["experiment_type", "P11_model", "P22_model"]:
                    dataframe[column] = dataframe[column].apply(lambda tensor: tensor.numpy())

            metrics = pd.DataFrame(plot_results_by_experiment_type(dataframe, test_train.path_to_save_weights))
            print(metrics)
            metrics.to_csv(os.path.join(test_train.path_to_save_weights, "metrics.csv"))
            dataframe.pop("P11")
            dataframe.pop("P22")

        plt.show()


if __name__ == "__main__":
    main()
