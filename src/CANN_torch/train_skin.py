import numpy as np
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
import matplotlib.gridspec as gridspec


# hyperparameters and paths
# num_points = 32
batch_size = 32
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


def preproc_data():
    # experiments_path = get_list_of_paths_to_experiments_type(experiment_mod)
    # data_frames = [load_and_extract(file, file[-11:-4]) for file in experiments_path]

    # print(data_frames)
    path_to_data = r"C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\data\pocine_skin\NODE_porcine_skin_data_1.csv"

    all_data = pd.read_csv(path_to_data, index_col=0)

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
    # for variable in mechanical_variables.keys():
    #     func_calc = mechanical_variables.get(variable)
    #
    #     all_data[variable] = all_data['lambda_y'].apply(lambda lambdas: func_calc(lambdas, 1))


    lam_x = all_data["lambda_x"]
    lam_y = all_data["lambda_y"]
    sigma_x = all_data["sigma_xx [MPa]"]
    sigma_y = all_data["sigma_yy [MPa]"]
    # F = np.zeros((len(lam_x), 2, 2))
    # F = np.zeros(len(lam_x))
    P = np.zeros((len(sigma_y), 2, 2))
    F = np.array([[[x, 0], [0, y]] for (x, y) in zip(lam_x, lam_y)])
    F[:] = np.linalg.inv(F[:])

    P_x = sigma_x * 1 / lam_x
    P_y = sigma_y * 1 / lam_y

    # P[:] = np.array([[[x, 0], [0, y]] for (x, y) in zip(sigma_x, sigma_y)])[:] * F[:]
    # P1 = all_data["sigma_xx [MPa]"] = P[:, 0, 0]
    # P2 = all_data["sigma_xx [MPa]"] = P[:, 1, 1]
    all_data["sigma_xx [MPa]"] = P_x
    all_data["sigma_yy [MPa]"] = P_y
    # print(P_x == sigma_x)
    # print(P_y == sigma_y)
    # sigma_x = np.array([sigma_x *

    # plt.plot(lam_x, P1)
    # plt.plot(lam_y, P2)
    # plt.title("experimental P_xx")
    # plt.grid()
    # plt.show()

    plt.title("experimental P")
    plt.plot(lam_x, P_x)
    plt.plot(lam_y, P_y)
    plt.grid()
    plt.show()

    print("test")
    # combined_data.columns = ['lambda1', 'P_experimental', 'I1', 'I2', 'F', 'experiment_type']
    # break_list = [72, 73 + 75, 73 + 76 + 80, 73 + 76 + 81 + 100, 73 + 76 + 81 + 101 + 71]
    #
    # # c_lis = ['b','g','r','k','m']
    #
    # st = 73 + 76 + 81 + 101
    # end = 73 + 76 + 81 + 101 + 71
    #
    # st = 0
    # all_lam_x = []
    # all_lam_y = []
    # all_Sigma_xx = []
    # all_Sigma_yy = []
    # for i, end in enumerate(break_list):
    #     all_lam_x.append(lam_x[st:end])
    #     all_lam_y.append(lam_y[st:end])
    #     all_Sigma_xx.append(sigma_x[st:end])
    #     all_Sigma_yy.append(sigma_y[st:end])
    #     st = end + 1
    #
    # fig = plt.figure(figsize=(800 / 72, 600 / 72))
    # spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    # ax1 = fig.add_subplot(spec[0:2, 0:2])
    # ax2 = fig.add_subplot(spec[2, 0:2])
    # ax3 = fig.add_subplot(spec[2, 2])
    # plt.show()

    return all_data


class SimpleDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        # self.features = [dataframe[0],dataframe[2], dataframe[3], dataframe[4], dataframe[5]]
        # self.targets  = dataframe[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # features = self.data.iloc[idx][:1]
        # target  = self.data.iloc[idx][1:]
        data = copy.deepcopy([*self.data.iloc[idx]])
        # f1, f2, t1, t2 = data
        feature1 = data.pop(0)
        feature2 = data.pop(0)
        target = torch.tensor(data)
        # target1 = features.pop(2)
        # target2 = features.pop(2)
        feature = [feature1, feature2, torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)]
        return feature, target
        # return torch.tensor((f1, f2, 0, 0, 0, 0, 0, 0)), torch.tensor((t1, t2))

    def to_tensor(self):
        for column in self.data.columns:
            if column != "experiment_type":
                self.data[column] = self.data[column].apply(
                    lambda x: torch.tensor(x, dtype=torch.float32)).copy()


def init_loaders(all_data=None, experiment_type: Optional[str] or Optional[list] = None):
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
    f, t = train_dataset[10]
    # print(train_dataset[10])
    print("f:", f)
    print("t:", t)
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


def calculate_r2_by_chunks(y_true, y_pred, chunk_sizes):
    """
    Calculate R² scores for each custom chunk of the data.

    Parameters:
    - y_true (np.array or pd.Series): Ground truth (correct) target values.
    - y_pred (np.array or pd.Series): Estimated target values.
    - chunk_sizes (list of int): List of sizes for each chunk.

    Returns:
    - r2_scores (list): List of R² scores for each chunk.
    """
    r2_scores = []
    start = 0
    for size in chunk_sizes:
        y_true_chunk = y_true[start:start + size]
        y_pred_chunk = y_pred[start:start + size]
        r2_chunk = r2_score(y_true_chunk, y_pred_chunk)
        r2_scores.append(r2_chunk)
        start += size
    return r2_scores


def plot_results_by_experiment_type(data: pd.DataFrame, path_to_save: str, plot_name_prefix: str = "plot", chunk_sizes=[72, 148, 229, 330]):
    """
    Visualize dataset and predictions.

    Parameters:
    - data (pd.DataFrame): The data to plot.
    - path_to_save (str): The directory where the plots will be saved.
    - plot_name_prefix (str): The prefix to use for plot filenames.
    - chunk_sizes (list of int): List of sizes for each chunk.
    """
    data.columns = ['lambda_x', 'lambda_y', 'stress_x', 'stress_y', 'P11', 'P22']

    experiment_types = ["biaxial"]
    r2s = dict.fromkeys(experiment_types)

    for experiment_type in experiment_types:
        subset = data

        # Calculate R² by custom chunks for P11 and stress_x
        r2_p11_chunks = calculate_r2_by_chunks(subset['stress_x'], subset['P11'], chunk_sizes)
        # Calculate R² by custom chunks for P22 and stress_y
        r2_p22_chunks = calculate_r2_by_chunks(subset['stress_y'], subset['P22'], chunk_sizes)

        # Plot the first graph (lambda_x, P11) and scatter stress_x for the experiment type
        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=subset, x='lambda_x', y='P11', ax=ax1, color='blue', label='P11')
        sns.scatterplot(data=subset, x='lambda_x', y='stress_x', ax=ax1, color='red', label='Stress_x')

        ax1.set_title(f'{experiment_type}: P_11_exp and P_11_model\nR² (by chunks) = {np.mean(r2_p11_chunks):.2f}')
        ax1.set_xlabel('Lambda_x')
        ax1.set_ylabel('Piola stress (MPa)')
        plt.legend()
        plt.savefig(os.path.join(path_to_save, f"{plot_name_prefix}_{experiment_type}_x.png"))
        plt.show()

        # Plot the second graph (lambda_y, P22) and scatter stress_y for the experiment type
        fig, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=subset, x='lambda_y', y='P22', ax=ax2, color='blue', label='P22')
        sns.scatterplot(data=subset, x='lambda_y', y='stress_y', ax=ax2, color='red', label='Stress_y')

        ax2.set_title(f'{experiment_type}: P_22_exp and P_22_model\nR² (by chunks) = {np.mean(r2_p22_chunks)::.2f}')
        ax2.set_xlabel('Lambda_y')
        ax2.set_ylabel('Piola stress (MPa)')
        plt.legend()
        plt.savefig(os.path.join(path_to_save, f"{plot_name_prefix}_{experiment_type}_y.png"))
        plt.show()

        r2s[experiment_type] = {
            'r2_p11_chunks': r2_p11_chunks,
            'r2_p22_chunks': r2_p22_chunks
        }

    return r2s


def main():

    dataframe = preproc_data()
    models = [ModelArchitecture_I5]

    for model in models:

        train_data_loader, test_data_loader = init_loaders(dataframe, None)
        name = "GoreTex_" + str(model.__name__)
        # name = "test_" + "GoreTex_" + str(model.__name__)
        print("----------------------------------------------------------------------")
        test_train = Trainer(
                            plot_valid=False,
                            epochs=10000,
                            experiment_name=name,
                            l2_reg_coeff=None,
                            l1_reg_coeff=None,
                            learning_rate=0.001,
                            checkpoint=None,
                            model=model,
                            )

        # trained_model = test_train.train(train_data_loader, None, weighting_data=False)
        trained_model = test_train.model
        test_train.model.load_state_dict(torch.load(r"C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\src\CANN_torch\pretrained_models\GoreTex_ModelArchitecture_I5\20240830_1625_2375.pth"))
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
