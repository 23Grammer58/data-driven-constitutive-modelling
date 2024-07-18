import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from datetime import datetime
import os
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.metrics import r2_score
from models.CNN import *
# from models.CNN import StrainEnergyCANN, StrainEnergyCANN_C, StrainEnergyCANN_polinomial3
from utils.dataload import ExcelDataset, normalize_data
from utils.visualisation import *
import seaborn as sns
import pandas as pd
from trainer import Trainer


def main():
    path_to_data = r"..\..\data\GoreTex"

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

    # get_list_of_paths_to_experiments_type("biaxial")

    experiment = "biaxial"
    experiments_path = get_list_of_paths_to_experiments_type(experiment)
    data_frames = [load_and_extract(file, file[-11:-4]) for file in experiments_path]

    # print(data_frames)

    df = pd.concat(data_frames, ignore_index=True)
    thinned_data_frames = []
    num_points = 60

    df_list = []

    for df in data_frames:
        if num_points != -1:
            indices = np.linspace(0, len(df) - 1, num_points, dtype=int)
            # df[1] = df[1] / 10**6
            df = pd.DataFrame(df.iloc[indices].copy())
        # print(type(sampled_df))
        df['lambdas'] = list(zip(df['lambda_clamps_X'], df['lambda_clamps_Y']))
        df['stresses'] = list(zip(df['mean_stress_x_mpa'], df['mean_stress_y_mpa']))
        df_list.append(df[:num_points//2])
        # df_list.append(df)
        # thinned_df = df.iloc[::len(df) // 20, :]  # Выбор каждого 45-го значения
        # thinned_data_frames.append(thinned_df)
    data_frames = df_list
    # df.iloc[40:60]
    # print(df_list[1])
    # thinned_data_frames
    # sampled_df_list[0]
    # for item in data_frames:
    # print(item)

    I1_bx = lambda lam1, lam2: lam1 ** 2 + lam2 ** 2 + 1 / (lam1 * lam2) ** 2
    I2_bx = lambda lam1, lam2: 1 / lam1 ** 2 + 1 / lam2 ** 2 + (lam1 * lam2) ** 2
    I4_bx = lambda lam1, lam2: lam1 ** 2 + math.cos(math.pi / 4) ** 2 + lam2 * math.sin(torch.pi / 4) ** 2
    I5_bx = lambda lam1, lam2: lam1 ** 2 + math.cos(math.pi / 4) ** 4 + lam2 * math.sin(torch.pi / 4) ** 4

    F_bx = lambda lam1, lam2: ([lam1, 0, 0], [0, lam2, 0], [0, 0, 1 / (lam1 * lam2)])

    mechanical_variables = {
        "I1": I1_bx,
        "I2": I2_bx,
        "I4": I4_bx,
        "I5": I5_bx,

        "F": F_bx,
        # "exp_type": [(lambda x: 1), (lambda x: 0)] # 1 - torsion&compression, 0 - shear
        # "torsion_compression": (lambda x: 1)
    }

    # calculate I1, I2, F from lambda (torsion&compression and shear)
    for variable in mechanical_variables.keys():
        func_calc = mechanical_variables.get(variable)

        for data_frame in data_frames:
            data_frame[variable] = data_frame['lambdas'].apply(lambda lambdas: func_calc(lambdas[0], lambdas[0]))

    experiments = ['Compression', 'Tensile', 'Shear']
    combined_data = pd.concat(data_frames).reset_index(drop=True, inplace=False)
    # combined_data.columns = ['lambda1', 'P_experimental', 'I1', 'I2', 'F', 'experiment_type']
    combined_data.pop("lambdas")
    combined_data.pop("stresses")
    experiment_type = combined_data.pop("experiment_type")
    combined_data["experiment_type"] = experiment_type
    print(combined_data)
    # combined_data.to_csv( "Uniaxial.csv")
    # combined_data = combined_data["lambda_clamps_X", "lambda_clamps_Y",	"mean_stress_x_mpa", "mean_stress_y_mpa", "experiment_type"
    from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
    import copy

    class CustomDataset(Dataset):
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

    start = 0
    end = 30
    train_dataset = CustomDataset(combined_data[start:end].copy())
    test_dataset = CustomDataset(combined_data.copy())
    f, t = train_dataset[1]
    # lam, i1, i2, F, exp_type = f
    # print(f)
    # print(t)

    def init_loaders(experiments: Optional[str] = ["Shear", "Tensile", "Comression"]):
        if type(experiments) == str:
            experiments = [experiments]
        elif type(experiments) == list:
            df = pd.concat([combined_data[combined_data["experiment_type"] == experiment] for experiment in
                            experiments]).reset_index(drop=True, inplace=False)
        else:
            df = combined_data
        train_dataset = CustomDataset(df.copy())
        test_dataset = CustomDataset(combined_data.copy())

        train_dataset.to_tensor()
        test_dataset.to_tensor()

        train_data_loader = DataLoader(
            train_dataset,
            shuffle=True,
            # num_workers=1,
            pin_memory=False,
            batch_size=2
        )
        test_data_loader = DataLoader(
            test_dataset,
            shuffle=False,
            # num_workers=1,
            pin_memory=False
        )

        return train_data_loader, test_data_loader

    train_data_loader, test_data_loader = init_loaders(None)
    print(train_data_loader.dataset.data)

    def plot_results_by_experiment_type(data: pd.DataFrame, plot_name_prefix="plot"):

        # Переименовываем столбцы для удобства
        # data.columns = ['index', 'lambda_x', 'lambda_y', 'stress_x', 'stress_y', 'I1', 'I2', 'I4', 'I5', 'F', 'experiment_type', 'P11', 'P22']
        data.columns = ['index', 'lambda_x', 'lambda_y', 'stress_x', 'stress_y', 'I1', 'I2', 'I4', 'I5', 'F',
                        'experiment_type', 'P11', 'P22']

        # Получим уникальные типы экспериментов
        experiment_types = data['experiment_type'].unique()

        for experiment in experiment_types:
            subset = data[data['experiment_type'] == experiment]

            # Создадим первый график (lambda_x, P11) и скаттер на нем stress_x для данного типа эксперимента
            fig, ax1 = plt.subplots(figsize=(10, 6))

            sns.lineplot(data=subset, x='lambda_x', y='P11', ax=ax1, label='P11')
            sns.scatterplot(data=subset, x='lambda_x', y='stress_x', ax=ax1, color='red', label='Stress_x')

            # Рассчитаем R² для P11
            r2_p11 = r2_score(subset['stress_x'], subset['P11'])

            ax1.set_title(f'{experiment}: P11 and Stress_x\nR² = {r2_p11:.2f}')
            ax1.set_xlabel('Lambda_x')
            ax1.set_ylabel('P11 / Stress_x (MPa)')
            plt.legend()
            plt.savefig(f"{plot_name_prefix}_{experiment}_plot1.png")
            plt.show()

            # Создадим второй график (lambda_y, P22) и скаттер на нем stress_y для данного типа эксперимента
            fig, ax2 = plt.subplots(figsize=(10, 6))

            sns.lineplot(data=subset, x='lambda_y', y='P22', ax=ax2, label='P22')
            sns.scatterplot(data=subset, x='lambda_y', y='stress_y', ax=ax2, color='blue', label='Stress_y')

            # Рассчитаем R² для P22
            r2_p22 = r2_score(subset['stress_y'], subset['P22'])

            ax2.set_title(f'{experiment}: P22 and Stress_y\nR² = {r2_p22:.2f}')
            ax2.set_xlabel('Lambda_y')
            ax2.set_ylabel('P22 / Stress_y (MPa)')
            plt.legend()
            plt.savefig(f"{plot_name_prefix}_{experiment}_plot2.png")
            plt.show()

    # Вызовем функцию plot_results_by_experiment_type с нашими данными

    # experiments=[["Tensile", "Comression"], ["Tensile", "Shear"], ["Shear", "Comression"], "Shear", "Tensile", "Compression"]
    # experiments = ["Tensile", "Comression", "Shear"]
    # models = [StrainEnergyCANN_C, StrainEnergyCANN_polinomial3]
    # models = [StrainEnergyCANN_Ani]
    from models.CANN_gpt import ModelArchitecture_I5
    models = [ModelArchitecture_I5]
    path = r"C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\src\CANN_torch\pretrained_models"
    for model in models:

        # for idx, experiment in enumerate(experiments):
        train_data_loader, test_data_loader = init_loaders(None)
        name = "GoreTex_eqbx_" + str(model.__name__)
        print("----------------------------------------------------------------------")
        # print(experiment)
        test_train = Trainer(
            plot_valid=False,
            epochs=200,
            experiment_name=name,
            l2_reg_coeff=None,
            l1_reg_coeff=None,
            learning_rate=0.01,
            checkpoint=None,
            model=model,

            # dtype = torch.float64
        )

        trained_model = test_train.train(train_data_loader, None, weighting_data=False)

        trained_model.eval()
        vpredictions = []
        vtargets = []
        for data in test_data_loader:
            features, target = data
            # vpredictions.append(zip(trained_model(features).detach().numpy()))
            vpredictions.append(trained_model(features).detach().numpy())
        # print(trained_model.get_potential())
        # print(trained_model.potential_constants)
        vpredictions = np.array(vpredictions)
        print(vpredictions.transpose()[0])
        combined_data["P11_model_" + name] = vpredictions.transpose()[0]
        combined_data["P22_model_" + name] = vpredictions.transpose()[1]
        combined_data.to_csv(os.path.join(os.path.join(path, str(name)), "data.csv"))
        plot_results_by_experiment_type(combined_data)
        combined_data.pop("P11_model_" + name)
        combined_data.pop("P22_model_" + name)
    # trained_model = StrainEnergyCANN_C()

    # print("R2:", r2_score_own(vtargets, vpredictions))
    # plt.figure(figsize=(10, 5))
    # plt.plot(vpredictions, label='P_pred', color='red')
    # plt.plot(vtargets, label='P_true', color='black')
    # plt.xlabel('lambda/gamma')
    # plt.ylabel('P')
    # plt.title('Predictions vs. Targets')
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()