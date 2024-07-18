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

from models.CANN_gpt import *

def r2_score_own(Truth, Prediction):
    R2 = r2_score(Truth,Prediction)
    return max(R2,0.0)


class Trainer:
    def __init__(self,
                 checkpoint: str = None,
                 experiment_name: Optional[str] = "test",
                 model: nn.Module = StrainEnergyCANN_Ani,
                 device: Optional[str] = "cpu",
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 plot_valid: bool = False,
                 batch_size: int = 1,
                 l1_reg_coeff: Optional[float] = 0.001,
                 l2_reg_coeff: Optional[float] = 0.001,
                 dtype = torch.float32
                 ):
        """
        Класс для обучения CANN моделей.

         Аргументы:
            - `experiment_name` (str): Название эксперимента. По умолчанию "test".
            - `model` (nn.Module): Архитектура модели для обучения. По умолчанию `StrainEnergyCANN_C`.
            - `path_to_save_weights` (str): Путь для сохранения весов модели.
            - `epochs` (int): Количество эпох обучения. По умолчанию 100.
            - `learning_rate` (float): Скорость обучения. По умолчанию 0.001.
            - `plot_valid` (bool): Отрисовка ошибки на валидационном датасете. По умолчанию False.
            - `l1_reg_coeff` (float): Коэффициент регуляризации L1. По умолчанию 0.001.
            - `l2_reg_coeff` (float): Коэффициент регуляризации L2. По умолчанию 0.001.
            - `checkpoint` (str): Путь до весов модели, с которыми модель инициализируется. По умолчанию 1.
            - `device` (str): Устройство для выполнения вычислений (cpu или cuda). По умолчанию "cpu".

        """

        self.l1_reg_coeff = l1_reg_coeff
        self.l2_reg_coeff = l2_reg_coeff
        if model == ModelArchitecture_I5:
            psi_model = StrainEnergy_i5()
            self.model = model(psi_model, setAl=True, init=torch.pi / 4)
        else:
            self.model = model(batch_size, device=device, dtype=dtype)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.experiment_name = experiment_name
        self.plot_valid = plot_valid
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        self.path_to_save_weights = os.path.join("pretrained_models", self.experiment_name)
        self.batch_size = batch_size
        self.path_to_best_weights = None
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint))

    def train(self,
              train_loader: Optional[DataLoader],
              test_loader: Optional[DataLoader] = None,
              weighting_data: bool = True,
              train_mode: str = "one_mode"):
        """
        Trains the model using the provided training and testing data loaders.

        Parameters:
        -----------
        train_loader : Optional[DataLoader]
            DataLoader containing the training data.
        test_loader : Optional[DataLoader], optional
            DataLoader containing the testing/validation data. If None, validation is not performed (default is None).
        weighting_data : bool, optional
            If True, applies different weights to the loss based on experiment type (default is True).
        train_mode : str, optional
            Determines the training mode. Options are "one_mode" and "multiple_mode" (default is "one_mode").

        Returns:
        --------
        model : nn.Module
            The trained model.

        Notes:
        ------
        - If `self.experiment_name` is provided, the model weights are saved in a directory named after the experiment.
        - Uses Mean Squared Error (MSE) loss for training.
        - Applies L1 and L2 regularization if coefficients are provided.
        - Clamps the model weights to ensure non-negativity.
        - Plots the training loss over epochs.
        - Saves the best model weights based on validation loss.
        """

        # Initialize the model, loss function, and optimizer
        if self.experiment_name is not None:
            path_to_save_weights = os.path.join("pretrained_models", self.experiment_name)
            if not os.path.exists(path_to_save_weights):
                os.makedirs(path_to_save_weights)
                print(f"Directory {path_to_save_weights} created successfully")
            else:
                print(f"Directory {path_to_save_weights} already exists")

        loss_fn = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_data_count = len(train_loader)

        if test_loader is None:
            test_data_count = train_data_count
        else:
            test_data_count = len(test_loader)
        def train_one_epoch(epoch_index):
            epoch_loss = 0.
            last_loss = 0.

            for i, data in enumerate(train_loader):
                features, target = data
                _, _, i1, i2, i4, i5, _, exp_type = features

                optimizer.zero_grad()
                stress_model = self.model(features)
                # print(target)
                # print(stress_model)
                # print(i1, i2, i4, i5)
                # target = target.squeeze()
                # loss = loss_fn(stress_model, target)
                # loss = loss.sum()


                loss_xx = loss_fn(stress_model.T[0], target.T[0])
                loss_yy = loss_fn(stress_model.T[1], target.T[1])
                loss = loss_xx + loss_yy
                loss = loss.sum()
                if weighting_data:
                    if exp_type == "Compression":
                        loss *= 0.5
                    elif exp_type == "Tensile":
                        loss *= 1.5

                if self.l2_reg_coeff is not None:
                    l2_reg = self.model.calc_regularization(2)
                    loss += 0.5 * self.l2_reg_coeff * l2_reg

                if self.l2_reg_coeff is not None:
                    l1_reg = self.model.calc_l1()
                    loss += self.l1_reg_coeff * l1_reg

                loss.backward(retain_graph=True)
                # loss.backward()

                optimizer.step()

                # turn negative weights to zero
                self.model.clamp_weights()

                # last_loss = loss.item()
                epoch_loss += loss.item()

            return epoch_loss

        epoch_number = 0
        best_vloss = torch.inf
        loss_history = []
        best_epoch = 0
        # vlosses = []
        # vpredictions = []
        # vtargets = []

        # Training the model
        for epoch in range(self.epochs):
            self.model.train(True)
            avg_loss = train_one_epoch(epoch_number) / train_data_count

            running_vloss = 0.0

            # validation = False
            if test_loader:
                self.model.train(False)

                # running_vloss = self.test(test_loader)
                for i, vdata in enumerate(test_loader):
                    vfeatures, vtarget = vdata

                    optimizer.zero_grad()
                    # stress_model = self.model(vfeatures)
                    vstress = self.model(vfeatures)
                    vloss = loss_fn(vtarget, vstress)
                    running_vloss += vloss

                avg_vloss = running_vloss / test_data_count
            else:
                avg_vloss = avg_loss

            # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.8f}, Test metric: {avg_vloss:.8f}')
            # print("psi = ", self.model.get_potential())
            if avg_vloss < best_vloss:
                best_epoch = epoch
                print("------------------------------------------------------------------")
                print("psi = ", self.model.get_potential())
                best_vloss = avg_vloss

            if epoch - best_epoch > 50 and best_vloss - avg_vloss < 10e-2: break

            # elif epoch % 100 == 0:
            #     print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.8f}, Test metric: {avg_vloss:.8f}')
            #     # print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}')
            #     # model_path = '{}_{}'.format(self.timestamp, epoch)
            #     # path_to_save_weights = os.path.join(self.path_to_save_weights, model_path + ".pth")
            #     # print(f"Saved PyTorch Model State to {path_to_save_weights}")
            #     # torch.save(self.model.state_dict(), path_to_save_weights)
            #     print("psi = ", self.model.get_potential())
            if epoch == (epochs - 1):
                print("psi = ", self.model.get_potential())
            loss_history.append(avg_loss)
            # epoch_number += 1

        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

        model_path = '{}_{}'.format(self.timestamp, best_epoch)
        path_to_save_weights = os.path.join(self.path_to_save_weights, model_path + ".pth")
        print(f"Saved PyTorch Model State to {path_to_save_weights}")
        torch.save(self.model.state_dict(), path_to_save_weights)
        self.path_to_best_weights = path_to_save_weights
        self.model.load_state_dict(torch.load(self.path_to_best_weights))
        # p rint(self.path_to_best_weights)

        # if self.plot_valid:
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(vpredictions, label='Ppred', color='red')
        #     plt.plot(vtargets, label='P_true', color='black')
        #     plt.xlabel('lambda/gamma')
        #     plt.ylabel('P')
        #     plt.title('Predictions vs. Targets')
        #     plt.legend()
        #     plt.show()

        return self.model

    def test(self, val_loader: DataLoader):
        """
        Validate the model on the validation dataset.

        Parameters:
        - val_loader (DataLoader): DataLoader object for validation data.
        """
        self.model.eval()
        val_loss = 0.0
        for data in val_loader:

            inputs, labels = data
            inputs, labels = inputs, labels

            outputs = self.model(inputs)
            loss = nn.MSELoss()(outputs, labels)
            val_loss += loss.item()

        # print(f'Validation loss: {val_loss / len(val_loader):.3f}')
        self.model.train()  # Return model to training mode
        return val_loss

    def load_data(self,
                  path_to_exp_names: str,
                  transform: Optional[object] = normalize_data,
                  shuffle: bool = True,
                  length_start: Optional[int] = None,
                  length_end: Optional[int] = None
                  ):

        dataset = ExcelDataset(
                           path=path_to_exp_names,
                           transform=transform,
                           device=self.device,
                           batch_size=self.batch_size
        )

        dataset.to_tensor()
        if length_end is not None:
            dataset.data = dataset.data[length_start:length_end]

        dataset_loader = DataLoader(
                                dataset,
                                batch_size=self.batch_size,
                                shuffle=shuffle,
                                num_workers=1,
                                pin_memory=False
        )

        return dataset_loader

    def visualize_predictions(self, data: pd.DataFrame):
        """
        Visualize dataset and predictions.

        Parameters:
        - experiment_col (str): The column name for the experiment identifier.
        - x_col (int or str): The column name or index for the x-axis data.
        - y_col (int or str): The column name or index for the y-axis data.
        """
        self.model.eval()
        vpredictions = []
        vtargets = []
        for data in test_data_loader:
            features, target = data
            vpredictions.append(trained_model(features).detach().numpy())
        print(trained_model.get_potential())
        combined_data["P_model"] = vpredictions


        # Преобразуем столбец с предсказанной силой в числовой формат
        data['P_model'] = data['P_model'].apply(lambda x: float(str(x).strip('[]')))

        # Создадим графики для каждого типа эксперимента
        experiment_types = data['experiment_type'].unique()

        def plot_with_r2(data, experiment_types):
            r2_scores = []
            fig, axes = plt.subplots(1, len(experiment_types), figsize=(15, 6), sharey=True)

            for ax, experiment in zip(axes, experiment_types):
                subset = data[data['experiment_type'] == experiment]
                r2 = r2_score(subset['P_experimental'], subset['P_model'])

                sns.scatterplot(data=subset, x='lambda', y='P_experimental', label='P_experimental', ax=ax)
                sns.lineplot(data=subset, x='lambda', y='P_model', label='P_model', color='orange', ax=ax)
                ax.set_title(f'Experiment Type: {experiment}\nR² = {r2:.2f}')
                ax.set_xlabel('Strain')
                ax.set_ylabel('Force (kPa)')
                r2_scores.append(r2)

            plt.tight_layout()
            plt.show()
            return r2_scores

        # Вызовем функцию для построения графиков с r2
        plot_with_r2(data, experiment_types)


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

    get_list_of_paths_to_experiments_type("biaxial")

    experiment = "biaxial"
    experiments_path = get_list_of_paths_to_experiments_type(experiment)
    data_frames = [load_and_extract(file, file[-11:-4]) for file in experiments_path]

    # print(data_frames)

    df = pd.concat(data_frames, ignore_index=True)
    thinned_data_frames = []
    num_points = -1

    df_list = []

    for df in data_frames:
        if num_points != -1:
            indices = np.linspace(10, len(df) - 1, num_points, dtype=int)
            # df[1] = df[1] / 10**6
            df = pd.DataFrame(df.iloc[indices].copy())
        # print(type(sampled_df))
        df['lambdas'] = list(zip(df['lambda_clamps_X'], df['lambda_clamps_Y']))
        df['stresses'] = list(zip(df['mean_stress_x_mpa'], df['mean_stress_y_mpa']))
        df_list.append(df[:50])
        # df_list.append(df)
        # thinned_df = df.iloc[::len(df) // 20, :]  # Выбор каждого 45-го значения
        # thinned_data_frames.append(thinned_df)
    data_frames = df_list
    # df.iloc[40:60]
    print(df_list[1])
    # thinned_data_frames
    # sampled_df_list[0]
    # for item in data_frames:
    # print(item)

if __name__ == "__main__":
    main()