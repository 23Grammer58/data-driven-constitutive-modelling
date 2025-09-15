import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from datetime import datetime
import os
import matplotlib.pyplot as plt
from typing import Optional, Union
from sklearn.metrics import r2_score
from CANN_torch.models import ModelArchitecture_I5, ModelArchitecture_I2

from CANN_torch.models import *
# from models.CNN import *
# from models.CNN import StrainEnergyCANN, StrainEnergyCANN_C, StrainEnergyCANN_polinomial3
# from utils.dataload import ExcelDataset, normalize_data
# from utils.visualisation import *
import seaborn as sns
import pandas as pd

# from models.CANN import *

def compute_loss_weighted(loss_fn, stress_model, target, exp_type, weight, weighting_data=True):
    """
    Вычисляет взвешенный лосс для батча экспериментов с учетом типа эксперимента.

    Предположения:
      - stress_model имеет форму (2, batch_size).
      - target имеет форму (batch_size, 2) для биаксиальных экспериментов и (batch_size, 1) для унииальных.
      - exp_type – тензор формы (batch_size,). Принято, что унииальные эксперименты помечаются значением 1000, а биаксиальные – любым другим.
      - weight – тензор формы (batch_size,), содержащий вес для каждого образца (предварительно добавленный в preproc_data).
      - Если weighting_data=False, используется вес 1 для всех образцов.

    Лосс для образцов с унииальным экспериментом (exp_type == 1000) вычисляется только по первому каналу,
    а для биаксиальных – суммируется лосс по обоим каналам.

    Итоговый лосс вычисляется как сумма (взвешенных) лоссов по всем образцам, деленная на сумму весов.
    """
    # Приводим выход модели к форме (batch_size, 2)
    stress_model_T = stress_model  # shape: (batch_size, 2)

    # Определяем маски для унииальных (например, exp_type == 1000) и биаксиальных экспериментов
    # mask_uni = (exp_type == "uni")
    # mask_biaxial = (exp_type != 1000)
    #
    mask_uni = torch.tensor([x == "uni" for x in exp_type])
    mask_biaxial = torch.tensor([x != "uni" for x in exp_type])

    loss_total = 0.0
    total_weight = 0.0

    # Если взвешивание не требуется, подставляем единичный вес
    ones_uni = torch.ones_like(exp_type[mask_uni], dtype=stress_model_T.dtype) if mask_uni.sum() > 0 else None
    ones_biaxial = torch.ones_like(exp_type[mask_biaxial],
                                   dtype=stress_model_T.dtype) if mask_biaxial.sum() > 0 else None

    # Обработка унииальных экспериментов (exp_type == 1000)
    if mask_uni.sum() > 0:
        # Выбираем для каждого образца первый канал (единственный релевантный)
        out_uni = stress_model_T[mask_uni, 0]  # shape: (N_uni,)
        target_uni = target[mask_uni, 0]  # shape: (N_uni,)
        loss_uni = loss_fn(out_uni, target_uni)  # (N_uni,)
        sample_weight_uni = weight[mask_uni] if weighting_data else ones_uni
        loss_total += (loss_uni * sample_weight_uni).sum()
        total_weight += sample_weight_uni.sum()

    # Обработка биаксиальных экспериментов (exp_type != 1000)
    if mask_biaxial.sum() > 0:
        out_xx = stress_model_T[mask_biaxial, 0]  # (N_biaxial,)
        out_yy = stress_model_T[mask_biaxial, 1]  # (N_biaxial,)
        target_xx = target[mask_biaxial, 0]  # (N_biaxial,)
        target_yy = target[mask_biaxial, 1]  # (N_biaxial,)
        loss_xx = loss_fn(out_xx, target_xx)
        loss_yy = loss_fn(out_yy, target_yy)
        loss_biaxial = loss_xx + loss_yy  # (N_biaxial,) – суммируем лосс по обоим каналам для каждого образца
        sample_weight_biaxial = weight[mask_biaxial] if weighting_data else ones_biaxial
        loss_total += (loss_biaxial * sample_weight_biaxial).sum()
        total_weight += sample_weight_biaxial.sum()

    return loss_total / total_weight if total_weight > 0 else loss_total


def compute_loss(loss_fn, stress_model, target, exp_type):
    """
    Вычисляет лосс для батча, содержащего образцы с разными типами экспериментов.

    Предположения:
      - stress_model имеет форму (2, batch_size)
      - target имеет форму (batch_size, 2) для образцов с exp_type != 1000,
        и (batch_size, 1) или (batch_size, 2) для образцов с exp_type == 1000 (в этом случае используется только первый канал)
      - exp_type имеет форму (batch_size,)

    Для образцов с exp_type == 1000 вычисляется лосс только по первому каналу,
    для остальных – суммируются лоссы по обоим каналам.

    Итоговый лосс усредняется по общему числу элементов (1 элемент на образец для exp_type == 1000 и 2 элемента на образец для остальных).
    """
    # mask_1000 = (exp_type == 1000)
    # mask_other = (exp_type != 1000)

    mask_1000 = torch.tensor([x == "uni" for x in exp_type])
    mask_other = torch.tensor([x != "uni" for x in exp_type])
    loss_total = 0.0
    n_elements = 0

    # Приводим stress_model к форме (batch_size, 2)
    stress_model_T = stress_model  # теперь stress_model_T имеет форму (batch_size, 2)

    # Обработка для образцов с exp_type == 1000 (одноканальный лосс)
    if mask_1000.sum() > 0:
        out_1000 = stress_model_T[mask_1000, 0]  # выбираем первый канал, форма (N_1000,)
        target_1000 = target[mask_1000, 0]  # форма (N_1000,)
        loss_1000 = loss_fn(out_1000, target_1000)
        loss_total += loss_1000.sum()
        n_elements += mask_1000.sum()

    # Обработка для образцов с exp_type != 1000 (двухканальный лосс)
    if mask_other.sum() > 0:
        out_xx = stress_model_T[mask_other, 0]  # форма (N_other,)
        out_yy = stress_model_T[mask_other, 1]  # форма (N_other,)
        target_xx = target[mask_other, 0]  # форма (N_other,)
        target_yy = target[mask_other, 1]  # форма (N_other,)
        loss_xx = loss_fn(out_xx, target_xx)
        loss_yy = loss_fn(out_yy, target_yy)
        loss_total += (loss_xx + loss_yy).sum()
        n_elements += 2 * mask_other.sum()

    loss = loss_total / n_elements
    return loss


def r2_score_own(Truth, Prediction):
    R2 = r2_score(Truth,Prediction)
    return max(R2,0.0)


class Trainer:
    def __init__(self,
                 checkpoint: str = None,
                 experiment_name: Optional[str] = "test",
                 model: Union[nn.Module, type] = ModelArchitecture_I5,
                 device: Optional[str] = "cpu",
                 learning_rate: float = 0.001,
                 epochs: int = 1000,
                 plot_valid: bool = False,
                 batch_size: int = 1,
                 l1_reg_coeff: Optional[float] = 0.001,
                 l2_reg_coeff: Optional[float] = 0.001,
                 dtype = torch.float32,
                 initial_weight = 0.1,
                 SingleInvNet = None
                 ):
        """
        Класс для обучения CANN моделей.

         Аргументы:
            - `experiment_name` (str): Название эксперимента. По умолчанию "test".
            - `model` (nn.Module): Архитектура модели для обучения. По умолчанию `StrainEnergyCANN_C`.
            - `path_to_save_weights` (str): Путь для сохранения весов модели.
            - `epochs` (int): Количество эпох обучения. По умолчанию 1000.
            - `learning_rate` (float): Скорость обучения. По умолчанию 0.001.
            - `plot_valid` (bool): Отрисовка ошибки на валидационном датасете. По умолчанию False.
            - `l1_reg_coeff` (float): Коэффициент регуляризации L1. По умолчанию 0.001.
            - `l2_reg_coeff` (float): Коэффициент регуляризации L2. По умолчанию 0.001.
            - `checkpoint` (str): Путь до весов модели, с которыми модель инициализируется. По умолчанию 1.
            - `device` (str): Устройство для выполнения вычислений (cpu или cuda). По умолчанию "cpu".

        """

        self.l1_reg_coeff = l1_reg_coeff
        self.l2_reg_coeff = l2_reg_coeff

        if not SingleInvNet:
            SingleInvNet = SingleInvNet4

        # ----------------------------------------------------------------------------
        # 1. Если передан ГОТОВЫЙ экземпляр модели → используем без изменений
        # ----------------------------------------------------------------------------
        if isinstance(model, nn.Module):
            self.model = model
        else:
            # ----------------------------------------------------------------------------
            # 2. Иначе работаем по старой логике, ожидая класс модели
            # ----------------------------------------------------------------------------
            if model == ModelArchitecture_I5:
                psi_model = StrainEnergy_i5(SingleInvNet=SingleInvNet)
                self.model = model(psi_model, setAl=True, init=torch.pi / 2, initial_weight=initial_weight)
            elif model == ModelArchitecture_I2:
                psi_model = BaseStrainEnergy(SingleInvNet)
                self.model = model(psi_model)
            else:
                psi_model = BaseStrainEnergy(SingleInvNet)
                self.model = model(psi_model)

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

        train_data_count = len(train_loader) * self.batch_size

        if test_loader is None:
            test_data_count = train_data_count
        else:
            test_data_count = len(test_loader)
        def train_one_epoch(epoch_index):
            epoch_loss = 0.
            last_loss = 0.

            for i, data in enumerate(train_loader):
                features, target = data
                exp_type = features[-2]
                weight = features[-1]
                # _, _, i1, i2, i4, i5, _, exp_type = features

                optimizer.zero_grad()
                stress_model = self.model(features)

                loss = compute_loss(loss_fn, stress_model, target, exp_type)

                # if weighting_data:
                #     if exp_type == "Compression":
                #         loss *= 0.5
                #     elif exp_type == "Tensile":
                #         loss *= 1.5

                if self.l2_reg_coeff is not None:
                    l2_reg = self.model.calc_regularization(2)
                    loss += 0.5 * self.l2_reg_coeff * l2_reg

                if self.l1_reg_coeff is not None:
                    l1_reg = self.model.calc_regularization(1)
                    loss += self.l1_reg_coeff * l1_reg

                loss.backward(retain_graph=True)

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

        # Training the model
        for epoch in range(self.epochs):
            self.model.train(True)
            avg_loss = train_one_epoch(epoch_number) / train_data_count
            # scheduler.step()

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

            # print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.8f}, Test metric: {avg_vloss:.8f}')
            # print("psi = ", self.model.get_potential())
            if avg_vloss < best_vloss:
                best_epoch = epoch
                # print("psi = ", self.model.get_potential())
                # print("------------------------------------------------------------------")
                best_vloss = avg_vloss

            if epoch - best_epoch > 300 and abs(best_vloss - avg_vloss) < 10e-5: break

            elif epoch % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.8f}, Test metric: {avg_vloss:.8f}')
                print("psi = ", self.model.get_potential())
                print("------------------------------------------------------------------")

            #     print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.8f}, Test metric: {avg_vloss:.8f}')
            #     # print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}')
            #     model_path = '{}_{}'.format(self.timestamp, epoch)
            #     print(epoch)
            #     print(self.model.state_dict().values())
            #     print(self.model.get_potential())
            #     print()
            #     # path_to_save_weights = os.path.join(self.path_to_save_weights, model_path + ".pth")
            #     # print(f"Saved PyTorch Model State to {path_to_save_weights}")
            #     # torch.save(self.model.state_dict(), path_to_save_weights)
            #     print("psi = ", self.model.get_potential())
            # if epoch == (epochs - 1):
            #     print("psi = ", self.model.get_potential())
            loss_history.append(avg_loss)
            # epoch_number += 1

        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.yscale('log')
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


# if __name__ == "__main__":
#   main()