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

def r2_score_own(Truth, Prediction):
    R2 = r2_score(Truth,Prediction)
    return max(R2,0.0)


class Trainer:
    def __init__(self,
                 checkpoint: str = None,
                 experiment_name: Optional[str] = "test",
                 model: nn.Module = StrainEnergyCANN_C,
                 device: Optional[str] = "cpu",
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 plot_valid: bool = False,
                 batch_size: int = 1,
                 l1_reg_coeff: Optional[float] = 0.001,
                 l2_reg_coeff: Optional[float] = 0.001,
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
        self.model = model(batch_size, device=device)
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
              train_loader=Optional[DataLoader],
              test_loader=Optional[DataLoader],
              weighting_data=True):
        # Initialize the model, loss function, and optimizer

        if self.experiment_name is not None:
            path_to_save_weights = os.path.join("pretrained_models", self.experiment_name)
            if not os.path.exists(path_to_save_weights):
                os.makedirs(path_to_save_weights)
                print(f"Directory {path_to_save_weights} created successfully")
            else:
                print(f"Directory {path_to_save_weights} already exists")

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        last_data = len(train_loader)
        def train_one_epoch(epoch_index):
            running_loss = 0.
            last_loss = 0.

            for i, data in enumerate(train_loader):
                features, target = data
                _, _, _, _, exp_type = features

                optimizer.zero_grad()
                stress_model = self.model(features)

                loss = loss_fn(stress_model, target)
                if weighting_data:
                    loss *= exp_type

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
                running_loss += loss.item()

            return running_loss / last_data

        epoch_number = 0
        best_vloss = torch.inf
        loss_history = []
        vlosses = []
        vpredictions = []
        vtargets = []

        # Training the model
        for epoch in range(self.epochs):
            self.model.train(True)
            avg_loss = train_one_epoch(epoch_number)

            running_vloss = 0.0

            # validation = False
            if test_loader:
                running_vloss = self.test(test_loader)
                # for i, vdata in enumerate(test_loader):
                #     vfeatures, vtarget = vdata
                #
                #     optimizer.zero_grad()
                #     # stress_model = self.model(vfeatures)
                #     vstress = self.model(vfeatures)
                #     vloss = loss_fn(vtarget, vstress)
                #     running_vloss += vloss

            else:
                running_vloss = avg_loss
            avg_vloss = running_vloss / last_data
            print()
            # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.8f}, Test metric: {avg_vloss:.8f}')
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = '{}_{}'.format(self.timestamp, epoch)
                path_to_save_weights = os.path.join(self.path_to_save_weights, model_path + ".pth")
                print(f"Saved PyTorch Model State to {path_to_save_weights}")
                torch.save(self.model.state_dict(), path_to_save_weights)
                self.path_to_best_weights = path_to_save_weights

                print("psi = ", self.model.get_potential())

            elif epoch % 100 == 0:
                # print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}')
                model_path = '{}_{}'.format(self.timestamp, epoch)
                path_to_save_weights = os.path.join(self.path_to_save_weights, model_path + ".pth")
                print(f"Saved PyTorch Model State to {path_to_save_weights}")
                torch.save(self.model.state_dict(), path_to_save_weights)
                print("psi = ", self.model.get_potential())
            loss_history.append(avg_loss)
            # epoch_number += 1

        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

        self.model.load_state_dict(torch.load(self.path_to_best_weights))
        print(self.path_to_best_weights)

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

    def visualize_predictions(self, dataset, predictions, experiment_col='Experiment', x_col=0, y_col=1):
        """
        Visualize dataset and predictions.

        Parameters:
        - experiment_col (str): The column name for the experiment identifier.
        - x_col (int or str): The column name or index for the x-axis data.
        - y_col (int or str): The column name or index for the y-axis data.
        """

        # Set seaborn style
        sns.set(style="whitegrid")

        # Create a combined plot of the dataset
        g = sns.relplot(
            data=dataset,
            x=x_col, y=y_col, col=experiment_col, kind='line', height=4, aspect=1.2,
            facet_kws={'sharey': False, 'sharex': False}
        )

        # Add predictions to the plot
        for ax, (exp, group) in zip(g.axes.flat, predictions.groupby(experiment_col)):
            ax.plot(group[x_col], group[y_col], label='Prediction', linestyle='--')
            ax.legend()

        plt.show()


def main():
    data_path = r"C:\Users\Biomechanics\PycharmProjects\dd\data-driven-constitutive-modelling\data\braid_bade\CANNsBRAINdata.xlsx"
    best_model_path = r"C:\Users\Biomechanics\PycharmProjects\dd\data-driven-constitutive-modelling\src\CANN_torch\pretrained_models\FIRST_weights\20240516_194300_147.pth"
    test_train = Trainer(
        plot_valid=False,
        epochs=10,
        experiment_name="FIRST_weights",
        l2_reg_coeff=0.01,
        learning_rate=0.001,
        checkpoint=None,

    )

    train_data_loader = test_train.load_data(data_path, transform=None)
    test_data_loader = test_train.load_data(data_path, transform=None, shuffle=False)
    trained_model = test_train.train(train_data_loader, test_data_loader, weighting_data=False)
    # trained_model = StrainEnergyCANN_C()
    # trained_model.load_state_dict(torch.load(best_model_path))
    trained_model.eval()
    vpredictions = []
    vtargets = []
    for data in test_data_loader:
        features, target = data
        # if features[-1] == 1.5:
        vpredictions.append(trained_model(features).detach().numpy())
        vtargets.append(target.detach().numpy())
    print(trained_model.get_potential())
    print(r2_score_own(vtargets, vpredictions))
    plt.figure(figsize=(10, 5))
    plt.plot(vpredictions, label='P_pred', color='red')
    plt.plot(vtargets, label='P_true', color='black')
    plt.xlabel('lambda/gamma')
    plt.ylabel('P')
    plt.title('Predictions vs. Targets')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()