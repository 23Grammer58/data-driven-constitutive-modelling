import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional

from models.CNN import StrainEnergyCANN, StrainEnergyCANN_C
from utils.dataload import ExcelDataset, normalize_data
from utils.visualisation import *


class Trainer:
    def __init__(self,
                 experiment_name: Optional[str] = "test",
                 model: Optional[nn.Module] = StrainEnergyCANN_C,
                 device: Optional[str] = "cpu",
                 learning_rate: Optional[float] = 0.1,
                 epochs: Optional[int] = 100,
                 plot_valid: Optional[bool] = False,
                 batch_size: Optional[int] = 1
                 ):
        """
        A class for training CANN model.

        Attributes:
            experiment_name (str): Name of the experiment. Defaults to "test".
            model (nn.Module): The neural network model to be trained. Defaults to `StrainEnergyCANN_C`.
            device (str): The device to run the training on. Can be "cpu" or "cuda". Defaults to "cpu".
            learning_rate (float): The learning rate for the optimizer. Defaults to 0.1.
            epochs (int): The number of epochs to train for. Defaults to 100.
            plot_valid (bool): Whether to plot the validation loss. Defaults to False.
            batch_size (int): The batch size for training. Defaults to 1.
            timestamp (str): The timestamp of the training session.
            writer (SummaryWriter): The TensorBoard writer for logging.
        """

        # self.input_size = input_size
        # self.output_size = output_size
        # self.hidden_size = hidden_size
        self.model = model(batch_size, device=device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.experiment_name = experiment_name
        self.plot_valid = plot_valid
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('runs/fashion_trainer_{}'.format(self.timestamp))

    def load_data(self,
                  path_to_exp_names,
                  batch_size=1,
                  transform=normalize_data,
                  shuffle=True,
                  length_start=None,
                  length_end=None):

        dataset = ExcelDataset(path=path_to_exp_names, transform=transform, device=self.device)
        if length_end is not None:
            dataset.data = dataset.data[length_start:length_end]



        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=False)
        return dataset_loader

    def train(self, train_loader, test_loader):
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

        def train_one_epoch(epoch_index, l2_reg_coeff=0.1, l1_reg_coeff=0.1):
            running_loss = 0.
            last_loss = 0.

            last_data = len(train_loader)
            for i, data in enumerate(train_loader):
                F, invariants, targets, exp_type = data
                F = F.reshape(-1, 3)
                inputs = (F, invariants, exp_type)
                targets = targets.reshape(-1, 3)

                optimizer.zero_grad()
                stress_model = self.model(inputs)

                loss = loss_fn(stress_model, targets)

                l1_reg = self.model.calc_l1()
                l2_reg = self.model.calc_regularization()
                if l2_reg is not None:
                    loss = loss + l2_reg_coeff * l2_reg + l1_reg_coeff * l1_reg

                loss.backward(retain_graph=True)
                optimizer.step()

                tb_x = epoch_index * len(train_loader) + i + 1
                self.writer.add_scalar('Loss/train', loss.item(), tb_x)
                last_loss = loss.item()
                running_loss += loss.item()

            return running_loss / last_data

        epoch_number = 0
        best_vloss = torch.inf
        elosses = []
        vlosses = []
        predictions_P11 = []
        predictions_P12 = []
        targets_P11 = []
        targets_P12 = []

        # Training the model
        for epoch in tqdm(range(self.epochs)):
            self.model.train(True)
            avg_loss = train_one_epoch(epoch_number)

            running_vloss = 0.0
            self.model.eval()

            with torch.no_grad():
                for i, vdata in enumerate(test_loader):
                    F, invariants, target, exp_type = vdata
                    F = F.reshape(-1, 3)
                    vinputs = (F, invariants)
                    vtargets = target.reshape(-1, 3)
                    targets_P11.append(vtargets[0, 0])
                    targets_P12.append(vtargets[0, 1])

                    voutputs = self.model(vinputs).to(self.device)
                    predictions_P11.append(voutputs[0, 0])
                    predictions_P12.append(voutputs[0, 1])
                    vloss = loss_fn(voutputs, vtargets)
                    running_vloss += vloss
                    vlosses.append(vloss)

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            model_path = '{}_{}'.format(self.timestamp, epoch_number)
            path_to_save_weights = os.path.join("pretrained_models", self.experiment_name)
            path_to_save_weights = os.path.join(path_to_save_weights, model_path + ".pth")

            if avg_loss < best_vloss or epoch % 10 == 0:
                best_vloss = avg_loss
                print(f"Saved PyTorch Model State to {path_to_save_weights}")
                torch.save(self.model.state_dict(), model_path)
                print(self.model.get_potential())

            elosses.append(avg_loss)
            epoch_number += 1
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}')

        plt.plot(elosses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
        if self.plot_valid:
            plt.figure(figsize=(10, 5))
            plt.plot(predictions_P11, label='P11_pred', color='blue')
            plt.plot(predictions_P12, label='P12_pred', color='red')
            plt.plot(targets_P11, label='P11', color='black')
            plt.plot(targets_P12, label='P12', color='gray')
            plt.xlabel('lambda/gamma')
            plt.ylabel('P')
            plt.title('Predictions vs. Targets')
            plt.legend()
            plt.show()

        return self.model


if __name__ == "__main__":
    data_path = r"C:\Users\drani\dd\data-driven-constitutive-modelling\data\brain_bade\CANNsBRAINdata.xlsx"

    test_train = Trainer(plot_valid=True)
    data_loader = test_train.load_data(data_path, transform=None)
    trained_model = test_train.train(data_loader, data_loader)
