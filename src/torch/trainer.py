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
                 learning_rate: Optional[float] = 0.001,
                 epochs: Optional[int] = 100,
                 plot_valid: Optional[bool] = False,
                 batch_size: Optional[int] = 1,
                 l1_reg_coeff: Optional[float] = 0.001,
                 l2_reg_coeff: Optional[float] = 0.001
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

        self.l1_reg_coeff = l1_reg_coeff
        self.l2_reg_coeff = l2_reg_coeff
        self.model = model(batch_size, device=device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.experiment_name = experiment_name
        self.plot_valid = plot_valid
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('runs/fashion_trainer_{}'.format(self.timestamp))
        self.path_to_save_weights = os.path.join("pretrained_models", self.experiment_name)
        self.batch_size = batch_size

    def load_data(self,
                  path_to_exp_names,
                  transform=normalize_data,
                  shuffle=True,
                  length_start=None,
                  length_end=None):

        dataset = ExcelDataset(
                               path=path_to_exp_names,
                               transform=transform,
                               device=self.device,
                               batch_size=self.batch_size)

        dataset.to_tensor()
        if length_end is not None:
            dataset.data = dataset.data[length_start:length_end]

        dataset_loader = DataLoader(
                                    dataset,
                                    batch_size=self.batch_size,
                                    shuffle=shuffle,
                                    num_workers=1,
                                    pin_memory=False)
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
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)

        def train_one_epoch(epoch_index):
            running_loss = 0.
            last_loss = 0.

            last_data = len(train_loader)
            for i, data in enumerate(train_loader):
                features, target = data
                _, _, _, _, exp_type = features
                # inputs = (F, i1, i2, exp_type)

                optimizer.zero_grad()
                stress_model = self.model(features)

                loss = loss_fn(stress_model, target) * exp_type

                # print(loss.item())
                # l1_reg = self.model.calc_l1()
                # l2_reg = self.model.calc_regularization(2)
                # if l2_reg is not None:
                #     loss = loss + 0.5 * self.l2_reg_coeff * l2_reg + self.l1_reg_coeff * l1_reg

                # loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()

                tb_x = epoch_index * len(train_loader) + i + 1
                self.writer.add_scalar('Loss/train', loss.item(), tb_x)
                # last_loss = loss.item()
                running_loss += loss.item()

            return running_loss

        epoch_number = 0
        best_vloss = torch.inf
        elosses = []
        vlosses = []
        vpredictions = []
        vtargets = []

        # Training the model
        for epoch in tqdm(range(self.epochs)):
            self.model.train(True)
            avg_loss = train_one_epoch(epoch_number)

            running_vloss = 0.0
            self.model.eval()

            validation = False
            if validation:
                with torch.no_grad():
                    for i, vdata in enumerate(test_loader):
                        vfeatures, vtarget = vdata

                        optimizer.zero_grad()
                        # stress_model = self.model(vfeatures)
                        vstress = self.model(vfeatures)
                        vloss = loss_fn(vtarget, vstress)
                        running_vloss += vloss

                        # targets_P11.append(vtarget[0, 0])
                        # targets_P12.append(vtarget[0, 1])
                        # predictions_P11.append(vstress[0, 0])
                        # predictions_P12.append(vstress[0, 1])

                        vlosses.append(vloss)
                        vpredictions.append(vstress)
                        vtargets.append(vtarget)
            else:
                running_vloss = avg_loss
            avg_vloss = running_vloss / 66
            print()
            # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}')
            if avg_loss < best_vloss:
                best_vloss = avg_loss
                model_path = '{}_{}'.format(self.timestamp, epoch)
                path_to_save_weights = os.path.join(self.path_to_save_weights, model_path + ".pth")
                print(f"Saved PyTorch Model State to {path_to_save_weights}")
                torch.save(self.model.state_dict(), path_to_save_weights)
                print(self.model.get_potential())

            elif epoch % (self.epochs // 100) == 0:
                # print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}')
                model_path = '{}_{}'.format(self.timestamp, epoch_number)
                path_to_save_weights = os.path.join(self.path_to_save_weights, model_path + ".pth")
                print(f"Saved PyTorch Model State to {path_to_save_weights}")
                torch.save(self.model.state_dict(), path_to_save_weights)
            elosses.append(avg_loss)
            # epoch_number += 1
            print(self.model.get_potential())

        plt.plot(elosses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
        if self.plot_valid:
            plt.figure(figsize=(10, 5))
            plt.plot(vpredictions, label='Ppred', color='red')
            plt.plot(vtargets, label='P_true', color='black')
            plt.xlabel('lambda/gamma')
            plt.ylabel('P')
            plt.title('Predictions vs. Targets')
            plt.legend()
            plt.show()

        return self.model


def main():
    data_path = r"C:\Users\drani\dd\data-driven-constitutive-modelling\data\brain_bade\CANNsBRAINdata.xlsx"

    test_train = Trainer(plot_valid=True, epochs=10000, experiment_name="FIRST_weights")
    train_data_loader = test_train.load_data(data_path, transform=None)
    test_data_loader = test_train.load_data(data_path, transform=None)
    trained_model = test_train.train(train_data_loader, test_data_loader)


if __name__ == "__main__":
    main()