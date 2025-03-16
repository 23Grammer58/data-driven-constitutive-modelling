

from models.CNN import StrainEnergyCANN_C

from trainer import Trainer
import argparse
import logging
import sys
import tempfile
from argparse import Namespace
from typing import List, Tuple, get_args

import torch
import torch.nn as nn
from torcheval.metrics import R2Score
from torchtnt.framework.state import State
from torchtnt.framework.train import train
from torchtnt.framework.unit import TrainUnit
from torchtnt.utils import copy_data_to_device, init_from_env, seed, TLRScheduler
from torchtnt.utils.loggers import TensorBoardLogger
from torchnet.logger import VisdomLogger


_logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
Batch = Tuple[torch.tensor, torch.tensor]
# specify type of the data in each batch of the dataloader to allow for typechecking

class CANNTrainUnit(TrainUnit[Batch]):
    def __init__(
        self,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        train_accuracy: R2Score,
        tb_logger: TensorBoardLogger,
        log_every_n_steps: int
    ):
        super().__init__()
        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

        # create an accuracy Metric to compute the accuracy of training
        self.train_accuracy = train_accuracy
        self.log_every_n_steps = log_every_n_steps

        self.tb_logger = tb_logger

    def train_step(self, state: State, data: Batch) -> None:
        data = copy_data_to_device(data, self.device)

        inputs, targets = data
        outputs = self.module(inputs)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, targets) * inputs[-1]
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.train_accuracy.update(outputs, targets)
        step_count = self.train_progress.num_steps_completed
        if (step_count + 1) % self.log_every_n_steps == 0:
            acc = self.train_accuracy.compute()
            self.tb_logger.log("loss", loss, step_count)
            self.tb_logger.log("R2", acc, step_count)

    def on_train_epoch_end(self, state: State) -> None:
        # compute and log the metric at the end of the epoch
        step_count = self.train_progress.num_steps_completed
        acc = self.train_accuracy.compute()
        self.tb_logger.log("accuracy_epoch", acc, step_count)

        # reset the metric at the end of every epoch
        self.train_accuracy.reset()

        # step the learning rate scheduler
        self.lr_scheduler.step()



def main(argv: List[str]) -> None:
    args = get_args(argv)

    device = "cpu"
    path = tempfile.mkdtemp()
    tb_logger = TensorBoardLogger(path)

    data_path = r"C:\Users\drani\dd\data-driven-constitutive-modelling\data\brain_bade\CANNsBRAINdata.xlsx"

    test_train = Trainer(plot_valid=True, epochs=6000, experiment_name="FIRST_weights", l2_reg_coeff=0.001)
    train_dataloader = test_train.load_data(data_path, transform=None)
    # test_data_loader = test_train.load_data(data_path, transform=None, shuffle=False)

    # trained_model = test_train.train(train_data_loader, test_data_loader)
    model = StrainEnergyCANN_C(1, device="cpu")
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
    train_accuracy = R2Score(device=device)

    my_unit = CANNTrainUnit(
        model,
        optimizer,
        lr_scheduler,
        device,
        train_accuracy,
        tb_logger,
        2,
    )

    train(
        my_unit,
        train_dataloader=train_dataloader,
        max_epochs=10,
    )


if __name__ == "__main__":
    main(sys.argv[1:])