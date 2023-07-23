from collections import OrderedDict
from typing import Callable, Dict

import numpy as np

import pytorch_lightning as pl
import torch
import wandb
from datasets import Dataset
from pytorch_lightning.metrics.functional import stat_scores
from torch import nn, optim
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(pl.LightningModule):
    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(
        self, lr: float, batch_size: int, dataset: Dataset, network_fn: Callable
    ):
        super().__init__()
        self.data = dataset
        self.network = network_fn(self.data.input_shape)
        self.lr = lr
        self.batch_size = batch_size

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=-1)
        acc = torch.mean((y == labels_hat).float())

        tensorboard_logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        labels_hat = torch.argmax(y_hat, dim=-1)

        val_acc = torch.mean((y == labels_hat).float())
        val_loss = self.loss(y_hat, y)
        output = OrderedDict({"val_acc": val_acc, "val_loss": val_loss})

        return output

    def validation_epoch_end(self, outputs):
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_acc": avg_acc, "val_loss": avg_loss}

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def evaluate(self):
        losses = np.array([]).astype(float)
        accuracies = np.array([]).astype(float)

        self.network.eval()
        with torch.no_grad():
            for x, y in self.val_dataloader():
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self(x)
                loss = self.loss(y_hat, y)
                labels_hat = torch.argmax(y_hat, dim=-1)

                loss = self.loss(y_hat, y)
                accuracy = torch.mean((y == labels_hat).float())
                losses = np.append(losses, loss.detach().item())
                accuracies = np.append(accuracies, accuracy.detach().item())

        avg_acc = accuracies.mean()
        avg_loss = losses.mean()
        return avg_acc, avg_loss

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def loss(self, y_hat, y):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def metrics(self):
        return ["accuracy"]

    def save_weights(self):
        pass
