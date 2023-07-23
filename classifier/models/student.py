import json
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
from datasets import Dataset, WebrequestDataset
from networks import cnn
from torch.utils.data import DataLoader

from . import Model


class Student(Model):
    def __init__(
        self,
        teachers_exp_name: str,
        lr: float,
        batch_size: int,
        train_labels: np.ndarray,
        dataset: Dataset = WebrequestDataset,
        network_fn: Callable = cnn,
    ):
        super().__init__(lr, batch_size, dataset, network_fn)
        self.teachers_exp_name = teachers_exp_name
        self.train_labels = train_labels
        self._loss = torch.nn.CrossEntropyLoss()

    def prepare_data(self):
        self.data.prepare_data()

    def save_weights(self, exp_name: str, exp_config: Dict):
        path = Path(f"classifier/weights/students/{exp_name}")
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), f"{path}/student")

    def loss(self, y_hat, y):
        return self._loss(y_hat, y)

    def train_dataloader(self):
        return self.data.get_student_train_dataloader(self.batch_size)

    def val_dataloader(self):
        return self.data.get_student_val_dataloader(self.batch_size)
