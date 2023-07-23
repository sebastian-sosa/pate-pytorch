import json
from pathlib import Path
from typing import Callable, Dict

import torch

from datasets import Dataset, MnistDataset
from networks import cnn

from . import Model


class Teacher(Model):
    def __init__(
        self,
        teacher_idx: int,
        num_teachers: int,
        lr: float,
        batch_size: int,
        dataset: Dataset = MnistDataset,
        network_fn: Callable = cnn,
    ):
        super().__init__(lr, batch_size, dataset, network_fn)
        self.teacher_idx = teacher_idx
        self.num_teachers = num_teachers
        self._loss = torch.nn.BCELoss()

    def prepare_data(self):
        self.data.prepare_data()

    def save_weights(self, exp_name: str, exp_config: Dict):
        path = Path(f"classifier/weights/teachers/{exp_name}")
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), f"{path}/teacher_{self.teacher_idx}")

    def loss(self, y_hat, y):
        return self._loss(y_hat, y)

    def train_dataloader(self):
        return self.data.get_teacher_train_dataloader(
            self.teacher_idx, self.num_teachers, self.batch_size
        )

    def val_dataloader(self):
        return self.data.get_teacher_val_dataloader(self.batch_size)
