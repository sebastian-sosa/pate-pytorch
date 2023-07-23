import numpy as np

import torch
from torch.utils.data import DataLoader

from . import Aggregator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def laplace_randomizer(row, scale):
    scale = 1 / scale
    noise = np.random.laplace(0, scale, row.shape)
    return row + noise


class LaplaceAggregator(Aggregator):
    def __init__(self, gamma: float, classes_count: int, student_data_eps: float = 0):
        self.gamma = gamma
        self.classes_count = classes_count
        self.student_data_eps = student_data_eps

    def apply_noise_to_samples(self, samples: np.ndarray) -> torch.Tensor:
        if self.student_data_eps == 0:
            return samples
        return torch.Tensor(
            np.apply_along_axis(
                laplace_randomizer, axis=1, arr=samples, scale=self.student_data_eps
            )
        )

    def aggregate_teacher_preds(self, teacher_preds: np.ndarray) -> np.ndarray:
        labels = np.array([]).astype(int)

        for sample_preds in np.transpose(teacher_preds):
            label_counts = np.bincount(
                sample_preds, minlength=self.classes_count
            ).astype(float)
            label_counts += np.random.laplace(0, 1 / self.gamma, self.classes_count)

            new_label = np.argmax(label_counts)
            labels = np.append(labels, new_label)

        return labels
