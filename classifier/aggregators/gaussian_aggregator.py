import numpy as np
import torch
from torch.utils.data import DataLoader

from . import Aggregator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GaussianAggregator(Aggregator):
    def __init__(self, sigma1: float, sigma2: float, threshold: float = 0.7):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.threshold = threshold

    def aggregate_teacher_preds(self, teacher_preds: np.ndarray) -> np.ndarray:
        labels = np.array([]).astype(int)
        num_teachers = teacher_preds.shape[0]
        num_samples = teacher_preds.shape[1]
        classes_count = 10

        for image_preds in np.transpose(teacher_preds):
            votes_per_class = np.bincount(image_preds, minlength=classes_count).astype(float)
            max_votes = np.max(votes_per_class)
            noise_1 = np.random.normal(loc=0, scale=self.sigma1, size=1)

            new_label = None
            if max_votes + noise_1 >= self.threshold:
                votes_per_class += np.random.normal(loc=0, scale=self.sigma2, size=classes_count)
                new_label = np.argmax(votes_per_class)

            labels = np.append(labels, new_label)

        return labels
