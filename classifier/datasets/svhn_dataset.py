import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import transforms

from . import Dataset


class SvhnDataset(Dataset):
    def __init__(
        self,
        max_teacher_samples: int = None,
        max_student_train_queries: int = None,
        num_workers: int = 1,
    ):
        super().__init__(max_teacher_samples, max_student_train_queries, num_workers)
        self.input_shape = (32, 32, 3)

    def prepare_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),]
        )
        train_data = datasets.SVHN(root="data", split="train", download=True, transform=transform)
        extra_data = datasets.SVHN(root="data", split="extra", download=True, transform=transform)
        self.train_data = ConcatDataset([train_data, extra_data])
        self.test_data = datasets.SVHN(
            root="data", split="test", download=True, transform=transform
        )

    def set_student_train_labels(self, new_labels: np.ndarray, not_none_label_indexes: np.ndarray):
        new_train_labels_idxs = np.random.choice(
            not_none_label_indexes, size=self.max_student_train_queries, replace=False
        )
        self.test_data.labels[new_train_labels_idxs] = new_labels[new_train_labels_idxs]
        self.student_train_samples_idxs = new_train_labels_idxs
