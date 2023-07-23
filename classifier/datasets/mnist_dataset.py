import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import transforms

from . import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MnistDataset(Dataset):
    def __init__(
        self,
        max_teacher_samples: int = None,
        max_student_train_queries: int = None,
        num_workers: int = 1,
    ):
        super().__init__(max_teacher_samples, max_student_train_queries, num_workers)
        self.input_shape = (28, 28, 1)

    def prepare_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.train_data = datasets.MNIST(
            root="data", train=True, download=True, transform=transform
        )
        self.test_data = datasets.MNIST(
            root="data", train=False, download=True, transform=transform
        )

    def set_student_train_labels(self, new_labels: np.ndarray, not_none_label_indexes: np.ndarray):
        new_train_labels_idxs = np.random.choice(
            not_none_label_indexes, size=self.max_student_train_queries, replace=False
        )
        self.test_data.targets[new_train_labels_idxs] = torch.from_numpy(
            new_labels[new_train_labels_idxs].astype(np.long)
        )
        self.student_train_samples_idxs = new_train_labels_idxs
