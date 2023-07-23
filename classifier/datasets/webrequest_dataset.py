import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from . import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WebrequestPytorchDataset(torch.utils.data.Dataset):
    """Malign and benign web requests pytorch dataset."""

    def __init__(self, csv_file):
        """Args:
        - csv_file (string): Path to the csv file."""
        self.data = pd.read_csv(csv_file, sep=" ", index_col=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.Tensor(self.data.iloc[idx, :-1].values)
        y = torch.Tensor([self.data.loc[idx, "Etiqueta"]])
        return x, y


class WebrequestDataset(Dataset):
    def __init__(
        self,
        max_teacher_samples: int = None,
        max_student_train_queries: int = 10000,
        num_workers: int = 1,
    ):
        super().__init__(max_teacher_samples, max_student_train_queries, num_workers)
        self.input_shape = (500,)

    def prepare_data(self):
        self.train_data = WebrequestPytorchDataset(
            csv_file="data/WebRequests/unigrams/train.csv"
        )
        self.test_data = WebrequestPytorchDataset(
            csv_file="data/WebRequests/unigrams/test.csv"
        )

    def set_student_train_labels(
        self, new_labels: np.ndarray, not_none_label_indexes: np.ndarray
    ):
        self.test_data.data.loc[not_none_label_indexes, "Etiqueta"] = new_labels
        self.student_train_samples_idxs = not_none_label_indexes

    def get_student_train_dataloader(self, batch_size: int):
        student_train_data = Subset(self.test_data, self.student_train_samples_idxs)
        student_train_loader = DataLoader(
            student_train_data, batch_size=batch_size, num_workers=self.num_workers
        )
        return student_train_loader

    def get_student_val_dataloader(self, batch_size: int) -> DataLoader:
        subset_data = Subset(self.test_data, self.student_val_idxs)
        return DataLoader(subset_data, batch_size, num_workers=self.num_workers)
