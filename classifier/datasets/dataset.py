"""Dataset class to be extended by dataset-specific classes."""
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class Dataset:
    """Simple abstract class for datasets."""

    def __init__(
        self,
        max_teacher_samples: int = None,
        max_student_train_queries: int = 5000,
        num_workers: int = 1,
    ):
        self.max_teacher_samples: int = max_teacher_samples
        self.max_student_train_queries: int = max_student_train_queries
        self.student_train_samples_idxs: np.array = np.array([])
        self.num_workers: int = num_workers

        # dataset used for training teachers
        self.train_data: Dataset = None
        # dataset used for calculating optimal teacher thresholds, validating teacher acc,
        # train student model, calculate optimal student threshold, validate and test student
        self.test_data: Dataset = None

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data"

    def prepare_data(self):
        pass

    def get_teacher_train_dataloader(
        self, teacher_idx: int, num_teachers: int, batch_size: int
    ):
        teacher_samples = len(self.train_data)
        if self.max_teacher_samples:
            teacher_samples = min(teacher_samples, self.max_teacher_samples)

        samples_per_teacher = teacher_samples // num_teachers
        indices = list(
            range(
                teacher_idx * samples_per_teacher,
                (teacher_idx + 1) * samples_per_teacher,
            )
        )
        subset_data = Subset(self.train_data, indices)
        dataloader = DataLoader(
            subset_data, batch_size=batch_size, num_workers=self.num_workers
        )

        return dataloader

    def get_teacher_val_dataloader(self, batch_size: int):
        teacher_val_data = Subset(self.test_data, list(range(3000)))
        return DataLoader(teacher_val_data, batch_size, num_workers=self.num_workers)

    def get_student_train_dataloader(self, batch_size: int):
        student_train_data = Subset(
            self.test_data, list(range(self.max_student_train_queries))
        )
        student_train_loader = DataLoader(
            student_train_data, batch_size=batch_size, num_workers=self.num_workers
        )
        return student_train_loader

    def get_student_val_dataloader(self, batch_size: int):
        student_test_data = Subset(
            self.test_data,
            list(range(self.max_student_train_queries, len(self.test_data))),
        )
        student_test_loader = DataLoader(
            student_test_data, batch_size=batch_size, num_workers=self.num_workers
        )
        return student_test_loader

    def set_student_train_labels(
        self, new_labels: np.ndarray, not_none_label_indexes: np.ndarray
    ):
        """Given new aggregated labels (some of which may be None if there wasn't a sufficient agreement),
        replace those labels so as to use them for training the student, and store their indexes on student_train_samples_idxs"""
        pass

    def set_student_val_idxs(self, indexes: np.ndarray):
        self.student_val_idxs = indexes
