from typing import List

import numpy as np

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Aggregator:
    """
    Provides utility functions to label student training data with provided teacher models.
    Abstract class to be subclassed by aggregators which follow distinct aggregation strategies
    when choosing the teacher majority vote.
    """

    def label_data(
        self, teachers, dataloader: DataLoader, teacher_thresholds: List[float] = None
    ) -> (torch.Tensor, np.ndarray):
        teacher_preds = self.get_teacher_preds(teachers, dataloader, teacher_thresholds)
        labels = self.aggregate_teacher_preds(teacher_preds)
        return teacher_preds, labels

    def get_teacher_preds(
        self,
        teacher_networks: List[torch.nn.Module],
        dataloader: DataLoader,
        teacher_thresholds: List[float] = None,
    ) -> torch.Tensor:
        """Return predictions of all teachers in tensor of dimension (num_teachers, num_samples)"""

        print(f"Labelling data with {len(teacher_networks)} teachers...")
        if teacher_thresholds is None:
            teacher_thresholds = [0.5] * len(teacher_networks)
        with torch.no_grad():
            preds = torch.zeros(
                (len(teacher_networks), len(dataloader.dataset)), dtype=torch.long
            )
            for i, teacher in enumerate(teacher_networks):
                print(i)
                results = self.__get_teacher_pred(
                    teacher, dataloader, teacher_thresholds[i]
                )
                preds[i] = results

        return preds

    def __get_teacher_pred(
        self, teacher, dataloader, teacher_threshold: float
    ) -> torch.Tensor:
        """Return prediction of single teacher"""
        outputs = torch.zeros(0, dtype=torch.long).to(device)
        teacher.to(device)
        teacher.eval()

        for samples, labels in dataloader:
            samples = self.apply_noise_to_samples(samples)
            samples, labels = samples.to(device), labels.to(device)
            output = teacher(samples)
            pred = (output > teacher_threshold).long()
            outputs = torch.cat((outputs, pred.reshape(pred.shape[0])))

        return outputs

    def aggregate_teacher_preds(self, teacher_preds) -> np.ndarray:
        """Aggregate teacher predictions based on the aggregation mechanism policy.
        Returns None for labels where confidence is below threshold"""
        pass

    def apply_noise_to_samples(self, samples: np.ndarray) -> torch.Tensor:
        pass
