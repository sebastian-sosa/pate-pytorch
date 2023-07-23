"""Aggregate teacher labels with random noise"""

import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import yaml
from sklearn import metrics

import torch
from classifier.aggregators import LaplaceAggregator
from coolname import generate_slug
from classifier.datasets import WebrequestDataset
from classifier.networks import fcn1
from torch import nn
from torch.utils.data import DataLoader, Subset

# import torch.multiprocessing


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "tasks/aggregate_labels_config.yaml"
# torch.multiprocessing.set_sharing_strategy("file_system")


def load_teachers(path: str, input_shape: Tuple[int, ...]) -> List[nn.Module]:
    teacher_networks = []

    for teacher_file in sorted(os.listdir(path)):
        if teacher_file.startswith("teacher_"):
            teacher_network = cnn(input_shape)
            teacher_network.load_state_dict(torch.load(f"{path}/{teacher_file}"))
            teacher_network.eval()
            teacher_networks.append(teacher_network)

    return teacher_networks


def calculate_teacher_thresholds(
    dataset,
    dataset_indices,
    teacher_networks: List[nn.Module],
    num_workers: int,
    samples_per_teacher: int,
    batch_size: int,
    aggregator: LaplaceAggregator,
) -> List[float]:
    teacher_thresholds = []

    print("Calculating teacher thresholds...")
    for i, teacher in enumerate(teacher_networks):
        print(i)
        subset = Subset(
            dataset,
            dataset_indices[i * samples_per_teacher : (i + 1) * samples_per_teacher],
        )
        dataloader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers)

        y_pred = torch.zeros(0, dtype=torch.float).to(device)
        y_true = torch.zeros(0, dtype=torch.float).to(device)
        teacher.to(device)

        for samples, labels in dataloader:
            samples = aggregator.apply_noise_to_samples(samples)
            samples, labels = samples.to(device), labels.to(device)
            output = teacher(samples)

            y_pred = torch.cat((y_pred, output.reshape(output.shape[0])))
            y_true = torch.cat((y_true, labels.reshape(labels.shape[0])))

        fpr, tpr, thresholds = metrics.roc_curve(
            y_true.cpu(), y_pred.cpu().detach().numpy()
        )
        optimal_idx = 1 + np.argmax(tpr[1:] - fpr[1:])
        optimal_threshold = thresholds[optimal_idx]
        teacher_thresholds.append(optimal_threshold)
        del teacher

    return teacher_thresholds


# TODO use method Dataset.get_student_train_dataloader()
def get_student_dataloader(
    dataset,
    dataset_indices,
    batch_size: int,
    num_workers: int,
    num_teachers: int,
    num_labels: int,
    samples_per_teacher: int,
) -> DataLoader:

    nr_samples_for_teacher_thresholds = num_teachers * samples_per_teacher
    train_idxs = dataset_indices[
        nr_samples_for_teacher_thresholds : nr_samples_for_teacher_thresholds
        + num_labels
    ]
    train_subset = Subset(dataset, train_idxs)
    dataloader = DataLoader(
        train_subset, batch_size=batch_size, num_workers=num_workers
    )
    return dataloader


# checkear si sklearn tfidf da valores negativos. Si no, podemos hacer scale=1/eps
# en sklearn la row tiene norma 1? sus valores son positivos?
# fijarse en el dataset si algun dato es negativo. No, ninguno es negativo.

# si subo gamma en LaplaceAggreegator, pierdo privacy y gano accuracy
# hacer eso y bajar este epsilon

# bajarlo a 1, 0.5, 0.3
# tambien usar 2000, 4000, 6000 training samples para el student


def laplace_randomizer(row):
    scale = (
        1 / 2
    )  # delta f of page 30 in book. Delta f es la norma mas grande de la resta de dos vectores.
    noise = np.random.laplace(0, scale, row.shape)
    return row + noise


# TODO use Aggregator.get_teacher_preds()
def make_teacher_preds(
    teacher_networks: List[nn.Module],
    student_dataloader: DataLoader,
) -> torch.Tensor:
    print("Predicting labels with", len(teacher_networks), "teachers...")
    with torch.no_grad():
        preds = torch.zeros(
            (len(teacher_networks), len(student_dataloader.dataset)), dtype=torch.long
        )
        for i, teacher in enumerate(teacher_networks):
            print(i)
            results = get_teacher_pred(teacher, student_dataloader)
            preds[i] = results


# TODO use Aggregator.__get_teacher_pred()
def get_teacher_pred(teacher: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    """Return prediction of single teacher"""
    outputs = torch.zeros(0, dtype=torch.long).to(device)
    teacher.to(device)
    teacher.eval()

    for images, labels in dataloader:
        labels = labels.to(device)
        images = torch.Tensor(
            np.apply_along_axis(laplace_randomizer, axis=1, arr=images.cpu())
        ).to(device)
        output = teacher(images)
        predictions = torch.argmax(torch.exp(output), dim=1)
        outputs = torch.cat((outputs, predictions))

    return outputs


# TODO use LaplaceAggregator.aggregate_teacher_preds()
def aggregate_teacher_preds(preds: torch.Tensor) -> np.array:
    new_labels = np.array([]).astype(int)
    teacher_preds = preds.cpu().numpy()
    gamma = 0.05
    classes_count = 10

    for image_preds in np.transpose(teacher_preds):
        label_counts = np.bincount(image_preds, minlength=classes_count).astype(float)
        label_counts += np.random.laplace(0, 1 / gamma, classes_count)

        new_label = np.argmax(label_counts)
        new_labels = np.append(new_labels, new_label)

    return new_labels


def aggregate_labels(config):
    dataset = MnistDataset(  # TODO change dataset
        max_teacher_samples=None,
        max_student_train_queries=config["max_student_train_queries"],
        num_workers=config["num_workers"],
    )
    dataset.prepare_data()
    dataset = dataset.test_data
    teacher_networks = load_teachers(config["teachers_path"], config["input_shape"])
    num_teachers = len(teacher_networks)

    aggregator = LaplaceAggregator(
        config["gamma"],
        config["classes_count"],
        config["student_data_eps"],
    )

    # student dataset indices
    dataset_indices = np.arange(len(dataset))
    np.random.shuffle(dataset_indices)

    teacher_thresholds = calculate_teacher_thresholds(
        dataset,
        dataset_indices,
        teacher_networks,
        aggregator,
    )
    student_dataloader, train_idxs = get_student_dataloader(
        dataset,
        dataset_indices,
        config["batch_size"],
        config["num_workers"],
        num_teachers,
        config["num_labels"],
        config["samples_per_teacher"],
    )
    teacher_preds, aggregated_labels = aggregator.label_data(
        teacher_networks, student_dataloader, teacher_thresholds
    )
    print("Student labels shape", aggregated_labels.shape)

    aggregation_name = (
        f"agg_{int(config['student_data_eps']*10)}_v_{generate_slug(2).split('-')[1]}"
    )
    aggregation_path = Path(f"{config['teachers_path']}/{aggregation_name}")
    aggregation_path.mkdir(parents=True, exist_ok=True)

    torch.save(teacher_preds, f"{aggregation_path}/teacher_preds.pt")
    torch.save(train_idxs, f"{aggregation_path}/dataset_indices.pt")
    np.save(f"{aggregation_path}/student_labels.npy", aggregated_labels)
    with open(f"{aggregation_path}/config.yaml", "w") as outfile:
        yaml.dump(config, outfile)


def main():
    with open(CONFIG_PATH) as config_file:
        config = yaml.safe_load(config_file)
    for _ in range(10):  # repeat same experiment 10 times to get median result
        aggregate_labels(config)


if __name__ == "__main__":
    main()
