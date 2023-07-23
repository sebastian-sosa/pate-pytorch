"""Evaluate student privacy metrics"""

import argparse
import gc
import importlib
import json
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as skmetrics

import torch
from models import Student
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import stat_scores
from syft.frameworks.torch.dp import pate
from torch.utils.data import DataLoader, Subset


def train_student(experiment_config: Dict, use_wandb: bool = True):
    """
    Evaluate a student's true & negative positive rate and privacy guarantees

    Parameters
    ----------
    experiment_config (dict)
        Of the form
        {
            "dataset": "MnistDataset",
            "dataset_args": {
                "max_student_train_queries": None,
            },
            "network": "cnn",
            "network_teacher": "cnn",
            "network_args": {},
            "exp_name": "2020-05-13-19:49-warm-antelope",
            "train_args": {
                "teachers_exp_name": "2020-05-13-19:49-warm-antelope",
                "gamma": 0.2,
                "batch_size": 32,
                "epochs": 30,
                "learning_rate": 0.003
            }
        }
    use_wandb (bool)
        sync training run to wandb
    """
    print(f"Running experiment with config {experiment_config}\n")

    exp_name = experiment_config["exp_name"]

    train_args = experiment_config["train_args"]
    batch_size = train_args["batch_size"]
    epochs = train_args["epochs"]
    learning_rate = train_args["learning_rate"]

    datasets_module = importlib.import_module("datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    dataset_args = experiment_config.get("dataset_args", {})
    dataset = dataset_class_(**dataset_args)
    dataset.prepare_data()

    networks_module = importlib.import_module("networks")
    network_fn = getattr(networks_module, experiment_config["network"])
    network_fn_teacher = getattr(networks_module, experiment_config["network_teacher"])

    aggregators_module = importlib.import_module("aggregators")
    aggregator_class_ = getattr(aggregators_module, experiment_config["aggregator"])
    aggregator_args = experiment_config.get("aggregator_args", {})
    aggregator = aggregator_class_(**aggregator_args)

    indices = torch.load("./dataset_indices_mnist.pt")
    teacher_preds = torch.load("./teacher_preds_mnist_1_05.pt")
    max_student_train_queries = dataset.max_student_train_queries

    train_labels_path = "./student_labels_mnist_1_05.npy"
    experiment_config["train_labels"] = train_labels_path
    train_labels = np.load(train_labels_path)
    train_idxs = indices[:max_student_train_queries]
    dataset.set_student_train_labels(
        train_labels[:max_student_train_queries], train_idxs
    )

    val_idxs = indices[max_student_train_queries : max_student_train_queries + 1000]
    dataset.set_student_val_idxs(val_idxs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student_weights_path = f"classifier/weights/students/{exp_name}/student"
    student = Student(
        "teachers_exp_name",
        learning_rate,
        batch_size,
        train_labels,
        dataset=dataset,
        network_fn=network_fn,
    ).to(device)
    student.network.load_state_dict(torch.load(student_weights_path))
    student.eval()

    logger = None
    if use_wandb:
        logger = WandbLogger(
            name=f"{exp_name}_Student",
            project="pate-pytorch",
            group=f"Student_{type(dataset).__name__}",
        )

    # Calculate student accuracy with val_2
    preds = torch.zeros(0, dtype=torch.long).to(device)
    val_idxs_2 = indices[max_student_train_queries + 1000 :]
    val_subset_2 = Subset(dataset.test_data, val_idxs_2)
    val_dataloader_2 = DataLoader(
        val_subset_2, batch_size=batch_size, num_workers=dataset_args["num_workers"]
    )

    for images, labels in val_dataloader_2:
        images, labels = images.to(device), labels.to(device)
        output = student(images)
        predictions = torch.argmax(torch.exp(output), dim=1)
        preds = torch.cat((preds, predictions))

    y_true = dataset.test_data.targets[
        indices[max_student_train_queries + 1000 :]
    ].cpu()

    val_acc = torch.mean((y_true == preds.cpu()).float())
    print("Calculating epsilon...")
    data_dep_eps, data_ind_eps = 99, 99
#     data_dep_eps, data_ind_eps = pate.perform_analysis(
#         teacher_preds=teacher_preds[:, :max_student_train_queries],
#         indices=train_labels[:max_student_train_queries],
#         noise_eps=aggregator_args["gamma"],
#         delta=1e-5,
#         moments=1,
#     )

    metrics = {
        "aggregator": experiment_config["aggregator"],
        "aggregator_args": aggregator_args,
        "student_train_queries": dataset_args["max_student_train_queries"],
        "student_data_eps": 1,
        "val_acc": val_acc,
        "data_dep_eps": data_dep_eps,
        "data_ind_eps": data_ind_eps,
    }

    if logger is not None:
        logger.log_metrics(metrics)
    del student
    del logger
    gc.collect()

    print(f"Finished experiment {exp_name}")
    print(metrics)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_config",
        type=str,
        help="Experimenet JSON",
    )
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="If true, use wandb for this run",
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()
    experiment_config = json.loads(args.experiment_config)

    train_student(experiment_config, use_wandb=args.use_wandb)


if __name__ == "__main__":
    main()
