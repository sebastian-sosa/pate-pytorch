import argparse
import gc
import importlib
import json
import os
import uuid
from datetime import datetime
from typing import Dict

import numpy as np

import torch
import wandb
from coolname import generate_slug
from models import Teacher
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def train_teachers(experiment_config: Dict, gpus: int, use_wandb: bool = True):
    """
    Run an experiment to train teachers

    Parameters
    ----------
    experiment_config (dict)
        Of the form
        {
            "dataset": "MnistDataset",
            "dataset_args": {
                "max_teacher_samples": 9000,
                "max_student_train_queries": None,
                "sample_distribution": [100, 5000, 900, 1000], //the remaining samples are distributed equally
            },
            "network": "cnn",
            "teacher_network": "cnn",
            "network_args": {},
            "train_args": {
                "num_teachers": 100,
                "batch_size": 128,
                "epochs": 30,
                "learning_rate": 0.003
            }
        }
    gpu_ind (int)
        specifies the number of GPUs to use
    use_wandb (bool)
        sync training run to wandb
    """
    print(f"Running experiment with config {experiment_config}\n")

    exp_name = "2020-08-25-04:05-winged-llama"

    train_args = experiment_config["train_args"]
    epochs = train_args["epochs"]
    lr = train_args["learning_rate"]
    batch_size = train_args["batch_size"]
    num_teachers = train_args["num_teachers"]

    datasets_module = importlib.import_module("datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    dataset_args = experiment_config.get("dataset_args", {})
    dataset = dataset_class_(**dataset_args)

    networks_module = importlib.import_module("networks")
    network_fn = getattr(networks_module, experiment_config["network"])

    for i in range(num_teachers):
        logger = None
        if use_wandb:
            logger = WandbLogger(
                name=f"{exp_name}_Teacher_{i}", project="pate-pytorch", group=exp_name
            )
            logger.log_hyperparams(experiment_config)
        teacher = Teacher(i, num_teachers, lr, batch_size, dataset, network_fn)
        early_stop_callback = EarlyStopping(patience=3)
        trainer = Trainer(
            checkpoint_callback=False,
            distributed_backend="dp",
            early_stop_callback=False,
            logger=logger,
            gpus=gpus,
            max_epochs=epochs,
        )
        trainer.fit(teacher)

        teacher.save_weights(exp_name, experiment_config)
        print(f"Saved trained teacher {teacher.teacher_idx}")
        del teacher
        del trainer
        del logger
        del early_stop_callback
        gc.collect()

    print(f"Finished experiment {exp_name}")


def experiment_name():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
    return f"{str(timestamp)}-{generate_slug(2)}"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Experimenet JSON (\'{"dataset": "MnistDataset", "dataset_args": {"total_train_samples": 80000}, "model": "Teacher", "network": "cnn", "network_args": {}, "train_args": {"num_teachers": 2, "batch_size": 128, "epochs": 1}}\'',
    )
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="If true, use wandb for this run",
    )
    parser.add_argument(
        "--gpus", type=int, default=0, help="Provide number of GPUs to use."
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()
    experiment_config = json.loads(args.experiment_config)

    train_teachers(experiment_config, args.gpus, use_wandb=args.use_wandb)


if __name__ == "__main__":
    main()
