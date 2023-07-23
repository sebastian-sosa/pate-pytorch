import argparse
import importlib
import json
import os
import uuid
from datetime import datetime
from typing import Dict

import torch
from coolname import generate_slug

from aggregators import LaplaceAggregator
from datasets.dataloader_utils import replace_labels


def save_teacher_predictions(experiment_config: Dict):
    """
      Run an experiment to train a student

      Parameters
      ----------
      experiment_config (dict)
          Of the form
          {
              "dataset": "MnistDataset",
              "dataset_args": {
                  "student_train_samples": None,
              },
              "network_teacher": "cnn2",
              "network_args": {},
              "train_args": {
                  "teachers_exp_name": "2020-05-13-19:49-warm-antelope",
                  "batch_size": 32,
              }
          }
      """
    print(f"Running experiment with config {experiment_config}\n")
    train_args = experiment_config["train_args"]
    teachers_exp_name = train_args["teachers_exp_name"]
    batch_size = train_args["batch_size"]

    datasets_module = importlib.import_module("datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    dataset_args = experiment_config.get("dataset_args", {})
    dataset = dataset_class_(**dataset_args)
    dataset.prepare_data()

    networks_module = importlib.import_module("networks")
    network_fn_teacher = getattr(networks_module, experiment_config["network_teacher"])

    teacher_networks = []
    teachers_path = f"classifier/weights/teachers/{teachers_exp_name}"
    for i, teacher_file in enumerate(sorted(os.listdir(teachers_path))):
        if teacher_file.startswith("teacher_"):
            teacher_network = network_fn_teacher(dataset.input_shape)
            teacher_network.load_state_dict(
                torch.load(f"{teachers_path}/{teacher_file}", map_location=torch.device("cpu"))
            )
            teacher_network.eval()
            teacher_networks.append(teacher_network)

    aggregator = LaplaceAggregator(None)
    dataloader = dataset.get_student_train_dataloader(batch_size)

    teacher_preds = aggregator.get_teacher_preds(teacher_networks, dataloader)
    torch.save(teacher_preds, f"{teachers_path}/student_predictions.pt")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Experimenet JSON (\'{"dataset": "MnistDataset", "dataset_args": {}, "model": "Student", "network": "cnn", "network_args": {}, "train_args": {"teachers_exp_name": "2020-05-13-21:45-nippy-goldfish", "gamma": 0.2, "batch_size": 32, "epochs": 30, "learning_rate": 0.003}}\'',
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()
    experiment_config = json.loads(args.experiment_config)
    save_teacher_predictions(experiment_config)


if __name__ == "__main__":
    main()
