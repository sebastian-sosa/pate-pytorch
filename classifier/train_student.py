import argparse
import gc
import importlib
import json
from datetime import datetime
from typing import Dict

import numpy as np

import torch
from coolname import generate_slug
from models import Student
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def train_student(experiment_config: Dict, gpus: int, use_wandb: bool = True):
    """
    Run an experiment to train a student

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
            "train_args": {
                "teachers_exp_name": "2020-05-13-19:49-warm-antelope",
                "gamma": 0.2,
                "batch_size": 32,
                "epochs": 30,
                "learning_rate": 0.003
            }
        }
    gpus (int)
        specifies the number of GPUs to use
    use_wandb (bool)
        sync training run to wandb
    """
    print(f"Running experiment with config {experiment_config}\n")

    exp_name = experiment_name()

    train_args = experiment_config["train_args"]
    teachers_exp_name = train_args["teachers_exp_name"]
    batch_size = train_args["batch_size"]
    epochs = train_args["epochs"]
    learning_rate = train_args["learning_rate"]
    aggregation_name = experiment_config["aggregation_name"]

    datasets_module = importlib.import_module("datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    dataset_args = experiment_config.get("dataset_args", {})
    dataset = dataset_class_(**dataset_args)
    dataset.prepare_data()

    networks_module = importlib.import_module("networks")
    network_fn = getattr(networks_module, experiment_config["network"])
    # if student labels aggregation is done in this script,
    # `network_fn_teacher` is used to instantiate a proper teacher model
    network_fn_teacher = getattr(networks_module, experiment_config["network_teacher"])

    # if student labels aggregation is done in this script, `aggregators` params would be used
    # to instatiate a proper aggregator
    aggregators_module = importlib.import_module("aggregators")
    aggregator_class_ = getattr(aggregators_module, experiment_config["aggregator"])
    aggregator_args = experiment_config.get("aggregator_args", {})

    max_student_train_queries = dataset.max_student_train_queries

    aggregation_path = (
        f"./classifier/weights/teachers/{teachers_exp_name}/{aggregation_name}"
    )
    train_labels = np.load(f"{aggregation_path}/student_labels.npy")
    indices = torch.load(f"{aggregation_path}/dataset_indices.pt")
    train_idxs = indices[:max_student_train_queries]

    dataset.set_student_train_labels(
        train_labels[:max_student_train_queries], train_idxs
    )

    val_idxs = indices[max_student_train_queries : max_student_train_queries + 5000]
    dataset.set_student_val_idxs(val_idxs)

    student = Student(
        teachers_exp_name,
        learning_rate,
        batch_size,
        train_labels,
        dataset=dataset,
        network_fn=network_fn,
    )

    logger = None
    if use_wandb:
        logger = WandbLogger(
            name=f"{exp_name}_Student",
            project="pate-pytorch",
            group=f"Student_{type(dataset).__name__}",
        )
        logger.log_hyperparams(experiment_config)

    early_stop_callback = EarlyStopping()
    trainer = Trainer(
        gpus=gpus,
        max_epochs=epochs,
        logger=logger,
        checkpoint_callback=False,
        early_stop_callback=False,
    )
    trainer.fit(student)
    student.save_weights(exp_name, experiment_config)

    del student
    del trainer
    del logger
    del early_stop_callback
    gc.collect()

    print(f"Finished experiment {exp_name}")
    return exp_name


def experiment_name():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
    return f"{str(timestamp)}-{generate_slug(2)}"


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
    parser.add_argument(
        "--gpus", type=int, default=0, help="Provide number of GPUs to use."
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()
    experiment_config = json.loads(args.experiment_config)
    exp_names = []

    # There's 10 experiments for each level of noise applied to student's data
    experiment_config["aggregation_name"] = "agg_1_v_skunk"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_1_v_ape"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_1_v_boobook"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_1_v_bug"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_1_v_coyote"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_1_v_fulmar"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_1_v_hornet"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_1_v_mustang"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_1_v_skunk"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_1_v_stoat"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_1_v_wombat"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_5_v_guillemot"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_5_v_caribou"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_5_v_honeybee"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_5_v_kudu"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_5_v_lion"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_5_v_mule"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_5_v_scallop"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_5_v_termite"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_5_v_toucanet"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_5_v_vulture"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_10_v_caracal"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_10_v_guppy"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_10_v_horse"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_10_v_hyrax"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_10_v_magpie"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_10_v_rhino"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_10_v_seagull"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_10_v_silkworm"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_10_v_trout"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )
    experiment_config["aggregation_name"] = "agg_10_v_asp"
    exp_names.append(
        train_student(experiment_config, args.gpus, use_wandb=args.use_wandb)
    )


if __name__ == "__main__":
    main()
