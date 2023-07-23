"""Evaluate student privacy metrics"""

import argparse
import gc
import importlib
import json
from typing import Dict

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
    aggregation_name = experiment_config["aggregation_name"]
    teachers_exp_name = train_args["teachers_exp_name"]

    datasets_module = importlib.import_module("datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    dataset_args = experiment_config.get("dataset_args", {})
    dataset = dataset_class_(**dataset_args)
    dataset.prepare_data()
    max_student_train_queries = dataset.max_student_train_queries

    networks_module = importlib.import_module("networks")
    network_fn = getattr(networks_module, experiment_config["network"])
    network_fn_teacher = getattr(networks_module, experiment_config["network_teacher"])

    aggregators_module = importlib.import_module("aggregators")
    aggregator_class_ = getattr(aggregators_module, experiment_config["aggregator"])
    aggregator_args = experiment_config.get("aggregator_args", {})
    aggregator = aggregator_class_(**aggregator_args)

    aggregation_path = (
        f"./classifier/weights/teachers/{teachers_exp_name}/{aggregation_name}"
    )
    train_labels = np.load(f"{aggregation_path}/student_labels.npy")
    indices = torch.load(f"{aggregation_path}/dataset_indices.pt")
    teacher_preds = torch.load(f"{aggregation_path}/teacher_preds.pt")

    samples_per_teacher = 800
    train_idxs = indices[:max_student_train_queries]
    dataset.set_student_train_labels(
        train_labels[:max_student_train_queries], train_idxs
    )
    val_idxs = indices[max_student_train_queries : max_student_train_queries + 5000]
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

    # A number of samples is chosen to calculate the optimal student classifier threshold, the optimal
    # threshold is calculated using a ROC curve, and then a bunch of metrics are computed.
    # The metrics are data-dependent & independent epsilon, true/false pos/neg rates, nr of samples used for
    # student threshold calculation, and optimal threshold obtained
    ddeps, dieps, tps, tns, fps, fns, sample_for_threshold_count, optimal_thresholds = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    all_samples_for_threshold_calculation = [200]
    nr_samples_for_training = max_student_train_queries
    for samples_for_threshold_calculation in all_samples_for_threshold_calculation:
        print(
            f"Finding optimal student threshold with {samples_for_threshold_calculation} samples..."
        )
        find_optim_threshold_idxs = indices[
            nr_samples_for_training : nr_samples_for_training
            + samples_for_threshold_calculation
        ]
        find_optim_threshold_subset = Subset(
            dataset.test_data, find_optim_threshold_idxs
        )
        find_optim_threshold_dataloader = DataLoader(
            find_optim_threshold_subset,
            batch_size=batch_size,
            num_workers=dataset_args["num_workers"],
        )
        y_pred = torch.zeros(0, dtype=torch.float).to(device)
        y_true = torch.zeros(0, dtype=torch.float).to(device)

        for images, labels in find_optim_threshold_dataloader:
            images, labels = images.to(device), labels.to(device)
            output = student(images)

            y_pred = torch.cat((y_pred, output.reshape(output.shape[0])))
            y_true = torch.cat((y_true, labels.reshape(labels.shape[0])))

        fpr, tpr, thresholds = skmetrics.roc_curve(
            y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        )

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        # thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1.
        # That's why we do tpr[1:] instead of tpr[:], to ignore this index
        optimal_idx = 1 + np.argmax(tpr[1:] - fpr[1:])
        optimal_threshold = thresholds[optimal_idx]
        optimal_thresholds.append(optimal_threshold)

        # Calculate student accuracy with optimal threshold
        optimal_student_preds = torch.zeros(0, dtype=torch.int).to(device)
        val_idxs_2 = indices[
            nr_samples_for_training + samples_for_threshold_calculation :
        ]
        val_subset_2 = Subset(dataset.test_data, val_idxs_2)
        val_dataloader_2 = DataLoader(
            val_subset_2, batch_size=batch_size, num_workers=dataset_args["num_workers"]
        )

        for images, labels in val_dataloader_2:
            images, labels = images.to(device), labels.to(device)
            output = student(images)
            y_hat = (output >= optimal_threshold).type(torch.int)
            optimal_student_preds = torch.cat(
                (optimal_student_preds, y_hat.reshape(y_hat.shape[0]))
            )

        y_true = torch.Tensor(
            dataset.test_data.data.loc[
                indices[nr_samples_for_training + samples_for_threshold_calculation :],
                "Etiqueta",
            ].values
        )
        y_true = y_true.to(device)

        tp, fn, tn, fp, _ = stat_scores(y_true, optimal_student_preds, class_index=1)
        tp = tp.item()
        tn = tn.item()
        fp = fp.item()
        fn = fn.item()
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        sample_for_threshold_count.append(samples_for_threshold_calculation)

        # si tengo problemas, aumentar moment range. es una especie de rango cuando esta minimizando los momentos
        moments_range = [10]
        data_dep_eps_per_moment, data_indep_eps_per_moment = [], []
        for moments in moments_range:
            print(f"Calculating epsilon with {moments} moments...")
            data_dep_eps, data_ind_eps = pate.perform_analysis(
                teacher_preds=teacher_preds[
                    :, : max_student_train_queries + samples_for_threshold_calculation
                ]
                .cpu()
                .numpy(),
                indices=train_labels[
                    : max_student_train_queries + samples_for_threshold_calculation
                ],
                noise_eps=aggregator_args["gamma"],
                delta=1e-5,
                moments=moments,
            )
            data_dep_eps_per_moment.append(data_dep_eps)
            data_indep_eps_per_moment.append(data_ind_eps)

        ddeps.append(data_dep_eps_per_moment)
        dieps.append(data_indep_eps_per_moment)

    tprs = [(tp / (tp + fn)) for (tp, fn) in zip(tps, fns)]
    tnrs = [(tn / (tn + fp)) for (tp, fn) in zip(tns, fps)]

    metrics = {
        "experiment": "no_noise",
        "aggregator": experiment_config["aggregator"],
        "aggregator_args": aggregator_args,
        "student_train_queries": dataset_args["max_student_train_queries"],
        "moments_range": [float(moment) for moment in moments_range],
        "student_data_eps": 0.1,
    }

    for i, samples_for_threshold_calculation in enumerate(
        all_samples_for_threshold_calculation
    ):
        for j, moments in enumerate(moments_range):
            metrics[
                f"dde_w_{moments}_mom_{samples_for_threshold_calculation}_thr"
            ] = ddeps[i][j]
            metrics[
                f"die_w_{moments}_mom_{samples_for_threshold_calculation}_thr"
            ] = dieps[i][j]

        metrics[f"tp_w_{samples_for_threshold_calculation}_thresh"] = tps[i]
        metrics[f"tn_w_{samples_for_threshold_calculation}_thresh"] = tns[i]
        metrics[f"fp_w_{samples_for_threshold_calculation}_thresh"] = fps[i]
        metrics[f"fn_w_{samples_for_threshold_calculation}_thresh"] = fns[i]
        metrics[f"tpr_w_{samples_for_threshold_calculation}_thresh"] = tprs[i]
        metrics[f"tnr_w_{samples_for_threshold_calculation}_thresh"] = tnrs[i]
        metrics[
            f"optimal_thresholds_w_{samples_for_threshold_calculation}_thresh"
        ] = optimal_thresholds[i]

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

    # There's 10 experiments for each level of noise applied to student's data.
    # For each experiment, `aggregation_name` indicates the path where noisy aggregated labels
    # are found, and `exp_name` indicates where the trained student model is found.
    experiment_config["aggregation_name"] = "agg_1_v_skunk"
    experiment_config["exp_name"] = "2020-12-07-21:01-accelerated-auk"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_1_v_ape"
    experiment_config["exp_name"] = "2020-12-07-21:01-purring-lemming"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_1_v_boobook"
    experiment_config["exp_name"] = "2020-12-07-21:02-wine-sawfly"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_1_v_bug"
    experiment_config["exp_name"] = "2020-12-07-21:03-proficient-tapir"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_1_v_coyote"
    experiment_config["exp_name"] = "2020-12-07-21:04-woodoo-scallop"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_1_v_fulmar"
    experiment_config["exp_name"] = "2020-12-07-21:05-refreshing-starling"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_1_v_hornet"
    experiment_config["exp_name"] = "2020-12-07-21:06-fast-cougar"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_1_v_mustang"
    experiment_config["exp_name"] = "2020-12-07-21:07-therapeutic-wren"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_1_v_stoat"
    experiment_config["exp_name"] = "2020-12-07-21:09-sociable-fulmar"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_1_v_wombat"
    experiment_config["exp_name"] = "2020-12-07-21:10-therapeutic-stallion"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_5_v_guillemot"
    experiment_config["exp_name"] = "2020-12-07-21:53-discerning-macaw"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_5_v_honeybee"
    experiment_config["exp_name"] = "2020-12-07-21:54-dexterous-pigeon"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_5_v_kudu"
    experiment_config["exp_name"] = "2020-12-07-21:55-tan-tanuki"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_5_v_lion"
    experiment_config["exp_name"] = "2020-12-07-21:56-jolly-meerkat"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_5_v_mule"
    experiment_config["exp_name"] = "2020-12-07-21:57-successful-sambar"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_5_v_scallop"
    experiment_config["exp_name"] = "2020-12-07-21:58-wisteria-jackdaw"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_5_v_termite"
    experiment_config["exp_name"] = "2020-12-07-21:59-hallowed-hound"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_5_v_toucanet"
    experiment_config["exp_name"] = "2020-12-07-22:00-tungsten-meerkat"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_5_v_vulture"
    experiment_config["exp_name"] = "2020-12-07-22:01-sapphire-teal"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_10_v_caracal"
    experiment_config["exp_name"] = "2020-12-07-22:02-accurate-markhor"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_10_v_guppy"
    experiment_config["exp_name"] = "2020-12-07-22:03-simple-nuthatch"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_10_v_horse"
    experiment_config["exp_name"] = "2020-12-07-22:04-spiked-herring"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_10_v_hyrax"
    experiment_config["exp_name"] = "2020-12-07-22:05-prophetic-termite"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_10_v_magpie"
    experiment_config["exp_name"] = "2020-12-07-22:06-rugged-chihuahua"
    train_student(experiment_config, use_wandb=args.use_wandb)
    experiment_config["aggregation_name"] = "agg_10_v_rhino"
    experiment_config["exp_name"] = "2020-12-07-22:07-purple-spaniel"
    train_student(experiment_config, use_wandb=args.use_wandb)
    # experiment_config["aggregation_name"] = "agg_10_v_seagull"
    # experiment_config["exp_name"] = "2020-12-07-22:08-lemon-crab"
    # train_student(experiment_config, use_wandb=args.use_wandb)
    # experiment_config["aggregation_name"] = "agg_10_v_silkworm"
    # experiment_config["exp_name"] = "2020-12-07-22:09-wakeful-uakari"
    # train_student(experiment_config, use_wandb=args.use_wandb)
    # experiment_config["aggregation_name"] = "agg_10_v_trout"
    # experiment_config["exp_name"] = "2020-12-07-22:10-courageous-squirrel"
    # train_student(experiment_config, use_wandb=args.use_wandb)
    # experiment_config["aggregation_name"] = "agg_10_v_asp"
    # experiment_config["exp_name"] = "2020-12-07-22:11-accomplished-mongrel"
    # train_student(experiment_config, use_wandb=args.use_wandb)


if __name__ == "__main__":
    main()
