import os

import torch
import wandb
from datasets import SvhnDataset
from models import Teacher
from networks import cnn2
from pytorch_lightning import Trainer
from syft.frameworks.torch.dp import pate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(exp_name: str):
    dataset = SvhnDataset()
    dataset.prepare_data()
    num_teachers = 100

    teacher_networks = []
    teachers_path = f"classifier/weights/teachers/{exp_name}"
    for i, teacher_file in enumerate(sorted(os.listdir(teachers_path))):
        if teacher_file.startswith("teacher_"):
            teacher_network = network_fn_teacher(dataset.input_shape)
            teacher_network.load_state_dict(
                torch.load(f"{teachers_path}/{teacher_file}")
            )
            teacher_network.eval()
            teacher_networks.append(teacher_network)

    aggregator = Aggregator(gamma)
    teacher_preds, train_labels = aggregator.label_data(
        teacher_networks, dataset.get_student_train_dataloader(batch_size)
    )

    print(f"Finished evaluating experiment {exp_name}")


if __name__ == "__main__":
    evaluate("2020-05-28-11:22-prehistoric-dodo")
