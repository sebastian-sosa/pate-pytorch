from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def cnn4(input_shape: Tuple[int, ...]) -> nn.Module:
    input_h = input_shape[0]
    input_w = input_shape[1] if len(input_shape) > 1 else input_h
    input_c = input_shape[2] if len(input_shape) > 2 else 1

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()

            self.cnn1 = nn.Conv2d(
                in_channels=input_c, out_channels=16, kernel_size=3, stride=1, padding=1
            )
            self.cnn1_bn = nn.BatchNorm2d(16)
            self.cnn2 = nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
            )
            self.cnn2_bn = nn.BatchNorm2d(16)
            self.drop1 = nn.Dropout(0.25)
            self.cnn3 = nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            )
            self.cnn3_bn = nn.BatchNorm2d(32)
            self.cnn4 = nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            )
            self.cnn4_bn = nn.BatchNorm2d(32)
            self.cnn5 = nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            )
            self.cnn5_bn = nn.BatchNorm2d(64)
            self.drop2 = nn.Dropout(0.25)
            dim_w = int(
                (((input_w / 2) / 2) / 2) / 2
            )  # [(W−K+2P)/S]+1 for each conv, /2 for maxpool
            dim_h = int(
                (((input_h / 2) / 2) / 2) / 2
            )  # [(W−K+2P)/S]+1 for each conv, /2 for maxpool
            fc1_in_features = int(dim_w * dim_h * 64)
            self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=256)
            self.drop3 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(in_features=256, out_features=10)

        def forward(self, x):
            x = self.cnn1(x)
            x = F.relu(x)
            x = self.cnn1_bn(x)

            x = self.cnn2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.drop1(x)
            x = self.cnn2_bn(x)

            x = self.cnn3(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.cnn3_bn(x)

            x = self.cnn4(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.cnn4_bn(x)

            x = self.cnn5(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.cnn5_bn(x)
            x = self.drop2(x)

            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.drop3(x)
            x = self.drop1(x)
            x = self.fc2(x)
            return x

    model = Classifier()
    return model
