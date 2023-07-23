from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_dim: int):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


def fcn1(input_shape: Tuple[int, ...]) -> nn.Module:
    input_dim = input_shape[0]
    model = Classifier(input_dim)
    return model
