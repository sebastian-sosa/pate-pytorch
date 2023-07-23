import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple


def cnn2(input_shape: Tuple[int, ...]) -> nn.Module:
    input_h = input_shape[0]
    input_w = input_shape[1] if len(input_shape)>1 else input_h
    input_c = input_shape[2] if len(input_shape)>2 else 1

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()

            self.cnn1 = nn.Conv2d(in_channels=input_c, out_channels=32, kernel_size=3, stride=1)
            self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
            self.dropout1 = nn.Dropout(0.3)
            self.dropout2 = nn.Dropout(0.3)
            self.dropout3 = nn.Dropout(0.5)
            dim_w = int((((input_w-2) / 2) - 2) / 2)  # [(W−K+2P)/S]+1 for each conv, /2 for maxpool
            dim_h = int((((input_h-2) / 2) - 2) / 2)  # [(W−K+2P)/S]+1 for each conv, /2 for maxpool
            fc1_in_features = int(dim_w*dim_h*64)
            self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=128)
            self.fc2 = nn.Linear(in_features=128, out_features=10)

        def forward(self, x):
            x = self.cnn1(x)
            x = F.relu(x)
            x = self.dropout1(x) # what happens if this goes after cnn1? most put this before maxpool
            x = F.max_pool2d(x, 2)
            x = self.cnn2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout2(x) # idem
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout3(x)
            x = self.fc2(x)
            return x

    model = Classifier()
    return model
