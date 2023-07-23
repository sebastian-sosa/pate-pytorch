import torch
from torch.utils.data import DataLoader

def replace_labels(dataloader: DataLoader, labels):
    for i, (data, _) in enumerate(iter(dataloader)):
        yield data, torch.from_numpy(labels[i*len(data):(i+1)*len(data)])
