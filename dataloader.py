import torch
from torch.utils.data import DataLoader, Dataset


class EEGSignalDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):

    def __len__(self):
        return self.data.shape[0]