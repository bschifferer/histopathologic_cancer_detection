import torch

from torch.utils.data import Dataset

class KaggleDataset(Dataset):
    """Simple dataset class for dataloader"""

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return self.x_train[idx, :,:,:], self.y_train[idx]