import numpy as np
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    """
    A Dataset object wrapper for the dataset initialized from numpy arrays.
    """

    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Retrieve images by their index.
        :param idx: index of the image in the dataset.
        :return: (X[idx], y[idx]): the image and its label
        """
        return self.X[idx], self.y[idx]
