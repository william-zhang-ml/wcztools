"""
This module contains ...
a) code to spoof a synthetic dataset
b) class to help index into the synthetic dataset
"""
import os
from pathlib import Path
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import pandas as pd
from torch.utils.data import Dataset


THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def spoof_data(n_samp: int = 3200, seed: int = 0) -> None:
    """
    Generates and writes synthetic data to CSV files.
    Y ~ 2X + 1 + N(0, 0.2)

    Args:
        n_samp: number of samples in both the train and test set
        seed:   random number generator seed
    """
    rng = RandomState(MT19937(SeedSequence(seed)))

    # training data
    x_data = rng.rand(n_samp)
    y_data = 2 * x_data + 1 + 0.2 * rng.randn(n_samp)
    table = pd.DataFrame({'x': x_data, 'y': y_data})
    table.to_csv(THIS_DIR / 'data' / 'train.csv', index=False)

    # test data
    x_data = rng.rand(n_samp)
    y_data = 2 * x_data + 1 + 0.2 * rng.randn(n_samp)
    table = pd.DataFrame({'x': x_data, 'y': y_data})
    table.to_csv(THIS_DIR / 'data' / 'test.csv', index=False)


class SpoofDataset(Dataset):
    """ Indexable dataset class for linear regression example. """
    def __init__(self, train: bool = True) -> None:
        """
        Constructor. Loads either the train or test set.

        Args:
            train: whether to load the train or test set
        """
        super().__init__()
        if train:
            self.data = pd.read_csv(THIS_DIR / 'data' / 'train.csv')
        else:
            self.data = pd.read_csv(THIS_DIR / 'data' / 'test.csv')
        for col in self.data:
            self.data[col] = self.data[col].astype(np.float32)

    def __len__(self) -> int:
        """ Returns: number of samples in the dataset """
        return len(self.data)

    def __getitem__(self, item: int) -> np.ndarray:
        """
        Get sample by index.

        Args:
            item: sample index

        Returns: (x, y) sample
        """
        return tuple(self.data.loc[item].to_numpy())
