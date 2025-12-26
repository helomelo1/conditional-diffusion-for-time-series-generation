import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, prices, seq_len=128, normalize=True):
        self.seq_len = seq_len

        prices = np.asarray(prices, dtype=np.float32)
        returns = np.log(prices[1:] / prices[:-1])

        sequences, conditions = [], []

        for i in range(len(returns) - seq_len):
            seq = returns[i : i + seq_len]

            mean = seq.mean()
            std = seq.std()

            sequences.append(seq)
            conditions.append([mean, std])

        self.x = np.stack(sequences)
        self.c = np.stack(conditions)

        self.normalize = normalize
        if normalize:
            self.global_mean = self.x.mean()
            self.global_std = self.x.std() + 1e-8
            self.x = (self.x - self.global_mean) / self.global_std
        else:
            self.global_mean = 1.0
            self.global_std = 0.0

        self.x = torch.from_numpy(self.x).float()
        self.c = torch.from_numpy(self.c).float()

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.c[idx]
    

def load_prices_from_csv(path, price_column='Close'):
    df = pd.read_csv(path)
    return df[price_column].values