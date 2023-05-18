import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union


def changeType(x, target='Tensor'):
    if type(x).__name__ != target:
        if target == 'Tensor':
            return torch.tensor(x)

class PairDataset(Dataset):
    def __init__(self, Y: np.ndarray, S: np.ndarray):
        assert Y.shape[1] == S.shape[1]
        self.len = S.shape[1]

        self.Y = torch.tensor(Y.T)
        if np.iscomplexobj(S):
            self.S = torch.tensor(np.concatenate([S.real, S.imag]).T)
        else:
            self.S = torch.tensor(S.T)

    def __getitem__(self, j: int) -> tuple[torch.Tensor]:
        return self.Y[j], self.S[j]

    def __len__(self):
        return self.len

def RelMSE(output, target):
    return torch.mean((torch.norm((output - target), dim=1) / torch.norm(target, dim=1)) ** 2)

def MSE(output, target):
    return torch.mean(torch.norm((output - target), dim=1) ** 2)

def read_data(path: Union[str, Path], n_trn: int = None, n_val: int = None):

    Y_trn = np.load(path / 'Y_trn.npy')
    S_trn = np.load(path / 'S_trn.npy')
    Y_val = np.load(path / 'Y_val.npy')
    S_val = np.load(path / 'S_val.npy')

    if n_trn:
        Y_trn, S_trn = Y_trn[:, :n_trn], S_trn[:, :n_trn]
    if n_val:
        Y_val, S_val = Y_trn[:, :n_val], S_trn[:, :n_val]

    return (Y_trn, S_trn, Y_val, S_val)
