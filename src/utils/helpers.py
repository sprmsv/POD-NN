import numpy as np
import torch
from torch.utils.data import Dataset


def changeType(x, target='Tensor'):
    if type(x).__name__ != target:
        if target == 'Tensor':
            return torch.tensor(x)

# CHECK: Could the parameters (y) be complex? NO.
class PairDataset(Dataset):
    def __init__(self, y, c):
        assert y.shape[1] == c.shape[1]
        self.len = c.shape[1]
        self.y = torch.tensor(y.T)
        if np.iscomplexobj(c):
            self.c = torch.tensor(np.concatenate([c.real, c.imag]).T)
        else:
            self.c = torch.tensor(c.T)

    def __getitem__(self, j) -> tuple[torch.Tensor]:
        return self.y[j], self.c[j]

    def __len__(self):
        return self.len

def RelMSE(output, target):
    return torch.mean((torch.norm((output - target), dim=1) / torch.norm(target, dim=1)) ** 2)

def MSE(output, target):
    return torch.mean(torch.norm((output - target), dim=1) ** 2)
