import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as tvdsets
from typing import Callable
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, M, L, V, hidden_layers=[10, 10],
            activation: Callable = torch.relu,
            dropout_probs=None, bn=False, gain: float = 1.,
            dtype=torch.dtype):

        # Store parameters
        self.L = L
        self.V = torch.tensor(V)
        self.complex = torch.is_complex(self.V)

        # Build the layers, batch norms, dropouts
        super().__init__()
        layers = [M] + hidden_layers + [(2 * L) if self.complex else L]
        if dropout_probs:
            assert len(dropout_probs) == len(layers) - 2
        else:
            dropout_probs = [0] * (len(layers) - 2)
        self.length = len(layers)
        self.activation = activation
        self.bn = bn
        self.lins = nn.ModuleList()
        self.drops = nn.ModuleList()
        self.bns = nn.ModuleList()
        for input, output in zip(layers, layers[1:]):
            self.lins.append(nn.Linear(input, output, dtype=dtype))
            self.bns.append(nn.BatchNorm1d(output, dtype=dtype))
            torch.nn.init.uniform_(self.lins[-1].weight, a=-gain, b=gain)
            torch.nn.init.constant_(self.lins[-1].bias, val=0)
        for p in dropout_probs:
            self.drops.append(nn.Dropout(p=p))

    def forward(self, y):
        for i, f, bn in zip(range(self.length), self.lins, self.bns):
            if i == 0:
                y = self.activation(bn(f(y)) if self.bn else f(y))
            elif i == len(self.lins) - 1:
                # NOTE: No activation on the last layer
                c = bn(f(y)) if self.bn else f(y)
            else:
                y = self.activation(bn(f(y)) if self.bn else f(y))
                y = self.drops[i - 1](y)

        S, c = self.project(c)

        return S, c

    def project(self, c):
        # Fetch L
        L = self.L

        # Build complex coefficients if necessary
        if self.complex:
            c = (c[:, :L] + 1j * c[:, L:])

        # Change the basis
        S = (self.V @ c.T).T

        # Separate the real and the imaginary parts if necessary
        # if self.complex:
        #     S = torch.concatenate([S.real, S.imag], dim=1)

        return S, c


    def train_(self, criterion, epochs, optimizer, trainloader, validationloader,
            scheduler=None, cuda=False, store_params=False):

        if cuda and not torch.cuda.is_available():
            raise Exception('CUDA is not available.')
        if cuda:
            self.cuda()
            criterion.cuda()
            optimizer.cuda()

        stats = {
            'epoch': [0],
            'loss_trn': [None],
            'loss_val': [None],
            'lr': [None],
        }
        params = {
            'params': {name: [p.data.detach().clone().numpy()] for name, p in self.named_parameters()},
            'grads': {name: [None] for name, _ in self.named_parameters()}
        }
        for epoch in tqdm(range(epochs)):

            loss_trn = 0
            for y, s in trainloader:
                self.train()
                if cuda:
                    y, s = y.cuda(), s.cuda()
                s_, c_ = self(y)
                loss = criterion(s_, s)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_trn += loss.item()
            if scheduler:
                scheduler.step()

            loss_val = 0
            for y, s in validationloader:
                self.eval()
                if cuda:
                    y, s = y.cuda(), s.cuda()
                s_, c_ = self(y)
                loss = criterion(s_, s)
                loss_val += loss.item()

            # Store statistics
            stats['epoch'].append(epoch+1)
            stats['loss_trn'].append(loss_trn)
            stats['loss_val'].append(loss_val)
            stats['lr'].append(optimizer.param_groups[0]['lr'])
            if store_params:
                for name, param in self.named_parameters():
                    params['params'][name].append(param.data.detach().clone().numpy())
                    params['grads'][name].append(param.grad.detach().clone().numpy() if (param.grad is not None) else None)

        return stats, params

    def __len__(self):
        return self.length - 2

    def numparams(self):
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count
