import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as tvdsets
from typing import Callable
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, M, L, hidden_layers=[10, 10],
            activation: Callable = torch.tanh,
            meanstd_in: tuple = (0., 1.),
            meanstd_out: tuple = (0., 1.),
            dropout_probs=None, bn=False, gain: float = 1.,
            complex: bool = False,
            dtype: torch.dtype = torch.float64):

        # Store the parameters
        self.L = L
        self.complex = complex
        self.activation = activation
        self.bn = bn

        # Store the means and standard deviations
        self.mean_in = torch.tensor(meanstd_in[0]).reshape(1, -1)
        self.std_in = torch.tensor(meanstd_in[1]).reshape(1, -1)
        self.mean_out = torch.tensor(meanstd_out[0]).reshape(1, -1)
        self.std_out = torch.tensor(meanstd_out[1]).reshape(1, -1)

        # Build the layers, batch norms, dropouts
        super().__init__()
        layers = [M] + hidden_layers + [(2 * L) if self.complex else L]
        if dropout_probs:
            assert len(dropout_probs) == len(layers) - 2
        else:
            dropout_probs = [0] * (len(layers) - 2)
        self.length = len(layers)
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
        # Normalize the input
        y = (y - self.mean_in) / self.std_in

        # Get the output
        for i, f, bn in zip(range(self.length), self.lins, self.bns):
            if i == 0:
                y = self.activation(bn(f(y)) if self.bn else f(y))
            elif i == len(self.lins) - 1:
                # NOTE: No activation on the last layer
                c = bn(f(y)) if self.bn else f(y)
            else:
                y = self.activation(bn(f(y)) if self.bn else f(y))
                y = self.drops[i - 1](y)

        return c

    def project(self, c, V):
        """
        Projects the output of the network to the high-fidelity basis with samples on the columns.
        The output is unnormalized before being projected.
        """

        # Fetch L
        L = self.L

        # Build complex coefficients if necessary
        if self.complex:
            c = (c[:, :L] + 1j * c[:, L:])

        # Unnormalize the output
        c = self.std_out * c + self.mean_out

        # Change the basis
        S = (V @ c.T).T

        return S

    def train_(
        self,
        criterion,
        error,
        V,
        epochs,
        optimizer,
        trainloader,
        validationloader,
        mod=100,
        scheduler=None,
        cuda=False,
    ):
        """
        Trains the network.
        The dataloaders should return non-normalized inputs and normalized outputs.
        """

        if cuda and not torch.cuda.is_available():
            raise Exception('CUDA is not available.')
        if cuda:
            self.cuda()
            criterion.cuda()
            optimizer.cuda()

        V = torch.tensor(V)

        stats = {
            'epoch': [0],
            'lr': [None],
            'loss_trn': [None],
            'loss_val': [None],
            'err_trn': [None],
            'err_val': [None],
        }
        for epoch in tqdm(range(epochs), mininterval=1):
            # Train one epoch
            loss_trn = 0
            self.train()
            for y, c in trainloader:
                if cuda: y, c = y.cuda(), c.cuda()
                c_ = self(y)
                loss = criterion(c_, c)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_trn += loss.item()

            # Change the learning rate
            if scheduler:
                scheduler.step()

            # Get stats
            if (epoch % mod == 0) or (epoch == (epochs - 1)):
                # Set to evaluation mode
                self.eval()

                # Get validation loss
                loss_val = 0
                for y, c in validationloader:
                    if cuda: y, c = y.cuda(), c.cuda()
                    c_ = self(y)
                    loss = criterion(c_, c)
                    loss_val += loss.item()

                # Get training error
                err_trn = 0
                for y, c in trainloader:
                    if cuda: y, c = y.cuda(), c.cuda()
                    c_ = self(y)
                    err = error(self.project(c_, V), self.project(c, V))
                    err_trn += err.item()

                # Get validation error
                err_val = 0
                for y, c in validationloader:
                    if cuda:
                        y, c = y.cuda(), c.cuda()
                    c_ = self(y)
                    err = error(self.project(c_, V), self.project(c, V))
                    err_val += err.item()

                # Store statistics
                stats['epoch'].append(epoch+1)
                stats['lr'].append(optimizer.param_groups[0]['lr'])
                stats['loss_trn'].append(loss_trn)
                stats['loss_val'].append(loss_val)
                stats['err_trn'].append(err_trn)
                stats['err_val'].append(err_val)

        return stats

    def __len__(self):
        return self.length - 2

    def numparams(self):
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count
