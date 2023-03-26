import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as tvdsets


class MLP(nn.Module):
    def __init__(self, M, L, hidden_layers=[10, 10],
            activation=torch.sigmoid, dropout_probs=None,
            bn=False, dtype=torch.dtype):
        super().__init__()
        layers = [M] + hidden_layers + [L]
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
            torch.nn.init.xavier_uniform_(self.lins[-1].weight)
        for p in dropout_probs:
            self.drops.append(nn.Dropout(p=p))

    def forward(self, x):
        for i, f, bn in zip(range(self.length), self.lins, self.bns):
            if i == 0:
                x = self.activation(bn(f(x)) if self.bn else f(x))
            elif i == len(self.lins) - 1:
                x = self.activation(bn(f(x)) if self.bn else f(x))
            else:
                x = self.activation(bn(f(x)) if self.bn else f(x))
                x = self.drops[i - 1](x)
        return x

    def train_(self, criterion, epochs, optimizer, trainloader, validationloader,
            scheduler=None, cuda=False, report=None):

        if cuda and not torch.cuda.is_available():
            raise Exception('CUDA is not available.')
        if cuda:
            self.cuda()
            criterion.cuda()
            optimizer.cuda()  # ?

        stats = {
            'epoch': [0],
            'loss_trn': [None],
            'loss_val': [None],
            'lr': [None],
            'params': {name: [p.data] for name, p in self.named_parameters()},
            'grads': {name: [None] for name, _ in self.named_parameters()}
        }
        for epoch in range(epochs):

            loss_trn = 0
            for x, y in trainloader:
                self.train()
                if cuda:
                    x, y = x.cuda(), y.cuda()
                y_ = self(x)
                loss = criterion(y_, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_trn += loss.item()
            if scheduler:
                scheduler.step()

            loss_val = 0
            for x, y in validationloader:
                self.eval()
                if cuda:
                    x, y = x.cuda(), y.cuda()
                y_ = self(x)
                loss = criterion(y_, y)
                loss_val += loss.item()

            # Store statistics
            stats['epoch'].append(epoch+1)
            stats['loss_trn'].append(loss_trn)
            stats['loss_val'].append(loss_val)
            stats['lr'].append(optimizer.param_groups[0]['lr'])
            for name, param in self.named_parameters():
                stats['params'][name].append(param.data)
                stats['grads'][name].append(param.grad)

            # Print statistics
            if report and (epoch % report) == 0:
                print('\t'.join([
                    f'EPOCH {epoch:05d}',
                    f'TRN {loss_trn:.2e}',
                    f'VAL {loss_val:.2e}' if loss_val else 'VAL N\A',
                    f'LR {optimizer.param_groups[0]["lr"]:.2e}'
                ]))

        return stats

    def __len__(self):
        return self.length - 2

    def numparams(self):
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count
