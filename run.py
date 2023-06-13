import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import modred as mr
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.models import MLP
from src.utils.helpers import MSE, PairDataset, RelMSE, read_data

sns.set_theme(style='whitegrid', palette='deep')

def get_parser():
    """Get argument parser"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--datadir', type=str, required=True, dest='datadir',
        help='Directory of data',
    )

    parser.add_argument(
        '--n_trn', type=int, default=0, dest='n_trn',
        help='Number of training snapshots',
    )

    parser.add_argument(
        '--n_val', type=int, default=0, dest='n_val',
        help='Number of validation snapshots',
    )

    parser.add_argument(
        '-L', type=int, required=True, dest='L',
        help='Dimension of the reduced basis',
    )

    parser.add_argument(
        '--repeats', type=int, default=1, dest='repeats',
        help='Repeat the training n times',
    )

    parser.add_argument(
        '-D', type=int, default=2, dest='D',
        help='Network depth',
    )

    parser.add_argument(
        '-W', type=int, default=10, dest='W',
        help='Network width',
    )

    parser.add_argument(
        '--normalize-output', action='store_true', dest='normalizeout',
        help='If passed, the outputs will be normalized.',
    )

    parser.add_argument(
        '--normalize-input', action='store_true', dest='normalizein',
        help='If passed, the inputs will be normalized.',
    )

    parser.add_argument(
        '--epochs', type=float, default=1000, dest='epochs',
        help='Number of training epochs',
    )

    parser.add_argument(
        '--lr', type=float, default=5e-03, dest='lr',
        help='Learning rate',
    )

    parser.add_argument(
        '--wd', type=float, default=0, dest='wd',
        help='Weight decay',
    )

    parser.add_argument(
        '--scheduler', action='store_true', dest='scheduler',
        help='Use a scheduler for learning rate',
    )

    return parser

def train_model(Y_trn, S_trn, Y_val, S_val, args):
    """Train"""

    # Check the inputs
    assert Y_trn.shape[0] == Y_val.shape[0]
    assert S_trn.shape[0] == S_val.shape[0]

    # Settings
    bsz = None
    batch_norm = False
    if batch_norm and bsz: assert bsz > 1

    # Get the reduced basis
    pod = mr.compute_POD_arrays_snaps_method(S_trn, list(mr.range(args.L)))
    V = pod.modes

    # Get the projected coefficients
    C_trn = V.conj().T @ S_trn
    C_val = V.conj().T @ S_val


    # Get mean and std of the inputs
    if args.normalizein:
        mean_Y_trn, std_Y_trn = Y_trn.mean(axis=1, keepdims=True), Y_trn.std(axis=1, keepdims=True)
    else:
        mean_Y_trn, std_Y_trn = (0., 1.)

    # Get mean and std of outputs
    if args.normalizeout:
        mean_C_trn, std_C_trn = C_trn.mean(axis=1, keepdims=True), C_trn.std(axis=1, keepdims=True)
        mean_C_val, std_C_val = C_val.mean(axis=1, keepdims=True), C_val.std(axis=1, keepdims=True)
    else:
        mean_C_trn, std_C_trn = (0., 1.)
        mean_C_val, std_C_val = (0., 1.)

    # Define the data loaders
    trainloader = DataLoader(
        dataset=PairDataset(Y=Y_trn, C=((C_trn - mean_C_trn) / std_C_trn)),
        batch_size=(bsz if bsz else Y_trn.shape[1]),
        shuffle=True,
        drop_last=batch_norm,
    )
    validationloader = DataLoader(
        dataset=PairDataset(Y=Y_val, C=((C_val - mean_C_val) / std_C_val)),
        batch_size=(bsz if bsz else Y_val.shape[1]),
        shuffle=False,
    )

    # Define model, criterion, optimizer, and scheduler
    model = MLP(
        M=Y_trn.shape[0],
        L=args.L,
        complex=False,
        meanstd_in=(mean_Y_trn, std_Y_trn),
        meanstd_out=(mean_C_trn, std_C_trn),
        hidden_layers=([args.W] * args.D),
        activation=torch.tanh,
        gain=1.,
        dtype=torch.float64,
    )
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=.9,
        weight_decay=args.wd,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-04, T_max=args.epochs)
        if args.scheduler else None
    )

    # Train the model
    print(f'{model.__class__.__name__} with {model.numparams()} parameters')
    stats = model.train_(
        criterion=RelMSE,
        error=RelMSE,
        V=V,
        epochs=args.epochs,
        optimizer=optimizer,
        trainloader=trainloader,
        validationloader=validationloader,
        mod=100,
        scheduler=scheduler,
        cuda=False,
    )

    return stats

if __name__ == '__main__':
    args = get_parser().parse_args()
    args.epochs = int(args.epochs)

    # Read the data
    datadir = Path(args.datadir)
    assert datadir.exists()
    Y_trn, S_trn, Y_val, S_val = read_data(path=datadir, n_trn=args.n_trn, n_val=args.n_val)

    # Set experiment name
    args.name = f'ntrn{args.n_trn:04d}_nval{args.n_val:04d}_L{args.L:03d}_D{args.D:02d}_W{args.W:03d}_lr{args.lr:.2e}'
    print(f'DATA: {datadir.as_posix()}')
    print(f'NAME: {args.name}')

    # Train a model
    stats = train_model(Y_trn, S_trn, Y_val, S_val, args)

    # Print the final losses and errors
    print('\t\t'.join([
        'LOSS ::',
        f"Train: {stats['loss_trn'][-1]:.2e}",
        f"Validation: {stats['loss_val'][-1]:.2e}",
    ]))
    print('\t\t'.join([
        'ERRS ::',
        f"Train: {stats['err_trn'][-1]:.2e}",
        f"Validation: {stats['err_val'][-1]:.2e}",
    ]))

    # Set the path of the results
    results = Path('./results') / datadir.relative_to('./data')
    results.mkdir(exist_ok=True, parents=True)
    # Store the stats
    df = pd.DataFrame(stats)
    df.to_csv(results / (args.name + '.csv'))
    # Store the plot of the stats
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    sns.lineplot(df, x='epoch', y='loss_trn', ax=axs[0], label='Training')
    sns.lineplot(df, x='epoch', y='loss_val', ax=axs[0], label='Validation')
    axs[0].set(ylabel='Loss', yscale='log', ylim=[1e-06, 1e04])
    axs[0].legend()
    sns.lineplot(df, x='epoch', y='err_trn', ax=axs[1], label='Training')
    sns.lineplot(df, x='epoch', y='err_val', ax=axs[1], label='Validation')
    axs[1].set(ylabel='Relative Error', yscale='log', ylim=[1e-06, 1e04])
    axs[1].legend()
    fig.savefig(results / (args.name + '.png'))
