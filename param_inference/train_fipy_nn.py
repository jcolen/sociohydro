import torch
import torch.nn as nn
import torch.nn.functional as F
from fipy_dataset import FipyDataset
from fipy_nn import SociohydroParameterNetwork

import os
import yaml
import datetime
from argparse import ArgumentParser
from time import time

class JointLossFunction(nn.Module):
    """ Multi-directional loss function 
        Loss = MSE(features+growth, target) + \
            alpha * MSE(features, target) + \
            beta * MSE(features+SINDy(growth), target)
        
        The first term is a simple MSE to improve accuracy
        The second term encourages the model to do as much as possible without the NN
        The third term encourages the NN to have a parsimonious representation
    """
    def __init__(self, alpha=0., beta=0.):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

    def polynomial_library(self, x):
        """ Return polynomial combinations up to degree 2 
            x has shape [B, 2, ...]
            Return something of shape [..., 6]
        """
        lib = [
            torch.ones_like(x[:, 0]),
            x[:, 0],
            x[:, 1],
            x[:, 0]**2,
            x[:, 1]**2,
            x[:, 0] * x[:, 1]
        ]
        lib = torch.stack(lib, dim=-1)
        return lib.reshape([-1, lib.shape[-1]])
    
    def forward(self, targets, outputs, inputs, features, growth):
        """ Apply the multi-directional loss function """
        loss = F.mse_loss(targets, outputs)

        if self.alpha > 0:
            loss += self.alpha * F.mse_loss(targets, features)

        if self.beta > 0:
            # Get polynomial feature library, shape [N, 6]
            A = self.polynomial_library(inputs) # shape [N, 6]
            b = torch.movedim(growth, 1, -1).reshape([-1, growth.shape[1]]) #shape [N, 2]

            # Solve least squares problem
            weights = torch.linalg.lstsq(A, b).solution

            # Get least squares predictions
            growth_pred = A @ weights
            growth_pred = growth_pred.reshape([growth.shape[0], -1, growth.shape[1]])
            growth_pred = torch.movedim(growth_pred, -1, 1)
            loss += self.beta * F.mse_loss(targets, features + growth_pred)
        
        return loss

def train(model, 
          train_loader, 
          val_loader, 
          loss_func,
          optimizer, 
          scheduler, 
          epochs, 
          device, 
          savedir='models/ParameterInference'):
    """ Run the full training and validation loop """
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        t = time()

        # Training loop
        train_loss.append(0)
        for i, batch in enumerate(train_loader):
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            outputs, (features, growth) = model(
                inputs, batch['features'].to(device))
            loss = loss_func(targets, outputs, inputs, features, growth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss[epoch] += loss.item() / len(train_loader)

        # Validation loop
        val_loss.append(0)
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                outputs, (features, growth) = model(
                    inputs, batch['features'].to(device))
                loss = loss_func(targets, outputs, inputs, features, growth)
                val_loss[epoch] += loss.item() / len(val_loader)

        # Bookkeeping
        scheduler.step()
        print(f'Epoch = {epoch:03d}\tTrain loss = {train_loss[-1]:.02e}\tVal loss = {val_loss[-1]:.02e}\tTime = {int(time()-t):d} s')
        model.print(indent=1)

        # Save results
        torch.save(
            {
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            },
            f'{savedir}/model.ckpt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--county', type=str, default='Georgia_Fulton')
    parser.add_argument('--coef_lr', type=float, default=1e-1)
    parser.add_argument('--base_lr', type=float, default=3e-4)
    parser.add_argument('--scheduler_step', type=float, default=0.98)
    parser.add_argument('--val_split', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=0.)
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Initialize the model
    model = SociohydroParameterNetwork(grid=args.grid).to(device)

    # Initialize optimizer and learning rate scheduler
    loss_func = JointLossFunction(alpha=args.alpha, beta=args.beta)
    optimizer = torch.optim.Adam([
        {'params': model.coefs, 'lr': args.coef_lr},
        {'params': model.local_network.parameters()}
    ], lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_step)

    # Build train and val datasets
    dataset = FipyDataset(path=f"./data/{args.county}_small/fipy_output", 
                          grid=args.grid, remove_extra=True, preload=True)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [1.-args.val_split, args.val_split])
    print(f'Train dataset has {len(train_dataset)} elements')
    print(f'Validation dataset has {len(val_dataset)} elements')

    # Initialize data loaders
    dl_args = dict(batch_size=args.batch_size, num_workers=2, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, **dl_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, **dl_args)

    # Save configuration for bookkeeping
    dt = datetime.datetime.now()
    time_code = dt.strftime('%d%m%y_%H%M')
    savedir = f'models/{args.county}_{"Grid" if args.grid else "noGrid"}_{model.__class__.__name__}_{time_code}'
    print(f'Saving to folder {savedir}')
    os.makedirs(savedir, exist_ok=True)
    with open(f'{savedir}/config.yml', 'w') as f:
        yaml.dump(args.__dict__, f)

    train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs, 
        device=device, 
        savedir=savedir
    )