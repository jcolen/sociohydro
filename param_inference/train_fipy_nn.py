import torch
import torch.nn.functional as F
from fipy_dataset import FipyDataset
from fipy_nn import SociohydroParameterNetwork

import os
import yaml
from argparse import ArgumentParser
from time import time

def train(model, train_loader, val_loader, optimizer, scheduler, epochs, device, savedir='models/ParameterInference'):
    """ Run the full training and validation loop """
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        t = time()

        # Training loop
        train_loss.append(0)
        for i, batch in enumerate(train_loader):
            outputs = model(batch['inputs'].to(device), batch['features'].to(device))
            targets = batch['targets'].to(device)
            loss = F.mse_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss[epoch] += loss.item() / len(train_loader)

        # Validation loop
        val_loss.append(0)
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                outputs = model(batch['inputs'].to(device), batch['features'].to(device))
                targets = batch['targets'].to(device)
                loss = F.mse_loss(outputs, targets)
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
    parser.add_argument('--coef_lr', type=float, default=1e-2)
    parser.add_argument('--base_lr', type=float, default=3e-4)
    parser.add_argument('--scheduler_step', type=float, default=0.98)
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Initialize the model
    model = SociohydroParameterNetwork().to(device)

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.Adam([
        {'params': model.coefs, 'lr': args.coef_lr},
        {'params': model.local_network.parameters()}
    ], lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_step)

    # Build train and val datasets
    dataset = FipyDataset(path=f"./data/{args.county}_small/fipy_output", 
                          grid=args.grid, remove_extra=True)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [1.-args.val_split, args.val_split])
    
    # Initialize data loaders
    dl_args = dict(batch_size=args.batch_size, num_workers=2, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, **dl_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, **dl_args)

    # Save configuration for bookkeeping
    savedir = f'models/{args.county}_{model.__class__.__name__}'
    print(f'Saving to folder {savedir}')
    os.makedirs(savedir, exist_ok=True)
    with open(f'{savedir}/config.yml', 'w') as f:
        yaml.dump(args.__dict__, f)

    train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs, 
        device=device, 
        savedir=savedir
    )