import torch
import numpy as np
import os
import glob
import json
import pandas as pd
from tqdm.auto import tqdm
from argparse import ArgumentParser

from forecasting_dataset import *
from forecasting_nn import *
  
def train(model, train_dataset, val_dataset, n_epochs, batch_size, device, 
          savename='models/simulation/SimulationForecasting'):
    '''
    Train a model
    '''
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    train_loss = []
    val_loss = []
    
    idxs = np.arange(len(train_dataset), dtype=int)

    dl_args = dict(batch_size=batch_size, num_workers=2, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, **dl_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, **dl_args)
    
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            with tqdm(total=len(train_loader), leave=False) as ebar:
                for i, batch in enumerate(train_loader):
                    for key in batch:
                        batch[key] = batch[key].to(device)
                    loss = model.batch_step(batch)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    train_loss.append(loss.item())
                    ebar.set_postfix(loss=np.mean(train_loss[-2:]))
                    ebar.update()

            val_loss.append(0)
            with tqdm(total=len(val_loader), leave=False) as ebar:
                with torch.no_grad():
                    for i, batch in enumerate(val_loader):
                        for key in batch:
                            batch[key] = batch[key].to(device)
                        loss = model.batch_step(batch)
                        val_loss[epoch] += loss.item()
                        ebar.update()

            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                },
                f'{savename}.ckpt')

            sch.step()
            pbar.update()
            pbar.set_postfix(val_loss=val_loss[-1])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--savename', type=str, default='./models/simulation/SimulationForecasting')
    parser.add_argument('--dynamic_type', type=str, default='segregated')
    args = parser.parse_args()
    
    info = pd.read_csv('/home/jcolen/data/sociohydro/2024-02-05_MCPhaseDiagram/data/dynamic_keys.csv')
    folders = info.loc[info.dynamic_type == args.dynamic_type, '#file'].values
    print(len(folders))
    #folders = folders[0:5]

    N = len(folders)
    Nc = len(folders) * 3 // 5
    train_folders = list(np.random.choice(folders, Nc))
    val_folders = [c for c in folders if not c in train_folders]
    print('Training on ', train_folders, flush=True)
    print('Validating on ', val_folders, flush=True)
    args.__dict__['train_folder'] = train_folders
    args.__dict__['val_folder'] = val_folders

    with open(f'{args.savename}_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train_datasets = []
    for folder in args.train_folder:
        train_datasets.append(SimulationDataset(folder))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        
    val_datasets = []
    for folder in args.val_folder:
        val_datasets.append(SimulationDataset(folder))
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SimulationForecasting().to(device)
    
    with torch.autograd.set_detect_anomaly(True): #Sometimes this just happens and I don't know why
        train(model, train_dataset, val_dataset, 
              args.n_epochs, args.batch_size, device, args.savename)
