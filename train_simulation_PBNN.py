import torch
import numpy as np
import os
import glob
import json
import pandas as pd
from tqdm.auto import tqdm
from argparse import ArgumentParser

from data_processing import *
from pbnn import *

'''
Training scripts
'''         
def train(model, train_dataset, val_dataset, n_epochs, batch_size, device, 
          savename='models/simulation/SimulationPBNN'):
    '''
    Train a model
    '''
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    train_loss = []
    val_loss = []
    step = 0
    
    idxs = np.arange(len(train_dataset), dtype=int)

    
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            np.random.shuffle(idxs)
            d_ad.set_working_tape(d_ad.Tape())

            with tqdm(total=len(train_dataset), leave=False) as ebar:
                for i in range(len(train_dataset)):
                    batch = train_dataset[idxs[i]]
                    batch['ab0'] = batch['ab0'].to(device)

                    params, J = model.training_step(batch)
                    train_loss.append(J)
                    step += 1
                    ebar.update()

                    if step % batch_size == 0:
                        ebar.set_postfix(loss=np.mean(train_loss[-batch_size:]))

                        opt.step()
                        d_ad.set_working_tape(d_ad.Tape())
                        opt.zero_grad()

            val_loss.append(0)

            with tqdm(total=len(val_dataset), leave=False) as ebar:
                with torch.no_grad():
                    for i in range(len(val_dataset)):
                        d_ad.set_working_tape(d_ad.Tape())
                        batch = val_dataset[i]
                        batch['ab0'] = batch['ab0'].to(device)

                        params, J = model.validation_step(batch)
                        val_loss[epoch] += J
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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--savename', type=str, default='./models/simulation/SimulationPBNN')
    args = parser.parse_args()
    
    info = pd.read_csv('/home/jcolen/data/sociohydro/2024-02-05_MCPhaseDiagram/data/dynamic_keys.csv')
    folders = info.loc[info.dynamic_type == 'segregated', '#file'].values
    folders = folders[0:1]

    N = len(folders)
    Nc = len(folders) * 3 // 5
    train_folders = list(np.random.choice(folders, Nc))
    val_folders = [c for c in folders if not c in train_folders]
    train_folders = list(folders)
    val_folders = list(folders)
    print('Training on ', train_folders, flush=True)
    print('Validating on ', val_folders, flush=True)
    args.__dict__['train_folder'] = train_folders
    args.__dict__['val_folder'] = val_folders

    with open(f'{args.savename}_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train_datasets = []
    for folder in args.train_folder:
        train_datasets.append(SimulationDataset(folder))
        train_datasets[-1].training()
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        
    val_datasets = []
    for folder in args.val_folder:
        val_datasets.append(SimulationDataset(folder))
        val_datasets[-1].validate()
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SimulationPBNN().to(device)
    
    with torch.autograd.set_detect_anomaly(True):
        train(model, train_dataset, val_dataset, 
              args.n_epochs, args.batch_size, device, args.savename)
