import numpy as np
import os
import glob
import torch
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser

from census_dataset import *
from census_pbnn import *

def train(model, train_dataset, val_dataset, n_epochs, batch_size, device, savename):
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
            model.train()

            np.random.shuffle(idxs)
            d_ad.set_working_tape(d_ad.Tape())

            # Schedule increase in loss weight of segregation
            if epoch == 50:
                print('Setting Dolfin alpha to 0.1')
                for ds in train_dataset.datasets:
                    ds.dolfin_kwargs = {'alpha': 0.1}

            with tqdm(total=len(train_dataset), leave=False) as ebar:
                for i in range(len(train_dataset)):
                    batch = train_dataset[idxs[i]]
                    batch['wb0'] = batch['wb0'].to(device)

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
            model.eval()

            with tqdm(total=len(val_dataset), leave=False) as ebar:
                with torch.no_grad():
                    for i in range(len(val_dataset)):
                        d_ad.set_working_tape(d_ad.Tape())
                        batch = val_dataset[i]
                        batch['wb0'] = batch['wb0'].to(device)

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
    #parser.add_argument('--county', nargs='+', default=['Georgia_Fulton', 'Illinois_Cook', 'Texas_Harris', 'California_Los Angeles'])
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_id', type=int, default=0)
    args = parser.parse_args()

    counties = [os.path.basename(c)[:-4] for c in glob.glob(
        '/home/jcolen/data/sociohydro/decennial/revision/meshes/*.xml')]
    counties = [c for c in counties if not 'San Bernardino' in c] # Too big, causes memory issues
    N = len(counties)
    Nv = len(counties) // 3

    rng = np.random.default_rng(1)
    rng.shuffle(counties)
    counties = np.roll(counties, Nv * args.model_id)

    train_counties = counties[Nv:]
    val_counties = counties[:Nv]
    args.__dict__['train_county'] = list(train_counties)
    args.__dict__['val_county'] = list(val_counties)

    print('Training counties')
    print('-----------------')
    train_datasets = []
    for county in args.train_county:
        print(county)
        train_datasets.append(CensusDataset(county))
        train_datasets[-1].training()
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    print()
    print('Validation counties')
    print('-------------------')
    val_datasets = []
    for county in args.val_county:
        print(county)
        val_datasets.append(CensusDataset(county))
        val_datasets[-1].validate()
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    '''
    # Legacy models had no train/test split
    datasets = []
    for county in args.county:
        datasets.append(CensusDataset(county))
    dataset = torch.utils.data.ConcatDataset(datasets)
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CensusPBNN().to(device)

    savename = f'models/{model.__class__.__name__}_scheduled_{args.model_id}'
    print(f'Saving information to {savename}')

    with open(f'{savename}_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    with torch.autograd.set_detect_anomaly(True):
        train(model, train_dataset, val_dataset, 
              args.n_epochs, args.batch_size, device, savename)
        
        for dataset in val_dataset.datasets:
            compute_saliency(model, dataset, device, savename)
