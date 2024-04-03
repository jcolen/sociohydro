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
    best_loss = 1e10

    idxs = np.arange(len(train_dataset), dtype=int)
    
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            model.train()

            np.random.shuffle(idxs)
            d_ad.set_working_tape(d_ad.Tape())

            for i in tqdm(range(len(train_dataset)), leave=False):
                batch = train_dataset[idxs[i]]
                batch['wb0'] = batch['wb0'].to(device)

                params, J = model.training_step(batch)
                train_loss.append(J)
                step += 1

                if step % batch_size == 0:
                    opt.step()
                    d_ad.set_working_tape(d_ad.Tape())
                    opt.zero_grad()

            val_loss.append(0)
            model.eval()

            # Change our evaluation metric to the forecasting accuracy over 40 years
            for dataset in tqdm(val_dataset.datasets, leave=False):
                dataset.validate()
                ic = dataset[0]
                ic['wb0'] = ic['wb0'].to(device)
                mask = ic['mask']
                wb2020 = torch.FloatTensor(dataset[39]['wb1']).to(device)

                with torch.no_grad(), d_ad.stop_annotating():
                    wbNN = model.simulate(ic, dataset.mesh, device, tmax=40, dt=1)
                    val_loss[epoch] += (wbNN - wb2020).pow(2).sum(0)[mask].mean().item()

            if val_loss[epoch] <= best_loss:
                torch.save(
                    {
                        'state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': train_loss,
                    },
                    f'{savename}.ckpt')
                best_loss = val_loss[epoch]

            sch.step()
            pbar.update()
            pbar.set_postfix(val_loss=val_loss[epoch])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_county', nargs='+', 
        default=['Georgia_Fulton', 'Illinois_Cook', 'Texas_Harris', 'California_Los Angeles'])
    parser.add_argument('--val_tmax', type=int, default=10)
    parser.add_argument('--model_id', type=str, default='new_validation')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--num_train_counties', type=int, default=8)
    args = parser.parse_args()

    '''
    # Legacy models in first draft had no train/test split
    datasets = []
    for county in args.county:
        datasets.append(CensusDataset(county))
    dataset = torch.utils.data.ConcatDataset(datasets)
    '''

    # In the revision, train on more counties
    counties = [os.path.basename(c)[:-4] for c in glob.glob(
        '/home/jcolen/data/sociohydro/decennial/revision/meshes/*.xml')]
    counties = [c for c in counties if not 'San Bernardino' in c] # Too big, causes memory issues

    rng = np.random.default_rng(args.random_seed)
    train_counties = rng.choice(counties, args.num_train_counties)
    args.__dict__['train_county'] = list(train_counties)

    train_datasets = [CensusDataset(county) for county in train_counties]
    val_datasets = [CensusDataset(county) for county in args.val_county]
    
    # Include first decade of Census data in from Figure 3 counties
    for county in args.val_county:
        train_datasets.append(torch.utils.data.Subset(
            CensusDataset(county), 
            np.arange(0, args.val_tmax, dtype=int)
        ))
    
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
    print(f'Training dataset length: {len(train_dataset)}')
    print(f'Validation dataset length: {len(val_dataset)}')

    # Build model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CensusPBNN().to(device)

    savename = f'models/{model.__class__.__name__}_{args.model_id}'
    print(f'Saving information to {savename}')

    with open(f'{savename}_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    with torch.autograd.set_detect_anomaly(True):
        train(model, train_dataset, val_dataset, 
              args.n_epochs, args.batch_size, device, savename)
