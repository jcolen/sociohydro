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

            if val_loss[-1] <= best_loss:
                torch.save(
                    {
                        'state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': train_loss,
                    },
                    f'{savename}.ckpt')
                best_loss = val_loss[-1]

            sch.step()
            pbar.update()
            pbar.set_postfix(val_loss=val_loss[-1])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--val_county', nargs='+', 
        default=['Georgia_Fulton', 'Illinois_Cook', 'Texas_Harris', 'California_Los Angeles'])
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_id', type=str, default='revision')
    args = parser.parse_args()

    '''
    # Legacy models in first draft had no train/test split
    datasets = []
    for county in args.county:
        datasets.append(CensusDataset(county))
    dataset = torch.utils.data.ConcatDataset(datasets)
    '''

    '''
    In the revision, train on more counties
    For the counties in Figure 3 use only the first decade of Census data
    '''
    counties = [os.path.basename(c)[:-4] for c in glob.glob(
        '/home/jcolen/data/sociohydro/decennial/revision/meshes/*.xml')]
    counties = [c for c in counties if not 'San Bernardino' in c] # Too big, causes memory issues

    # Build datasets
    train_datasets = []
    val_datasets = []
    train_idxs = np.arange(0, 10, dtype=int) # First decade goes to training
    val_idxs = np.arange(10, 40, dtype=int) # Remaining 3 decades for validation
    for county in counties:
        ds = CensusDataset(county)
        if county in args.val_county:
            train_datasets.append(torch.utils.data.Subset(ds, train_idxs))
            val_datasets.append(torch.utils.data.Subset(ds, val_idxs))
        else:
            train_datasets.append(ds)
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
    print(f'Training dataset length: {len(train_dataset)}')
    print(f'Validation dataset length: {len(val_dataset)}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CensusPBNN().to(device)

    savename = f'models/{model.__class__.__name__}_{args.model_id}'
    print(f'Saving information to {savename}')

    with open(f'{savename}_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    with torch.autograd.set_detect_anomaly(True):
        train(model, train_dataset, val_dataset, 
              args.n_epochs, args.batch_size, device, savename)

        # Load best model from checkpoint
        info = torch.load(f'{savename}.ckpt')
        model.load_state_dict(info['state_dict'])
        
        #for dataset in val_dataset.datasets:
        #    compute_saliency(model, dataset, device, savename)
