import numpy as np
import os
import glob
import torch
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser

from census_dataset import *
from census_nn import *

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

            for i in tqdm(range(len(train_dataset)), leave=False):
                batch = train_dataset[idxs[i]]
                wb = batch['wb'].to(device)
                wb[wb.isnan()] = 0.
                mask = batch['mask'] # Only use points in county for loss

                wbNN = wb[0] + batch['dt'] * model(wb[0:1])[0]
                loss = F.l1_loss(wbNN[:,mask], wb[1,:,mask])

                loss.backward()
                train_loss.append(loss.item())
                step += 1

                if step % batch_size == 0:
                    opt.step()
                    opt.zero_grad()

            val_loss.append(0)
            model.eval()

            # Our evaluation metric is the forecasting accuracy over 40 years
            with torch.no_grad():
                for dataset in tqdm(val_dataset.datasets, leave=False):
                    batch = dataset[0]
                    wb = batch['wb'].to(device)
                    wb[wb.isnan()] = 0.
                    mask = batch['mask'] # Only use points in county for loss

                    wbNN = model.simulate(wb[0:1], n_steps=wb.shape[0]-1, dt=batch['dt'])[0]
                    val_loss[epoch] += (wbNN[-1,:,mask]-wb[-1,:,mask]).pow(2).sum(0).mean().item()

            # Save if the model showed an improvement
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
    parser.add_argument('--model_id', type=str, default='gridded')
    parser.add_argument('--train_set_seed', type=int, default=0)
    parser.add_argument('--num_train_counties', type=int, default=8)
    args = parser.parse_args()

    # Legacy models in first draft had no train/test split
    train_datasets = []
    val_datasets = []
    for county in args.val_county:
        train_datasets.append(CensusDataset(county))
        val_datasets.append(CensusDataset(county))

    '''
    # In the revision, train on more counties
    counties = [os.path.basename(c)[:-4] for c in glob.glob(
        '/home/jcolen/data/sociohydro/decennial/revision/meshes/*.xml')]
    counties = [c for c in counties if not 'San Bernardino' in c] # Too big, causes memory issues

    rng = np.random.default_rng(args.train_set_seed)
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
    '''
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    for ds in train_dataset.datasets:
        ds.training()
    for ds in val_dataset.datasets:
        ds.validate()
    
    print(f'Training dataset length: {len(train_dataset)}')
    print(f'Validation dataset length: {len(val_dataset)}')

    # Build model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CensusForecasting().to(device)

    savename = f'models/{model.__class__.__name__}_{args.model_id}'
    print(f'Saving information to {savename}')

    with open(f'{savename}_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    with torch.autograd.set_detect_anomaly(True):
        train(model, train_dataset, val_dataset, 
              args.n_epochs, args.batch_size, device, savename)
