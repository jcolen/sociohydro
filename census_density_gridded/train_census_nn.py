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
    step = 0
    best_loss = 1e10
    best_epoch = 0

    idxs = np.arange(len(train_dataset), dtype=int)
    
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            model.train()
            train_loss = np.zeros(len(train_dataset))
            np.random.shuffle(idxs)

            # Training loop
            for i in tqdm(range(len(train_dataset)), leave=False):
                batch = train_dataset[idxs[i]]
                wb = batch['wb'].to(device)
                wb[wb.isnan()] = 0.
                mask = batch['mask'] # Only use points in county for loss

                wbNN = wb[0] + batch['dt'] * model(wb[0:1])[0]
                loss = F.l1_loss(wbNN[:,mask], wb[1,:,mask])

                loss.backward()
                train_loss[i] += loss.item()
                step += 1

                if step % batch_size == 0:
                    opt.step()
                    opt.zero_grad()

            val_loss = 0.
            model.eval()

            # Our evaluation metric is the forecasting error over the full county time series
            with torch.no_grad():
                for dataset in tqdm(val_dataset.datasets, leave=False):
                    batch = dataset[0]
                    wb = batch['wb'].to(device)
                    wb[wb.isnan()] = 0.
                    mask = batch['mask'] # Only use points in county for loss

                    wbNN = model.simulate(wb[0:1], n_steps=wb.shape[0]-1, dt=batch['dt'])[0]
                    val_loss += (wbNN[-1,:,mask]-wb[-1,:,mask]).pow(2).sum(0).mean().item()

            # Save if the model showed an improvement
            if val_loss <= best_loss:
                torch.save(
                    {
                        'state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': np.mean(train_loss),
                    },
                    f'{savename}.ckpt')
                best_loss = val_loss
                best_epoch = epoch
            
            # Early stopping if the model is no longer improving
            if epoch - best_epoch > 20:
                return

            sch.step()
            pbar.update()
            pbar.set_postfix(val_loss=val_loss, best_loss=best_loss)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_county', nargs='+', 
        default=['Georgia_Fulton', 'Illinois_Cook', 'Texas_Harris', 'California_Los Angeles'])
    parser.add_argument('--val_tmax', type=int, default=10)
    parser.add_argument('--use_max_scaling', action='store_true')
    parser.add_argument('--use_fill_frac', action='store_true')
    parser.add_argument('--model_id', type=str, default='test')
    args = parser.parse_args()

    '''
    # Legacy models in first draft had no train/test split
    # This was used for CensusForecasting_gridded.ckpt
    train_datasets = []
    val_datasets = []
    for county in args.val_county:
        train_datasets.append(CensusDataset(county).training())
        val_datasets.append(CensusDataset(county).validate())
    '''

    # In the revision, train on more counties
    counties = [os.path.basename(c)[:-4] for c in glob.glob(
        '/home/jcolen/data/sociohydro/decennial/revision/meshes/*.xml')]
    counties = [c for c in counties if not 'San Bernardino' in c] # Too big, causes memory issues
    train_counties = [c for c in counties if not c in args.val_county]

    args.__dict__['num_train_counties'] = len(train_counties)
    args.__dict__['train_county'] = list(train_counties)

    dataset_kwargs = dict(
        use_fill_frac=args.use_fill_frac,
        use_max_scaling=args.use_max_scaling,
    )

    train_datasets = [CensusDataset(county, **dataset_kwargs).training() for county in train_counties]
    val_datasets = [CensusDataset(county, **dataset_kwargs).validate() for county in args.val_county]
    
    # Include first decade of Census data from Figure 3 counties
    for county in args.val_county:
        train_datasets.append(torch.utils.data.Subset(
            CensusDataset(county, **dataset_kwargs).training(), 
            np.arange(0, args.val_tmax, dtype=int)
        ))

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
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
