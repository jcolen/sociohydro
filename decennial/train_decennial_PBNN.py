import numpy as np
import os
import torch
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser

from census_dataset import *
from census_pbnn import *

def train(model, dataset, n_epochs, batch_size, device, savename):
    '''
    Train a model
    '''
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    train_loss = []
    val_loss = []
    step = 0
    
    idxs = np.arange(len(dataset), dtype=int)

    
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            for ds in dataset.datasets:
                ds.training()
            np.random.shuffle(idxs)
            d_ad.set_working_tape(d_ad.Tape())
            model.train()

            # Schedule increase in loss weight of segregation
            if epoch == 50:
                print('Setting Dolfin alpha to 0.1')
                for ds in dataset.datasets:
                    ds.dolfin_kwargs = {'alpha': 0.1}

            with tqdm(total=len(dataset), leave=False) as ebar:
                for i in range(len(dataset)):
                    batch = dataset[idxs[i]]
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

            for ds in dataset.datasets:
                ds.validate()
            val_loss.append(0)
            model.eval()

            with tqdm(total=len(dataset), leave=False) as ebar:
                with torch.no_grad():
                    for i in range(len(dataset)):
                        d_ad.set_working_tape(d_ad.Tape())
                        batch = dataset[i]
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
    parser.add_argument('--county', nargs='+', default=['Georgia_Fulton', 'Illinois_Cook', 'Texas_Harris', 'California_Los Angeles'])
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_id', type=str, default='0')
    args = parser.parse_args()

    datasets = []
    for county in args.county:
        datasets.append(CensusDataset(county))
    dataset = torch.utils.data.ConcatDataset(datasets)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CensusPBNN().to(device)

    savename = f'models/{model.__class__.__name__}_{args.model_id}'
    print(f'Saving information to {savename}')

    with open(f'{savename}_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    
    with torch.autograd.set_detect_anomaly(True):
        train(model, dataset, args.n_epochs, args.batch_size, device, savename)
        
        for ds in dataset.datasets:
            compute_saliency(model, ds, device, savename)
