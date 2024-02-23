import numpy as np
import os
import torch
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser

from data_processing import *
from pbnn import *

def train(model, dataset, n_epochs, batch_size, device, savename='SourcedOnlyPBNN'):
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
    parser.add_argument('--county', nargs='+', default=['cook_IL', 'fulton_GA', 'harris_TX', 'la_CA'])
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--modeltype', type=str, default='SourcedOnlyPBNN')
    parser.add_argument('--savename', type=str, default='./validation/decennial/SourcedOnlyPBNN')
    parser.add_argument('--housing_method', type=str, default='constant')
    args = parser.parse_args()

    with open(f'{args.savename}_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    datasets = []
    for county in args.county:
        datasets.append(CensusDataset(county, housing_method=args.housing_method))
    dataset = torch.utils.data.ConcatDataset(datasets)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(args.modeltype)(phi_dim=2).to(device)
    
    with torch.autograd.set_detect_anomaly(True):
        #train(model, dataset, args.n_epochs, args.batch_size, device, args.savename)
        
        for ds in dataset.datasets:
            compute_saliency(model, ds, device, args.savename)
