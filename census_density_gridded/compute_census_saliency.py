import numpy as np
import os
import glob
import torch
import json
from pprint import pprint
from argparse import ArgumentParser

from census_dataset import *
from census_nn import *

import h5py
from tqdm.auto import trange, tqdm

def compute_saliency(model, dataset, device, savename, aggregate=True):
    """ Modified compared to census_gridded/compute_census_saliency
        For storage purposes, just store GS_sum so we don't need to
        aggregate later for plotting. The h5 files are too big to git 
        track anyway.

        Non-aggregated saliency: 15 GB
        Aggregated saliency (what we plot): 4.7 MB

    """
    with h5py.File(f'{savename}_aggregated_saliency.hdf5', 'a') as h5f:
        ds = h5f.require_group(dataset.county)

        if 'X' in ds:
            del ds['X']
            del ds['Y']
        if 'G_S' in ds:
            del ds['G_S']
        
        ds.create_dataset('X', data=dataset.x) # REMEMBER TO SUBTRACT MEAN TO ALIGN
        ds.create_dataset('Y', data=dataset.y) # ACROSS MULTIPLE COUNTIES
        
        G_S_sum = np.zeros([2, 2, *dataset.x.shape])

        batch = dataset[0] # Predictions at each year
        wb = batch['wb'].to(device)
        wb[wb.isnan()] = 0.

        nnz = np.asarray(np.nonzero(batch['mask'])).T
        
        for tt in tqdm(range(wb.shape[0])):
            # Select inputs at this time
            inputs = wb[tt:tt+1].clone()
            inputs.requires_grad = True
            dt_wb = model(inputs) # [1, 2, H, W]

            # Select 100 points to compute saliency at
            np.random.shuffle(nnz)
            pts = nnz[:100]

            # Compute prediction saliency
            G_S = []
            for pt in pts:
                loc = torch.zeros(batch['mask'].shape, dtype=wb.dtype, device=wb.device)
                loc[pt[0], pt[1]] = 1.

                grad = []
                for cc in range(dt_wb.shape[1]):
                    grad.append(
                        torch.autograd.grad(dt_wb[0,cc], inputs, grad_outputs=loc, retain_graph=True)[0]
                    )
                grad = torch.cat(grad, dim=0) #[2, 2, H, W]
                G_S.append(grad.detach().cpu().numpy().squeeze())

            # Shift gradients to have common origin
            center = np.asarray([G_S[0].shape[-2]/2, G_S[0].shape[-1]/2]).astype(int)
            shifts = np.asarray(center-pts)
            G_S_shifted = np.asarray([np.roll(g, shift, axis=(-2,-1)) for shift, g in zip(shifts, G_S)])

            # Add to aggregated G_S_sum
            G_S_sum += np.sum(G_S_shifted, axis=0)
        
        G_S_sum = ds.create_dataset('G_S_sum', data=G_S_sum)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_id', type=str, default='gridded')
    parser.add_argument('--use_max_scaling', action='store_true')
    parser.add_argument('--use_fill_frac', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CensusForecasting().to(device)

    savename = f'models/{model.__class__.__name__}_{args.model_id}'
    info = torch.load(f'{savename}.ckpt')
    model.load_state_dict(info['state_dict'])

    with open(f'{savename}_args.txt', 'r') as f:
        params = json.load(f)
    pprint(params)

    dataset_kwargs = dict(
        use_fill_frac=args.use_fill_frac,
        use_max_scaling=args.use_max_scaling,
    )
    
    with torch.autograd.set_detect_anomaly(True):
        for county in params['val_county']:
            dataset = CensusDataset(county, **dataset_kwargs)
            dataset.validate()
            compute_saliency(model, dataset, device, savename)

