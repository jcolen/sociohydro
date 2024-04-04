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

def compute_saliency(model, dataset, device, savename):
    with h5py.File(f'{savename}_saliency.h5', 'a') as h5f:
        ds = h5f.require_group(dataset.county)

        if 'X' in ds:
            del ds['X']
            del ds['Y']
        if 'G_S' in ds:
            del ds['G_S']
        
        ds.create_dataset('X', data=dataset.x) # REMEMBER TO SUBTRACT MEAN TO ALIGN
        ds.create_dataset('Y', data=dataset.y)

        gs = ds.require_group('G_S')

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

            # Store in h5py
            gs.create_dataset(f'{int(batch["t"][tt])}', data=G_S_shifted)
        
        G_S_sum = np.sum(np.asarray([gs[t] for t in gs.keys()]), axis=(0,1))
        ds.create_dataset('G_S_sum', data=G_S_sum)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_id', type=str, default='gridded')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CensusForecasting().to(device)

    savename = f'models/{model.__class__.__name__}_{args.model_id}'
    info = torch.load(f'{savename}.ckpt')
    model.load_state_dict(info['state_dict'])

    with open(f'{savename}_args.txt', 'r') as f:
        params = json.load(f)
    pprint(params)
    
    with torch.autograd.set_detect_anomaly(True):
        for county in params['val_county']:
            dataset = CensusDataset(county)
            dataset.validate()
            compute_saliency(model, dataset, device, savename)
