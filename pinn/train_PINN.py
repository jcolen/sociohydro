import numpy as np
import torch
import h5py
import os

import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt

from pinn import *
from stationary_pinn import *


from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator

def smooth_with_fill(arr, sigma=2):
    msk = np.isnan(arr)
    mask = np.where(~msk)

    interp = NearestNDInterpolator(np.transpose(mask), arr[mask])
    arr = interp(*np.indices(arr.shape))
    arr = gaussian_filter(arr, sigma=sigma)
    arr[msk] = np.nan
    return arr

from argparse import ArgumentParser
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--stationary', action='store_true')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--model', type=str, default='DiagonalOnly')
    parser.add_argument('--num_points', type=int, default=5000)
    parser.add_argument('--sigma', type=float, default=2)
    parser.add_argument('--county', type=str, default='cook_IL')
    parser.add_argument('--loaddir', type=str, default='.')
    
    parser.add_argument('--n_adam', type=int, default=500000)
    parser.add_argument('--n_lbfgs', type=int, default=0)
    
    args = parser.parse_args()
    '''
    Load data from HDF5 files 
    '''
    with h5py.File(f"../data/{args.county}.hdf5", "r") as d:
        x_grid = d["x_grid"][:] / 1e3 #Units of km, in 0823 it was 1e5
        y_grid = d["y_grid"][:] / 1e3 #Units of km, in 0823 it was 1e5
        w_grid = d["w_grid_array_masked"][:].transpose(2, 0, 1)
        b_grid = d["b_grid_array_masked"][:].transpose(2, 0, 1)

        for ii in range(w_grid.shape[0]):
            w_grid[ii] = smooth_with_fill(w_grid[ii], sigma=args.sigma)
            b_grid[ii] = smooth_with_fill(b_grid[ii], sigma=args.sigma)

        #Convert to occupation fraction and assume 
        max_grid = (w_grid + b_grid).max(axis=0) * 1.1
        w_grid /= max_grid
        b_grid /= max_grid
        
    '''
    Select points for PINN training
    '''
    if args.stationary:
        w_grid = w_grid.mean(axis=0)
        b_grid = b_grid.mean(axis=0)
    else:
        T = np.array([1980, 1990, 2000, 2010, 2020])
        
        x_grid = np.tile(x_grid[None], (len(T), 1, 1))
        y_grid = np.tile(y_grid[None], (len(T), 1, 1))
        t = np.tile(T[:, None, None], (1, *x_grid.shape[1:]))

    keep = np.logical_and(~np.isnan(w_grid), ~np.isnan(b_grid))
    w = w_grid[keep]
    b = b_grid[keep]
    y = y_grid[keep]
    x = x_grid[keep]

    print(x.shape, y.shape, w.shape, b.shape)

    #Select points for reconstruction loss
    idx = np.random.choice(np.prod(x.shape), min(args.num_points, np.prod(x.shape) // 2), replace=False)

    x_u = x[idx]
    y_u = y[idx]
    w_u = w[idx]
    b_u = b[idx]
    
    print(x_u.shape, y_u.shape, w_u.shape, b_u.shape)

    margs = [torch.from_numpy(x_u).float(),
             torch.from_numpy(y_u).float(),
             torch.from_numpy(w_u).float(),
             torch.from_numpy(b_u).float()]   
    if not args.stationary:
        t = t[keep]
        t_u = t[idx]
        margs.insert(2, torch.from_numpy(t_u).float())
        
    '''
    Select model
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modeltype = args.model + ('StationaryPINN' if args.stationary else 'PINN')
    print(f'Training a {modeltype}', flush=True)
    if os.path.exists(os.path.join(args.loaddir, modeltype)):
        print('Loading model from file')
        info = torch.load(os.path.join(args.loaddir, modeltype))
        model = eval(modeltype)(*margs, **info['hparams']).to(device)
        model.model.load_state_dict(info['state_dict'])
        model.iter = info['iteration']
        model.print()
    else:
        model = eval(modeltype)(*margs).to(device)
    model.train(n_lbfgs=args.n_lbfgs, n_adam=args.n_adam)
            
    
    

