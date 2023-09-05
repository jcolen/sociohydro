import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import h5py
from tqdm.auto import tqdm
import time

from torch.utils.data import IterableDataset
from scipy.interpolate import interp1d
from sociohydro import *

class SociohydrodynamicsDataset(IterableDataset):
    def __init__(self, 
                 filename='data/cook_IL.hdf5', 
                 window_size=(32,24),
                 batch_size=16,
                 test_bbox=[20, 43, 60, 23]):
        self.filename = filename
        self.window_size = window_size
        self.batch_size = batch_size
        
        with h5py.File(filename, 'r') as d:
            x_grid = d["x_grid"][:]
            y_grid = d["y_grid"][:]
            w_grid = d["w_grid_array_masked"][:]
            b_grid = d["b_grid_array_masked"][:]

        self.x = x_grid[0]
        self.y = y_grid[:, 0]
        self.t = np.array([1980, 1990, 2000, 2010, 2020])
        
        self.w_func = interp1d(self.t, w_grid, axis=2)
        self.b_func = interp1d(self.t, b_grid, axis=2)
        
        y0, x0, h, w = test_bbox
        w_test = w_grid[y0:y0+h, x0:x0+w].transpose(2, 0, 1)
        b_test = b_grid[y0:y0+h, x0:x0+w].transpose(2, 0, 1)
        self.test_batch = {
            't': torch.from_numpy(self.t).float(),
            'w': torch.from_numpy(w_test).float(),
            'b': torch.from_numpy(b_test).float(),
        }
        
        
    def __next__(self):
        '''
        Yield a sample interpolated from the data
        '''
        t0 = np.random.random()
        t1 =  t0 + (1 - t0) * np.random.random()
        
        t = np.array([t0, t1])
        t = t * np.ptp(self.t) + self.t[0]

        w = self.w_func(t)
        b = self.b_func(t)
        
        w_windows = sliding_window_view(w, self.window_size, axis=(0, 1))
        b_windows = sliding_window_view(b, self.window_size, axis=(0, 1))
        
        valid = np.logical_and(~np.isnan(w_windows), ~np.isnan(b_windows))
        valid = np.all(valid, axis=(-1, -2, -3))
        w_windows = w_windows[valid]
        b_windows = b_windows[valid]

        idx = np.random.choice(w_windows.shape[0], self.batch_size, replace=False)
        w = w_windows[idx]
        b = b_windows[idx]
                
        return {
            't': torch.from_numpy(t).float(),
            'w': torch.from_numpy(w).float(),
            'b': torch.from_numpy(b).float(),
        }
        
    def __iter__(self):
        return self
    
class Conv_ODEFunc(nn.Module):
    def __init__(self, n_input=2, n_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvNextBlock(2, n_hidden),
            ConvNextBlock(n_hidden, n_hidden),
            nn.Conv2d(n_hidden, 2, kernel_size=1)
        )
    
    def forward(self, t, y):
        return self.net(y)
    
class FCN_ODEFunc(nn.Module):
    def __init__(self, n_input=1380, n_hidden=2048,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            Sin(),
            nn.Linear(n_hidden, n_hidden),
            Sin(),
            nn.Linear(n_hidden, n_input)
        )
    
    def forward(self, t, y):
        b, c, h, w = y.shape
        y = y.reshape([b, -1])
        dt = self.net(y)
        return dt.reshape([b, c, h, w])
    
from argparse import ArgumentParser

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', choices=['conv', 'fcn'], default='conv')
    parser.add_argument('--n_steps', type=int, default=10000)
    args = parser.parse_args()
    
    dataset = SociohydrodynamicsDataset(test_bbox=[20, 53, 20, 30])
    device = torch.device('cuda:0')

    if args.model == 'conv':
        model_kwargs = dict(n_input=2, n_hidden=64)
        model = Conv_ODEFunc(**model_kwargs).to(device)
    else:
        model_kwargs = dict(n_input=1200, n_hidden=2000)
        model = FCN_ODEFunc(**model_kwargs).to(device)
        
    print(f'Training a {model.__class__.__name__}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loader = iter(dataset)

    end = time.time()
    ode_kwargs = dict(method='rk4', options=dict(step_size=0.1))

    best_loss = 1e10
    for step in range(args.n_steps):
        optimizer.zero_grad()
        batch = dataset.test_batch
        t = batch['t'].to(device)
        wb = torch.stack([batch['w'], batch['b']], dim=-3).to(device)
        wb_pred = odeint(model, wb[0:1], t, **ode_kwargs).squeeze()
        loss = F.l1_loss(wb, wb_pred)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f'Iter {step:04d} | Test Loss: {loss.item():.6g}\tTime: {time.time() - end}')
            if loss.item() < best_loss:
                torch.save(dict(
                    hparams=model_kwargs,
                    state_dict=model.state_dict(),
                    loss=loss.item()),
                f'data/{model.__class__.__name__}.ckpt')
                
                best_loss = loss.item()

        end = time.time()