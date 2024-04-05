import torch
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
import numpy as np
import h5py

import os

class CensusDataset(torch.utils.data.Dataset):
    def __init__(self,
                 county='cook_IL',
                 data_dir='/home/jcolen/data/sociohydro/decennial/revision/',
                 spatial_scale=1e3,
                 max_seq_len=4,
                 sigma=3,
                 dolfin_kwargs={},
                ):
        self.county = county
        self.data_dir = data_dir
        self.spatial_scale = spatial_scale
        self.max_seq_len = max_seq_len
        self.sigma = sigma
        self.dolfin_kwargs = {}
        self.init_data()

        self.train = True
        
    def training(self):
        self.train = True
        return self
    
    def validate(self):
        self.train = False
        return self

    def smooth_with_fill(self, arr):
        # before we can smooth, we need to interpolate (NN) to outside the county boundary
        msk = np.isnan(arr)
        mask = np.where(~msk)

        interp = NearestNDInterpolator(np.transpose(mask), arr[~msk])
        arr = interp(*np.indices(arr.shape))

        # Apply smoothing filter
        arr = gaussian_filter(arr, sigma=self.sigma)
        arr[msk] = np.nan # Reapply masked or NaN points

        return arr
    
    def init_data(self):
        w_grid = []
        b_grid = []
        t = []
        with h5py.File(f'{self.data_dir}/gridded/{self.county}.hdf5', 'r') as h5f:
            for year in h5f.keys():
                t.append(float(year))
                g = h5f[year]
                x_grid = g['x_grid'][:] / self.spatial_scale
                y_grid = g['y_grid'][:] / self.spatial_scale
                mask = ~g['county_mask'][:].astype(bool)

                w_grid.append(gaussian_filter(g['white_grid'][:], sigma=self.sigma))
                b_grid.append(gaussian_filter(g['black_grid'][:], sigma=self.sigma))

        self.x = x_grid
        self.y = y_grid
        self.mask = mask
        self.t = np.array(t)
        
        # Convert population to occupation fraction
        wb = np.stack([w_grid, b_grid], axis=1)
        self.mask = np.logical_and(self.mask, np.all(~np.isnan(wb), axis=(0,1)))
        self.housing = np.sum(wb, axis=1).max(axis=0) # Assume housing is the maximum occupation level over time
        wb /= self.housing
        self.wb = interp1d(self.t, wb, axis=0, fill_value='extrapolate')
    
    def __len__(self):
        return int(np.ptp(self.t))
    
    def __getitem__(self, idx):
        t0 = self.t[0] + idx
        dt = 1
        if self.train:
            t0 += np.random.random()
            dt = min(0.2, dt*np.random.random())
            t1 = t0 + dt
        elif self.validate: # Return the whole sequence
            t1 = self.t[-1]

        t = np.arange(t0, t1+dt, dt)

        sample = {
            't': t,
            'dt': dt,
            'x': self.x,
            'y': self.y,
            'mask': self.mask,
            'wb': torch.FloatTensor(self.wb(t)),
        }

        return sample