import torch
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
import numpy as np
import h5py

import ufl
import dolfin_adjoint as d_ad
import pyadjoint as pyad

from dolfin_problems import TwoDemographicsDynamics
import os

class CensusDataset(torch.utils.data.Dataset):
    def __init__(self,
                 county='cook_IL',
                 data_dir='/home/jcolen/data/sociohydro/decennial/processed',
                 spatial_scale=1e3,
                 sigma=3,
                 get_dolfin=TwoDemographicsDynamics,
                ):
        self.county = county
        self.data_dir = data_dir
        self.spatial_scale = spatial_scale
        self.sigma = sigma
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
        arr[msk] = np.nan # Reapply county boundary

        return arr
    
    def init_data(self):
        with h5py.File(f'{self.data_dir}/{self.county}.hdf5', 'r') as d:
            x_grid = d["x_grid"][:] / self.spatial_scale
            y_grid = d["y_grid"][:] / self.spatial_scale
            w_grid = d["w_grid_array_masked"][:].transpose(2, 0, 1)
            b_grid = d["b_grid_array_masked"][:].transpose(2, 0, 1)

            for i in range(5): # 5 recorded time points
                w_grid[i] = self.smooth_with_fill(w_grid[i])
                b_grid[i] = self.smooth_with_fill(b_grid[i])

        self.x = x_grid
        self.y = y_grid
        self.t = np.array([1980, 1990, 2000, 2010, 2020], dtype=float)
        
        # Convert population to occupation fraction
        wb = np.stack([w_grid, b_grid], axis=1)
        self.housing = np.sum(wb, axis=1).max(axis=0) # Assume housing is the maximum occupation level over time
        wb /= self.housing
        self.wb = interp1d(self.t, wb, axis=0, fill_value='extrapolate')

        # Get county geometry -- mask and mesh
        self.mask = np.all(~np.isnan(wb), axis=(0,1))
        self.mesh = d_ad.Mesh(f'{self.data_dir}/{self.county}_mesh.xml')
        self.mesh_area = d_ad.assemble(1*ufl.dx(self.mesh))
    
    def __len__(self):
        return int(np.ptp(self.t))
    
    def __getitem__(self, idx):
        t0 = self.t[0] + idx
        dt = 1
        if self.train:
            t0 += np.random.random()
            dt *= np.random.random()

        sample = {
            't': t0,
            'dt': dt,
            'x': self.x,
            'y': self.y,
            'mask': self.mask,
            'wb0': self.wb(t0),
            'wb1': self.wb(t0+dt),
        }
        sample['problem'] = TwoDemographicsDynamics(self, sample)
        sample['wb0'] = torch.FloatTensor(sample['wb0'])

        return sample