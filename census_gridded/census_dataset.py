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
                 data_dir='/home/jcolen/data/sociohydro/decennial/revision/',
                 spatial_scale=1e3,
                 sigma=3,
                 dolfin_kwargs={},
                ):
        self.county = county
        self.data_dir = data_dir
        self.spatial_scale = spatial_scale
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
        arr[msk] = np.nan # Reapply county boundary

        return arr
    
    def init_data(self):
        # First, get county geometry -- mask and mesh
        self.mesh = d_ad.Mesh(f'{self.data_dir}/meshes/{self.county}.xml')
        self.mesh_area = d_ad.assemble(1*ufl.dx(self.mesh)) # for normalizing residuals by area
        self.mask = np.load(f'{self.data_dir}/meshes/{self.county}_dilated_mask.npy')

        w_grid = []
        b_grid = []
        t = []
        with h5py.File(f'{self.data_dir}/gridded/{self.county}.hdf5', 'r') as h5f:
            for year in h5f.keys():
                t.append(float(year))
                g = h5f[year]
                x_grid = g['x_grid'][:] / self.spatial_scale
                y_grid = g['y_grid'][:] / self.spatial_scale

                w_grid.append(gaussian_filter(g['white_grid'][:], sigma=self.sigma))
                b_grid.append(gaussian_filter(g['black_grid'][:], sigma=self.sigma))

        self.x = x_grid
        self.y = y_grid
        self.t = np.array(t)
        
        # Convert population to occupation fraction
        wb = np.stack([w_grid, b_grid], axis=1)
        wb[..., ~self.mask] = np.nan
        self.mask = np.all(~np.isnan(wb), axis=(0,1))
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
        sample['problem'] = TwoDemographicsDynamics(self, sample, **self.dolfin_kwargs)
        sample['wb0'] = torch.FloatTensor(sample['wb0'])

        return sample