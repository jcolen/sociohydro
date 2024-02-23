import torch
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
import numpy as np
import h5py

import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad

from dolfin_problems import *
import pygmsh
import cv2
import os

dlf.set_log_level(40)
    
def smooth_with_fill(arr, sigma=3):
    msk = np.isnan(arr)
    mask = np.where(~msk)

    interp = NearestNDInterpolator(np.transpose(mask), arr[~msk])
    arr = interp(*np.indices(arr.shape))
    arr = gaussian_filter(arr, sigma=sigma)
    arr[msk] = np.nan
    return arr

class CensusDataset(torch.utils.data.Dataset):
    def __init__(self,
                 county='cook_IL',
                 spatial_scale=1e3,
                 housing_method='constant',
                 get_dolfin=SociohydrodynamicsProblem,
                ):
        self.county = county
        self.spatial_scale = spatial_scale
        self.train = True
        self.housing_method = housing_method
        self.get_dolfin = get_dolfin
        self.init_data()
        
    def training(self):
        self.train = True
    
    def validate(self):
        self.train = False
    
    def init_data(self):
        with h5py.File(f'/home/jcolen/sociohydro/data/{self.county}.hdf5', 'r') as d:
            x_grid = d["x_grid"][:] / self.spatial_scale
            y_grid = d["y_grid"][:] / self.spatial_scale
            w_grid = d["w_grid_array_masked"][:].transpose(2, 0, 1)
            b_grid = d["b_grid_array_masked"][:].transpose(2, 0, 1)
            for i in range(5):
                w_grid[i] = smooth_with_fill(w_grid[i])
                b_grid[i] = smooth_with_fill(b_grid[i])

        self.x = x_grid
        self.y = y_grid
        self.t = np.array([1980, 1990, 2000, 2010, 2020], dtype=float)
        
        #Convert population to occupation fraction
        wb = np.stack([w_grid, b_grid], axis=1)

        if self.housing_method == 'constant':
            print('Building dataset with constant housing in time', flush=True)
            self.housing = np.sum(wb, axis=1).max(axis=0)
        else: #Housing can vary in time
            print('Building dataset with time-varying housing', flush=True)
            self.housing = np.sum(wb, axis=1, keepdims=True)
            
        wb /= self.housing
        
        self.mask = np.all(~np.isnan(wb), axis=(0, 1))
        self.wb = interp1d(self.t, wb, axis=0, fill_value='extrapolate')
        
        self.mesh = d_ad.Mesh(f'/home/jcolen/sociohydro/data/{self.county}_mesh.xml')
        self.mesh_area = d_ad.assemble(1*ufl.dx(self.mesh))
    
    def get_time(self, t, dt=1):
        wb0 = self.wb(t)
        wb1 = self.wb(t+dt)
        
        sample = {
            't': t,
            'x': self.x,
            'y': self.y,
            'mask': self.mask,
            'dt': dt,
            'wb0': self.wb(t),
            'wb1': self.wb(t+dt),
        }
        sample['problem'] = self.get_dolfin(self, sample)
        return sample
    
    def __len__(self):
        return int(np.ptp(self.t))
    
    def __getitem__(self, idx):
        t0 = self.t[0] + idx
        dt = 1
        if self.train:
            t0 += np.random.random()
            dt *= np.random.random()
            
        sample = self.get_time(t0, dt)
        sample['wb0'] = torch.FloatTensor(sample['wb0'])
                
        return sample

class YearlyDataset(CensusDataset):
    '''
    Yearly dataset pulls data from a different source
    It also includes a hispanic population
    '''
    def __init__(self,
                 county='Virginia_Fairfax',
                 spatial_scale=1e3,
                 sigma=4,
                 housing_method='constant',
                 get_dolfin=ThreeDemographicsDynamics):
        self.sigma = sigma
        super().__init__(county=county, 
                         spatial_scale=spatial_scale,
                         housing_method=housing_method,
                         get_dolfin=get_dolfin)
    
    def init_data(self):
        with h5py.File(f'/home/jcolen/sociohydro/yearly/processed/{self.county}.hdf5', 'r') as h5f:
            w_grid = []
            b_grid = []
            h_grid = []
            for key in h5f.keys():
                d = h5f[key]
                x_grid = d["x_grid"][:] / self.spatial_scale
                y_grid = d["y_grid"][:] / self.spatial_scale
                mask = d["mask"][:].astype(bool)
                w_grid.append(smooth_with_fill(d["w_grid"], sigma=self.sigma))
                b_grid.append(smooth_with_fill(d["b_grid"], sigma=self.sigma))
                h_grid.append(smooth_with_fill(d["h_grid"], sigma=self.sigma))

        self.x = x_grid
        self.y = y_grid
        self.t = np.arange(2010, 2022, dtype=float)
        
        #Convert population to occupation fraction
        wb = np.stack([w_grid, b_grid, h_grid], axis=1)
        print(wb.shape, flush=True)

        if self.housing_method == 'constant':
            print('Building dataset with constant housing in time', flush=True)
            self.housing = np.sum(wb, axis=1).max(axis=0)
        else: #Housing can vary in time
            print('Building dataset with time-varying housing', flush=True)
            self.housing = np.sum(wb, axis=1, keepdims=True)
            
        wb /= self.housing
        
        #self.mask = np.all(~np.isnan(wb), axis=(0, 1))
        self.mask = mask
        self.wb = interp1d(self.t, wb, axis=0, fill_value='extrapolate')
        
        mesh_path = f'/home/jcolen/sociohydro/yearly/processed/{self.county}_mesh.xml'
        
        self.mesh = d_ad.Mesh(mesh_path)
        self.mesh_area = d_ad.assemble(1*ufl.dx(self.mesh))