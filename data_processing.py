import torch
from scipy.interpolate import interp1d
import numpy as np
import h5py
import json
from glob import glob

import dolfin as dlf
import dolfin_adjoint as d_ad
dlf.set_log_level(40)

from dolfin_problems import SimulationProblem


class SimulationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 get_dolfin=SimulationProblem):
        self.train = True
        self.path = path
        self.get_dolfin = get_dolfin
        self.init_data()
        
    def training(self):
        self.train = True
    
    def validate(self):
        self.train = False
    
    def init_data(self):
        data_file = glob(f'{self.path}/*hdf5')
        param_file = glob(f'{self.path}/*json')

        with open(param_file[0]) as pfile:
            params = json.load(pfile)
        
        with h5py.File(data_file[0], 'r') as h5f:
            phiAB = np.zeros((len(h5f.keys()), 2, params['grid_size']))
            t = np.zeros(len(h5f.keys()))

            for ii, key in enumerate(h5f.keys()):
                phiAB[ii] = h5f[key]['state'][()]
                t[ii] = h5f[key]['sweep'][()]
            
        self.t = t * params['dt']
        self.dt = np.ptp(self.t) / len(self.t)
        self.x = np.linspace(0, 1, params['grid_size'])
        phiAB = phiAB / params['capacity']

        self.phiAB = interp1d(self.t, phiAB, axis=0, fill_value='extrapolate')
        self.mesh = d_ad.UnitIntervalMesh(self.x.shape[0]-1)
    
    def get_time(self, t, dt=1):
        sample = {
            't': t,
            'dt': dt,
            'x': self.x,
            'ab0': self.phiAB(t),
            'ab1': self.phiAB(t+dt),
        }
        sample['problem'] = self.get_dolfin(self, sample)
        return sample
    
    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, idx):
        t0 = self.t[idx]
        dt = self.dt * 1.0
        if self.train:
            t0 += np.random.random() * self.dt
            dt *= np.random.random()
            
        sample = self.get_time(t0, dt)
        sample['ab0'] = torch.FloatTensor(sample['ab0'])
                
        return sample