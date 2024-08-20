import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import os
import numpy as np
from glob import glob
from fipy.tools.dump import read
from scipy.interpolate import griddata

from fvm_utils import calc_gradients

class ComputeFeatures:
    """ Compute and return the features composed of sociohydrodynamic gradients """
    def __init__(self, features='sociohydro'):
        """ features = ['sociohydro', 'all'] specifies which grouping to use """
        self.features = features

    def __call__(self, sample):
        all_features, sociohydro_features = calc_gradients(sample['W0_mesh'], sample['B0_mesh'])
        if self.features == 'all':
            sample['features_mesh'] = [all_features, all_features]
        elif self.features == 'sociohydro':
            sample['features_mesh'] = sociohydro_features
        
        return sample

class InterpolateToGrid:
    """ Interpolate ϕW, ϕB to a regular grid using scipy.interpolate.griddata """
    def __init__(self, method='nearest', d=2.):
        self.method = method
        self.d = d
    
    def interpolate_fipy(self, data, X, Y):
        # Get boundaries of each triangular cell
        vertices = data.mesh.vertexCoords[:, data.mesh._orderedCellVertexIDs]
        # Get centroids
        centroids = np.mean(vertices, axis=1).T
        # Interpolate
        return griddata(centroids, data, (X, Y))

    def __call__(self, sample):
        # Establish common target coordinate system
        mesh = sample['W0_mesh'].mesh
        xmin, ymin = mesh.extents['min']
        xmax, ymax = mesh.extents['max']
        xx = np.arange(xmin, xmax+self.d, self.d)
        yy = np.arange(ymin, ymax+self.d, self.d)
        X, Y = np.meshgrid(xx, yy)

        sample['X'] = X
        sample['Y'] = Y

        # Initial point
        sample['W0'] = self.interpolate_fipy(sample['W0_mesh'], X, Y)
        sample['B0'] = self.interpolate_fipy(sample['B0_mesh'], X, Y)

        # End point
        sample['W1'] = self.interpolate_fipy(sample['W1_mesh'], X, Y)
        sample['B1'] = self.interpolate_fipy(sample['B1_mesh'], X, Y)

        # Interpolate and stack features
        features = [
            [self.interpolate_fipy(grad, X, Y) for grad in sample['features_mesh'][0]],
            [self.interpolate_fipy(grad, X, Y) for grad in sample['features_mesh'][1]]
        ]
        sample['features'] = np.asarray(features)

        return sample

class MeshToNumpyArray:
    """ Discard the mesh information and replace with a numpy array """
    def __call__(self, sample):
        # Establish coordinate system
        mesh = sample['W0_mesh'].mesh
        vertices = mesh.vertexCoords[:, mesh._orderedCellVertexIDs]
        centroids = np.mean(vertices, axis=1).T
        sample['X'] = centroids[:, 0]
        sample['Y'] = centroids[:, 1]

        # Initial point
        sample['W0'] = sample['W0_mesh'].value
        sample['B0'] = sample['B0_mesh'].value

        # End point
        sample['W1'] = sample['W1_mesh'].value
        sample['B1'] = sample['B1_mesh'].value

        # Stack features
        features = [
            [grad.value for grad in sample['features_mesh'][0]],
            [grad.value for grad in sample['features_mesh'][1]]
        ]
        sample['features'] = np.asarray(features)

        return sample


class ComputeDerivative:
    """ Compute the discrete time derivative using forward differences """
    def __call__(self, sample):
        sample['dt_W'] = (sample['W1'] - sample['W0']) / (sample['t1'] - sample['t0'])
        sample['dt_B'] = (sample['B1'] - sample['B0']) / (sample['t1'] - sample['t0'])
        return sample

class StackInputsTargets:
    """ Group the inputs and targets (W0, B0, dtW, dtB) into terms for NN processing """
    def __call__(self, sample):
        sample['inputs'] = np.stack([sample['W0'], sample['B0']])
        sample['targets'] = np.stack([sample['dt_W'], sample['dt_B']])
        return sample

class ToTensor:
    """ Compute relevant numpy objects into torch tensors """
    def __init__(self, convert_keys=['inputs', 'targets', 'features']):
        self.convert_keys = convert_keys

    def __call__(self, sample):
        for key in self.convert_keys:
            sample[key] = torch.FloatTensor(sample[key])
        
        return sample


class FipyDataset(Dataset):
    """ Dataset that loads and transforms data from fipy """
    def __init__(self, path="./data/Georgia_Fulton_small/fipy_output", grid=False):
        """ Initialize dataset. Kwargs "grid" indicates whether grid interpolation is performed """
        self.files = sorted(glob(os.path.join(path, "*.fipy")))
        self.transform = Compose([
            ComputeFeatures(),
            InterpolateToGrid() if grid else MeshToNumpyArray(),
            ComputeDerivative(),
            StackInputsTargets(),
            ToTensor(),
        ])
    
    def __len__(self):
        return len(self.files) - 1

    def __getitem__(self, idx):
        W0, B0, t0 = read(self.files[idx])
        W1, B1, t1 = read(self.files[idx+1])

        sample = {
            'W0_mesh': W0,
            'B0_mesh': B0,
            'W1_mesh': W1,
            'B1_mesh': B1,
            't0': t0,
            't1': t1
        }

        sample = self.transform(sample)
        return sample