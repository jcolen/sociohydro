import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import os
import numpy as np
from glob import glob
from fipy.tools.dump import read
from scipy.interpolate import interp1d, griddata
from scipy.spatial import distance_matrix
from time import time

from fvm_utils import calc_gradients, gaussian_blur_mesh


class Smooth:
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def __call__(self, sample):
        if self.sigma > 0:
            x, y = sample["W0_mesh"].mesh.cellCenters
            coords = np.stack([x, y], axis=-1)
            dist_mat = distance_matrix(coords, coords)
            
            sample["W0_mesh"] = gaussian_blur_mesh(sample["W0_mesh"],
                                                   self.sigma,
                                                   dist_mat=dist_mat)
            sample["B0_mesh"] = gaussian_blur_mesh(sample["B0_mesh"],
                                                   self.sigma,
                                                   dist_mat=dist_mat)
            sample["W1_mesh"] = gaussian_blur_mesh(sample["W1_mesh"],
                                                   self.sigma,
                                                   dist_mat=dist_mat)
            sample["B1_mesh"] = gaussian_blur_mesh(sample["B1_mesh"],
                                                   self.sigma,
                                                   dist_mat=dist_mat)

        return sample


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
        sample['dt_W'] = sample["W0_mesh"].mesh._cellVolumes * (sample['W1'] - sample['W0']) / (sample['t1'] - sample['t0'])
        sample['dt_B'] = sample["B0_mesh"].mesh._cellVolumes * (sample['B1'] - sample['B0']) / (sample['t1'] - sample['t0'])
        return sample

class StackInputsTargets:
    """ Group the inputs and targets (W0, B0, dtW, dtB) into terms for NN processing """
    def __call__(self, sample):
        sample['inputs'] = np.stack([sample['W0'], sample['B0']])
        sample['targets'] = np.stack([sample['dt_W'], sample['dt_B']])
        return sample

class ToTensor:
    """ Compute relevant numpy objects into torch tensors """
    def __init__(self,
                 convert_keys=['inputs', 'targets', 'features', 'X', 'Y'], 
                 remove_extra=False):
        self.convert_keys = convert_keys
        self.remove_extra = remove_extra

    def __call__(self, sample):
        for key in self.convert_keys:
            sample[key] = torch.FloatTensor(sample[key])

        # Optionally remove extra elements to play nice with dataloader
        if self.remove_extra:
            to_remove = [key for key in sample if not key in self.convert_keys]
            for key in to_remove:
                sample.pop(key)
        
        return sample


class CensusDataset(Dataset):
    """ Dataset that loads and transforms data from fipy """
    def __init__(self, 
                 path="./data/Georgia_Fulton_small/fipy_output", 
                 grid=False, 
                 remove_extra=True,
                 preload=False,
                 region="all",
                 sigma=0.0,
                 interp_time=False):
        """ 
        Initialize dataset

        Inputs
        ------
        path : str (required)
            path to dataset
        grid : bool (optional)
            indicates whether grid interpolation is performed.
            Defaults to False
        remove_extra : bool (optional)
            remove extra elements to play nice with dataloader.
            Defaults to True
        preload : bool (optional)
            whether to load data at initialization or at runtime.
            Deaults to False
        region : str (optional)
            either "county" or "all", to load data from just county, or
            also load in data from surrounding regions as well.
            Defaults to "all"
        sigma : float (optional)
            size of Gaussian to use if smoothing data. Pass 0.0 to avoid
            smoothing.
            Defaults to 0.0
        interp_time : bool (optional)
        """
        self.files = sorted(glob(os.path.join(path, "*.fipy")))

        regions = ["all", "county"]
        if region not in regions:
            raise ValueError(f"region is {region}. Must be in {regions}")
        else:
            self.region = region
            self.region_idx = regions.index(region)

        self.sigma = sigma

        self.transform = Compose([
            Smooth(sigma=sigma),
            ComputeFeatures(),
            InterpolateToGrid() if grid else MeshToNumpyArray(),
            ComputeDerivative(),
            StackInputsTargets(),
            ToTensor(remove_extra=remove_extra),
        ])

        if preload:
            print('Pre-loading all Fipy data from files')
            
            t = time()
            self.data = []

            # preload capacity
            capacity = read(self.files[-1])[self.region_idx]
            # if capacity = 0, make sure phiW = 0 there
            capacity.value[np.where(capacity.value==0.0)] = np.inf

            for file in self.files[:-1]:
                # only get white and black demographic data
                contents = read(file)
                W = contents[0 + self.region_idx] / capacity
                B = contents[2 + self.region_idx] / capacity
                T = contents[-1]
                self.data.append([W, B, T])
            print(f'Done (t = {time()-t:.1f} s)')
        else:
            print('Fipy data files will be read in at runtime')
            self.data = None

    def __len__(self):
        return len(self.files) - 2

    def __getitem__(self, idx):
        if self.data:
            W0, B0, t0 = self.data[idx]
            W1, B1, t1 = self.data[idx+1]
        else:
            capacity = read(self.files[-1])[self.region_idx]
            # if capacity = 0, make sure phiW = 0 there
            capacity.value[np.where(capacity.value==0.0)] = np.inf

            contents = read(self.files[idx])
            W0 = contents[0 + self.region_idx] / capacity
            B0 = contents[2 + self.region_idx] / capacity
            t0 = centents[-1]

            contents = read(self.files[idx+1])
            W1 = contents[0 + self.region_idx] / capacity
            B1 = contents[2 + self.region_idx] / capacity
            t1 = centents[-1]

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