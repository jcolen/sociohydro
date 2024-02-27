import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from dolfin_problems import *

import dolfin as dlf
import dolfin_adjoint as d_ad
import h5py
from tqdm.auto import trange, tqdm

'''
Building blocks
'''
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dropout_rate=0.):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding='same', padding_mode='replicate', groups=in_channels)
        self.conv2 = nn.Conv1d(out_channels, 4*out_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(4*out_channels, out_channels, kernel_size=1)
        
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sin(x)
        x = self.conv3(x)
        
        x = self.dropout(x)
        return x
    
'''
Neural network
'''
class SimulationPBNN(nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=2,
                 kernel_size=7,
                 N_hidden=8,
                 hidden_size=64,
                 dropout_rate=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.N_hidden = N_hidden
        self.hidden_size = hidden_size
    
        self.read_in = ConvNextBlock(input_dim, hidden_size, kernel_size, dropout_rate)

        self.downsample = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=4, stride=4),
            nn.GELU()
        )
        self.read_out = nn.Conv1d(2*hidden_size, output_dim, kernel_size=1)
        
        self.cnn1 = nn.ModuleList()
        self.cnn2 = nn.ModuleList()
        for i in range(N_hidden):
            self.cnn1.append(ConvNextBlock(hidden_size, hidden_size, kernel_size))
            self.cnn2.append(ConvNextBlock(hidden_size, hidden_size, kernel_size))
            
        self.device = torch.device('cpu')        
    
    def training_step(self, sample):
        problem = sample['problem']
        params = self.forward(sample['ab0'][None], problem.FctSpace)
        params['Si'] = params['Si'].flatten()
        problem.set_params(**params)
        loss = problem.residual()
        grad = problem.grad(sample['ab0'].device)

        params['Si'].backward(gradient=grad['Si'] / sample['dt'])
        
        return params, loss
    
    def validation_step(self, sample):
        problem = sample['problem']
        params = self.forward(sample['ab0'][None], problem.FctSpace)
        
        problem.set_params(**params)
        loss = problem.residual()
                
        return params, loss
    
    def forward(self, ab0, FctSpace):
        assert FctSpace is not None
        
        #Neural network part
        ab0[ab0.isnan()] = 0
        
        x = self.read_in(ab0)
        for cell in self.cnn1:
            x = x + cell(x)
        latent = self.downsample(x)
        for cell in self.cnn2:
            latent = latent + cell(latent)
        latent = F.interpolate(latent, x.shape[-1:])
        x = torch.cat([x, latent], dim=1)
        Si = self.read_out(x).squeeze()
        
        Si_mesh  = torch.zeros([Si.shape[0],  FctSpace.dim()], dtype=Si.dtype,  device=Si.device)
        for i in range(Si.shape[0]):
            Si_mesh[i] = scalar_to_mesh(Si[i], FctSpace, vals_only=True)
        
        return {
            'Si': Si_mesh, 
        }
    
    def simulate(self, sample, mesh, tmax, dt=0.1):
        problem = sample['problem']
        problem.dt.assign(d_ad.Constant(dt))
        FctSpace = problem.FctSpace
        
        Dt = d_ad.Constant(dt)
        device = sample['ab0'].device
        
        #Interpolate phiA, phiB to grid
        ab = []
        for i in range(self.input_dim):
            problem.ab0[i].assign(scalar_to_mesh(sample['ab0'][i], FctSpace))
            ab.append(mesh_to_scalar(problem.ab0[i], mesh))
        
        ab = torch.FloatTensor(np.stack(ab))
        ab = ab.to(device)
        
        steps = int(np.round(tmax / dt))
        for tt in trange(steps):
            #Run PBNN to get Dij, constants and assign to variational parameters
            params = self.forward(ab[None], FctSpace)
            params['Si'] *= 1
            problem.set_params(**params)
            
            #Solve variational problem and update w0, b0
            ab = problem.forward()
            w, b = ab.split(True)
            problem.ab0[0].assign(w)
            problem.ab0[1].assign(b)
            
            #Interpolate w0, b0 to grid
            ab = torch.FloatTensor(np.stack([
                mesh_to_scalar_img(problem.ab0[i], mesh, x_img, y_img, mask) for i in range(self.input_dim)]))
            ab = ab.to(device)
                                    
        return problem.ab0[0], problem.ab0[1], ab