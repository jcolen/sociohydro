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

def compute_saliency(model, dataset, device, savename='SourcedOnlyPBNN'):
    def forward(model, wb0, FctSpace, xy):
        #Neural network part
        x = model.read_in(wb0)
        for cell in model.cnn1:
            x = x + cell(x)
        latent = model.downsample(x)
        for cell in model.cnn2:
            latent = latent + cell(latent)
        latent = F.interpolate(latent, x.shape[-2:])
        x = torch.cat([x, latent], dim=1)
        S = model.read_out(x).squeeze()
        return {
            'Si': S,
        }
    def aggregate_sample(model, sample):
        sample['wb0'] = sample['wb0'].to(device)[None]
        sample['wb0'][sample['wb0'].isnan()] = 0
        sample['wb0'].requires_grad = True
        params = forward(
            model,
            sample['wb0'],
            sample['problem'].FctSpace,
            (sample['x'], sample['y']))

        nnz = np.asarray(np.nonzero(sample['mask'])).T
        np.random.shuffle(nnz)
        pts = nnz[:100]

        G_S = []
        for pt in pts:
            loc = torch.zeros_like(params['Si'][0])
            loc[pt[0], pt[1]] = 1.

            grad = []
            for j in range(params['Si'].shape[0]):
                grad.append(torch.autograd.grad(params['Si'][j], sample['wb0'], grad_outputs=loc, retain_graph=True)[0])
            grad = torch.stack(grad) #[3, 3, Y, X]
            G_S.append(grad.detach().cpu().numpy().squeeze())

        center = np.asarray([G_S[0].shape[-2]/2, G_S[0].shape[-1]/2]).astype(int)
        shifts = np.asarray(center-pts)

        G_S_shifted = np.asarray([np.roll(g, shift, axis=(-2,-1)) for shift, g in zip(shifts, G_S)])

        return G_S_shifted
    
    with h5py.File(f'{savename}_saliency.h5', 'a') as h5f:
        ds = h5f.require_group(dataset.county)

        if 'X' in ds:
            del ds['X']
            del ds['Y']
        if 'G_S' in ds:
            del ds['G_S']
        
        ds.create_dataset('X', data=dataset.x) # REMEMBER TO SUBTRACT MEAN TO ALIGN
        ds.create_dataset('Y', data=dataset.y)

        gs = ds.require_group('G_S')
        for i in trange(len(dataset)):
            sample = dataset[i]
            G_S = aggregate_sample(model, sample)
            gs.create_dataset(f'{int(sample["t"])}', data=G_S)

        ds.create_dataset(
            'G_S_sum',
            data=np.sum(np.asarray([gs[t] for t in gs.keys()]), axis=(0,1))
        )
            

'''
Building blocks
'''
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dropout_rate=0.):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding='same', padding_mode='replicate', groups=in_channels)
        self.conv2 = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(4*out_channels, out_channels, kernel_size=1)
        
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
class CensusPBNN(nn.Module):
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
    
        if hidden_size % input_dim != 0:
            self.read_in = nn.Sequential(
                nn.Conv2d(input_dim, 4, kernel_size=1),
                ConvNextBlock(4, hidden_size, kernel_size, dropout_rate)
            )
        else:
            self.read_in = ConvNextBlock(input_dim, hidden_size, kernel_size, dropout_rate)

        self.downsample = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=4, stride=4),
            nn.GELU()
        )
        self.read_out = nn.Conv2d(2*hidden_size, output_dim, kernel_size=1)
        
        self.cnn1 = nn.ModuleList()
        self.cnn2 = nn.ModuleList()
        for i in range(N_hidden):
            self.cnn1.append(ConvNextBlock(hidden_size, hidden_size, kernel_size))
            self.cnn2.append(ConvNextBlock(hidden_size, hidden_size, kernel_size))
            
        self.blur = transforms.GaussianBlur(kernel_size, sigma=(0.1, 3))
        self.device = torch.device('cpu')        
    
    def training_step(self, sample):
        problem = sample['problem']
        params = self.forward(sample['wb0'][None], 
                              problem.FctSpace, 
                              (sample['x'], sample['y']))
        params['Si'] = params['Si'].flatten()
        problem.set_params(**params)
        loss = problem.residual()
        grad = problem.grad(sample['wb0'].device)

        params['Si'].backward(gradient=grad['Si'] / sample['dt'])
        
        return params, loss
    
    def validation_step(self, sample):
        problem = sample['problem']
        params = self.forward(sample['wb0'][None], 
                              problem.FctSpace, 
                              (sample['x'], sample['y']))
        
        problem.set_params(**params)
        loss = problem.residual()
                
        return params, loss
    
    def forward(self, wb0, FctSpace, xy):
        #Neural network part
        wb0[wb0.isnan()] = 0
        
        x = self.read_in(wb0)
        for cell in self.cnn1:
            x = x + cell(x)
        latent = self.downsample(x)
        for cell in self.cnn2:
            latent = latent + cell(latent)
        latent = F.interpolate(latent, x.shape[-2:])
        x = torch.cat([x, latent], dim=1)
        Si = self.read_out(x).squeeze() # Remove batch dimension

        if self.training:
            Si  = self.blur(Si)

        if FctSpace is None:
            return Si # For computing saliency predictions, we can stop here
        
        Si_mesh  = torch.zeros([Si.shape[0],  FctSpace.dim()], dtype=Si.dtype,  device=Si.device)
        for i in range(Si.shape[0]):
            Si_mesh[i] = scalar_img_to_mesh(Si[i], *xy, FctSpace, vals_only=True)

        return Si_mesh
    
    def simulate(self, sample, mesh, device, tmax=40, dt=1):
        problem = sample['problem']
        problem.dt.assign(d_ad.Constant(dt))
        FctSpace = problem.FctSpace
        
        Dt = d_ad.Constant(dt)
        x_img = sample['x']
        y_img = sample['y']
        mask = sample['mask']
        
        # Interpolate initial condition to the mesh for the Dolfin problem
        for i in range(self.input_dim):
            problem.wb0[i].assign(scalar_img_to_mesh(sample['wb0'][i], x_img, y_img, FctSpace))

        #Interpolate w0, b0 to grid for neural network
        wb = torch.FloatTensor(np.stack([
            mesh_to_scalar_img(problem.wb0[i], mesh, x_img, y_img, mask) \
            for i in range(self.input_dim)
        ])).to(device)
        
        steps = int(np.round(tmax / dt))
        for tt in range(steps):
            #Run PBNN to get sources and assign to variational parameters
            Si = self.forward(wb[None], FctSpace, (x_img, y_img))
            problem.set_params(Si=Si.detach().cpu().numpy())
            
            #Solve variational problem and update w0, b0
            wb = problem.forward()
            w, b = wb.split(True)
            problem.wb0[0].assign(w)
            problem.wb0[1].assign(b)
            
            #Interpolate w0, b0 to grid for neural network
            wb = torch.FloatTensor(np.stack([
                mesh_to_scalar_img(problem.wb0[i], mesh, x_img, y_img, mask) \
                for i in range(self.input_dim)
            ])).to(device)
                                    
        return wb