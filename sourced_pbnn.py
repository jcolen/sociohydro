import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from dolfin_problems import *

import dolfin as dlf
import dolfin_adjoint as d_ad
from tqdm.auto import trange, tqdm

'''
Training scripts
'''         
def train(model, dataset, n_epochs, batch_size, device, savedir='dynamic'):
    '''
    Train a model
    '''
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    train_loss = []
    val_loss = []
    step = 0
    
    idxs = np.arange(len(dataset), dtype=int)

    
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            for ds in dataset.datasets:
                ds.training()
            np.random.shuffle(idxs)
            d_ad.set_working_tape(d_ad.Tape())

            with tqdm(total=len(dataset), leave=False) as ebar:
                for i in range(len(dataset)):
                    batch = dataset[idxs[i]]
                    batch['wb0'] = batch['wb0'].to(device)

                    params, J = model.training_step(batch)
                    train_loss.append(J)
                    step += 1
                    ebar.update()

                    if step % batch_size == 0:
                        ebar.set_postfix(
                            loss=np.mean(train_loss[-batch_size:]),
                            gammas=params['Gammas'].detach().cpu().numpy())

                        opt.step()
                        d_ad.set_working_tape(d_ad.Tape())
                        opt.zero_grad()

            for ds in dataset.datasets:
                ds.validate()
            val_loss.append(0)

            with tqdm(total=len(dataset), leave=False) as ebar:
                with torch.no_grad():
                    for i in range(len(dataset)):
                        d_ad.set_working_tape(d_ad.Tape())
                        batch = dataset[i]
                        batch['wb0'] = batch['wb0'].to(device)

                        params, J = model.validation_step(batch)
                        val_loss[epoch] += J
                        ebar.update()


            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                },
                f'{savedir}/{model.__class__.__name__}.ckpt')

            sch.step()
            pbar.update()
            pbar.set_postfix(val_loss=val_loss[-1])



'''
Building blocks
'''
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding='same', padding_mode='replicate', groups=in_channels)
        self.conv2 = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(4*out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sin(x)
        x = self.conv3(x)
        return x
    
'''
Neural network
'''
class PBNN(nn.Module):
    '''
    CNN computes local diffusion matrix from phi_W/B inputs
    The restriction is that Gamma is positive and there is 
        no further restriction on D_{ij}
    '''
    def __init__(self,
                 kernel_size=7,
                 N_hidden=8,
                 hidden_size=64):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.N_hidden = N_hidden
        self.hidden_size = hidden_size
        
        self.gammas = nn.Parameter(torch.zeros(2, dtype=torch.float), requires_grad=True)
        
        self.read_in = ConvNextBlock(2, hidden_size, kernel_size)
        self.read_out = nn.Conv2d(hidden_size, 6, kernel_size=1)
        self.cnn = nn.ModuleList()
        for i in range(N_hidden):
            self.cnn.append(ConvNextBlock(hidden_size, hidden_size, kernel_size))
            
        self.blur = transforms.GaussianBlur(kernel_size, sigma=2.5)
        self.device = torch.device('cpu')        
    
    def training_step(self, sample):
        problem = sample['problem']
        params = self.forward(sample['wb0'][None], 
                              problem.FctSpace, 
                              (sample['x'], sample['y']))
        for key in params:
            params[key] = params[key].flatten()
        problem.set_params(**params)
        loss = problem.residual()
        grad = problem.grad(sample['wb0'].device)

        DS = torch.cat([params['Dij'], params['Si']])
        DS.backward(gradient=grad['DS'] / sample['dt'])
        params['Gammas'].backward(gradient=grad['Gammas'] / sample['dt'])
        
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
        assert FctSpace is not None
        assert xy is not None
        
        wb0[wb0.isnan()] = 0
        DS = self.read_in(wb0)
        for cell in self.cnn:
            DS = DS + cell(DS)
        DS = self.read_out(DS).squeeze()
        D = DS[:4]
        S = DS[4:]
        
        Dij = self.blur(D)
        Si  = self.blur(S)
        gammas = self.gammas.exp() #Gammas are positive∂

        Dij_mesh = torch.zeros([Dij.shape[0], FctSpace.dim()], dtype=Dij.dtype, device=Dij.device)
        Si_mesh  = torch.zeros([Si.shape[0],  FctSpace.dim()], dtype=Si.dtype,  device=Si.device)
        for i in range(Dij.shape[0]):
            Dij_mesh[i] = scalar_img_to_mesh(Dij[i], *xy, FctSpace, vals_only=True)
        for i in range(Si.shape[0]):
            Si_mesh[i] = scalar_img_to_mesh(Si[i], *xy, FctSpace, vals_only=True)
        
        return {
            'Dij': Dij_mesh, 
            'Si': Si_mesh, 
            'Gammas': gammas,
        }
    
    def simulate(self, sample, mesh, tmax, dt=0.1):
        problem = sample['problem']
        problem.dt.assign(d_ad.Constant(dt))
        FctSpace = problem.FctSpace
        
        Dt = d_ad.Constant(dt)
        x_img = sample['x']
        y_img = sample['y']
        mask = sample['mask']
        device = sample['wb0'].device
        
        #Interpolate w0, b0 to grid
        for i in range(2):
            problem.wb0[i].assign(scalar_img_to_mesh(sample['wb0'][i], x_img, y_img, FctSpace))

        wb = torch.FloatTensor(np.stack([
            mesh_to_scalar_img(problem.wb0[i], mesh, x_img, y_img, mask) for i in range(2)]))
        wb = wb.to(device)
        
        steps = int(np.round(tmax / dt))
        for tt in trange(steps):
            #Run PBNN to get Dij, constants and assign to variational parameters
            params = self.forward(wb[None], FctSpace, (x_img, y_img))
            params['Si'] *= 1
            problem.set_params(**params)
            
            #Solve variational problem and update w0, b0
            wb = problem.forward()
            w, b = wb.split(True)
            problem.wb0[0].assign(w)
            problem.wb0[1].assign(b)
            
            #Interpolate w0, b0 to grid
            wb = torch.FloatTensor(np.stack([
                mesh_to_scalar_img(problem.wb0[i], mesh, x_img, y_img, mask) for i in range(2)]))
            wb = wb.to(device)
                                    
        return problem.wb0[0], problem.wb0[1], wb
        
class DiagonalOnlyPBNN(PBNN):
    '''
    CNN computes local diffusion matrix from phi_W/B inputs
    No off-diagonal components, no Gamma terms
    Linear stability requires both diagonal diffusion coefficients are positive
    '''
    def forward(self, wb0, FctSpace, xy):
        params = super().forward(wb0, FctSpace, xy)
        params['Si'] *= 0 #Don't allow Si
        params['Gammas'] = params['Gammas'] * 0 #Don't allow gamma
        
        D = params['Dij'].clone()
        params['Dij'][1:3] *= 0
        params['Dij'][0] = D[0].exp()
        params['Dij'][3] = D[3].exp()
        
        return params
    
class SymmetricCrossDiffusionPBNN(PBNN):
    '''
    CNN computes local diffusion matrix from phi_W/B inputs
    Allow symmetric off-diagonal components, no gamma terms
    Linear stability requires both eigenvalues of diffusion matrix are positive
        For a matrix Dij = A    P
                           P    B
        The constraint that both eigenvalues are positive reduces to:
            A > 0
            B > 0
            P^2 < AB
    
    No Gamma terms
    Note that including anti-symmetric off-diagonal components requires
        the gamma terms for stability
    '''
    def forward(self, wb0, FctSpace, xy):
        params = super().forward(wb0, FctSpace, xy)
        params['Si'] *= 0 #Don't allow Si
        params['Gammas'] = params['Gammas'] * 0 #Don't allow gamma
        
        D = params['Dij']
        
        Dij = torch.zeros_like(D)
        Dij[0] += D[0].exp() #Positive D_A
        Dij[3] += D[3].exp() #Positive D_B
        sqAB = torch.sqrt(D[0].exp() * D[3].exp())
        Dij[1] += D[2].tanh() * sqAB # |D_+| < sqrt(D_a D_b)
        Dij[2] += D[2].tanh() * sqAB # |D_+| < sqrt(D_a D_b)
        params['Dij'] = Dij
        
        return params
    
class CrossDiffusionPBNN(PBNN):
    '''
    CNN computes local diffusion matrix from phi_W/B inputs
    Allow symmetric off-diagonal components, no gamma terms
    Linear stability requires both eigenvalues of diffusion matrix are positive
        For a matrix Dij = A   B
                           C   D
        The constraint that both eigenvalues are positive reduces to:
            A > 0, D > 0,  C = 0 || b in [ad/c, -(a-d)^2/4c] OR
            A < 0, D > -A, b in [ad/c, -(a-d)^2/4c] OR
            A > -D, D < 0, b in [ad/c, -(a-d)^2/4c]
        Note that in the limit b == c, the constraint on b reduces to
            b^2 in [ad, -(a-d)^2] -> b^2 < ad
            
        The common general constraints are A > -D, B in [AD/C, -(A-D)^2/4C]
        
    To enforce linear stability, we have the following pipeline:
        D can be anything
        A can be anything from -D to +infinity
        
    
    No Gamma terms
    '''
    def forward(self, wb0, FctSpace=None, xy=None):
        params = super().forward(wb0, FctSpace, xy)
        params['Si'] *= 0 #Don't allow Si
        params['Gammas'] = params['Gammas'] * 0 #Don't allow gamma
        return params
    
class SourcedDiffusionPBNN(PBNN):
    def forward(self, wb0, FctSpace=None, xy=None):
        params = super().forward(wb0, FctSpace, xy)
        params['Gammas'] = params['Gammas'] * 0 #Don't allow gamma
        return params
    
class SourcedSymmetricPBNN(PBNN):
    def forward(self, wb0, FctSpace, xy):
        params = super().forward(wb0, FctSpace, xy)
        params['Gammas'] = params['Gammas'] * 0 #Don't allow gamma
        
        D = params['Dij']
        
        Dij = torch.zeros_like(D)
        Dij[0] += D[0].exp() #Positive D_A
        Dij[3] += D[3].exp() #Positive D_B
        sqAB = torch.sqrt(D[0].exp() * D[3].exp())
        Dij[1] += D[2].tanh() * sqAB # |D_+| < sqrt(D_a D_b)
        Dij[2] += D[2].tanh() * sqAB # |D_+| < sqrt(D_a D_b)
        params['Dij'] = Dij
        
        return params
    

class DerivativePBNN(PBNN):
    def forward(self, wb0, FctSpace=None, xy=None):
        params = super().forward(wb0, FctSpace, xy)
        params['Gammas'] = params['Gammas'] * 0 #Don't allow gamma
        params['Dij'] = params['Dij'] * 0 #Don't allow Dij
        return params    