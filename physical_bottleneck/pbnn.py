import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from data_processing import scalar_img_to_mesh

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
        self.read_out = nn.Conv2d(hidden_size, 4, kernel_size=1)
        self.cnn = nn.ModuleList()
        for i in range(N_hidden):
            self.cnn.append(ConvNextBlock(hidden_size, hidden_size, kernel_size))
            
        self.blur = transforms.GaussianBlur(kernel_size, sigma=1)
        self.device = torch.device('cpu')        
    
    def pretrain(self, sample):        
        Dij, _, _ = self.forward(sample['wb0'][None])
        if 'mask' in sample:
            mask = sample['mask']
            loss = F.mse_loss(Dij[:, mask], sample['Dij'][:, mask])
        else:
            loss = F.mse_loss(Dij, sample['Dij'])
        
        loss += F.mse_loss(self.gammas, sample['gammas'])
        loss.backward()
        
        return loss, Dij.detach().cpu().numpy(), self.gammas.detach().cpu().numpy()
    
    def training_step(self, sample):
        Dij, Dij_mesh, constants = self.forward(sample['wb0'][None], 
                                                sample['FctSpace'], 
                                                (sample['x'], sample['y']))
        N = sample['FctSpace'].dim()
        D = len(Dij_mesh)

        Dij_mesh = Dij_mesh.flatten()
        control_arr = sample['control_arr'].copy()
        control_arr[:D*N] = Dij_mesh.detach().cpu().numpy()
        control_arr[D*N:D*N+len(constants)] = constants.detach().cpu().numpy()
        control_arr[D*N+len(constants)] = sample['dt']
        
        J = sample['Jhat'](control_arr)
        dJdD = sample['Jhat'].derivative(control_arr, forget=True, project=False)
        grad = torch.tensor(dJdD, device=Dij.device) / sample['dt']
        
        Dij_mesh.backward(gradient=grad[:D*N])
        constants.backward(gradient=grad[D*N:D*N+len(constants)])
        
        return Dij, Dij_mesh, constants, J, dJdD
    
    def validation_step(self, sample):
        Dij, Dij_mesh, constants = self.forward(sample['wb0'][None], 
                                                sample['FctSpace'], 
                                                (sample['x'], sample['y']))
        N = sample['FctSpace'].dim()
        D = len(Dij_mesh)

        Dij_mesh = Dij_mesh.flatten()
        control_arr = sample['control_arr'].copy()
        control_arr[:D*N] = Dij_mesh.detach().cpu().numpy()
        control_arr[D*N:D*N+len(constants)] = constants.detach().cpu().numpy()
        control_arr[D*N+len(constants)] = sample['dt']
        
        J = sample['Jhat'](control_arr)
        
        return Dij, Dij_mesh, constants, J
    
    def forward(self, wb0, FctSpace=None, xy=None):
        wb0[wb0.isnan()] = 0
        D = self.read_in(wb0)
        for cell in self.cnn:
            D = D + cell(D)
        Dij = self.read_out(D).squeeze()
        Dij = self.blur(Dij)
        
        gammas = self.gammas.exp() #Gammas are positive
                
        if FctSpace is not None:
            assert xy is not None
            Dij_mesh = torch.zeros([Dij.shape[0], FctSpace.dim()], dtype=Dij.dtype, device=Dij.device)
            for i in range(Dij.shape[0]):
                Dij_mesh[i] = scalar_img_to_mesh(Dij[i], *xy, FctSpace, vals_only=True)
                
            return Dij, Dij_mesh, gammas
    
        return Dij, None, gammas

class DiagonalOnlyPBNN(PBNN):
    '''
    CNN computes local diffusion matrix from phi_W/B inputs
    No off-diagonal components, no Gamma terms
    Linear stability requires both diagonal diffusion coefficients are positive
    '''
    def forward(self, wb0, FctSpace=None, xy=None):
        wb0[wb0.isnan()] = 0
        D = self.read_in(wb0)
        for cell in self.cnn:
            D = D + cell(D)
        D = self.read_out(D).squeeze()
        
        #Diagonal components should be positive
        Dij = torch.zeros_like(D)
        Dij[0] += D[0].exp() #Positive D_A
        Dij[3] += D[3].exp() #Positive D_B
        Dij = self.blur(Dij)
        
        gammas = self.gammas * 0
                
        if FctSpace is not None:
            assert xy is not None
            Dij_mesh = torch.zeros([Dij.shape[0], FctSpace.dim()], dtype=Dij.dtype, device=Dij.device)
            for i in range(Dij.shape[0]):
                Dij_mesh[i] = scalar_img_to_mesh(Dij[i], *xy, FctSpace, vals_only=True)
                
            return Dij, Dij_mesh, gammas
    
        return Dij, None, gammas
    
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
    def forward(self, wb0, FctSpace=None, xy=None):
        wb0[wb0.isnan()] = 0
        D = self.read_in(wb0)
        for cell in self.cnn:
            D = D + cell(D)
        D = self.read_out(D).squeeze()
        
        #Diagonal components should be positive
        Dij = torch.zeros_like(D)
        Dij[0] += D[0].exp() #Positive D_A
        Dij[3] += D[3].exp() #Positive D_B
        sqAB = torch.sqrt(Dij[0] * Dij[3])
        Dij[1:3] += D_base[2].tanh() * sqAB # |D_+| < sqrt(D_a D_b)
        Dij = self.blur(Dij)
        
        gammas = self.gammas * 0
                
        if FctSpace is not None:
            assert xy is not None
            Dij_mesh = torch.zeros([Dij.shape[0], FctSpace.dim()], dtype=Dij.dtype, device=Dij.device)
            for i in range(Dij.shape[0]):
                Dij_mesh[i] = scalar_img_to_mesh(Dij[i], *xy, FctSpace, vals_only=True)
                
            return Dij, Dij_mesh, gammas
    
        return Dij, None, gammas