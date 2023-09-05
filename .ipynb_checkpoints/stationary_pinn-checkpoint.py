import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from pinn import *
from convnext_models import Sin
        
class StationaryPINN(PINN):
    def __init__(self, 
                 x_u, y_u, #Coordinates
                 w_u, b_u,      #Populations
                 N_h=256,       #Hidden width
                 N_l=5,         #Hidden layers
                 act=Sin,
                 lbfgs_lr=1e0,
                 adam_lr=3e-4,
                 adam_patience=500,
                 print_every=1000,
                 save_every=5000):
        nn.Module.__init__(self)
        self.n_inputs = 2
        self.n_outputs = 2
        self.N_h = N_h
        self.N_l = N_l
        self.act = act
        self.lbfgs_lr = lbfgs_lr
        self.adam_lr = adam_lr
        self.adam_patience = adam_patience
        self.print_every = print_every
        self.save_every = save_every
        self.iter = 0
        
        self.x_u = nn.Parameter(x_u[:, None], requires_grad=True)
        self.y_u = nn.Parameter(y_u[:, None], requires_grad=True)
        self.w_u = nn.Parameter(w_u[:, None], requires_grad=True)
        self.b_u = nn.Parameter(b_u[:, None], requires_grad=True)
        
        self.w_scale = self.w_u.pow(2).mean().item()
        self.b_scale = self.b_u.pow(2).mean().item()
        
        X = torch.stack([y_u, x_u], dim=-1)
        self.lb = nn.Parameter(X.min(0)[0], requires_grad=False)
        self.ub = nn.Parameter(X.max(0)[0], requires_grad=False)
        self.ub[self.ub == self.lb] += 1e-3
        
        self.setup_model()
        self.setup_optimizers()
        
    def training_step(self):
        wb_u = self(torch.cat([self.y_u, self.x_u], dim=-1))
        mse = self.mse_loss(wb_u)
        phys = self.phys_loss(wb_u)
                
        return mse, phys
    
    def mse_loss(self, wb_u):
        mse = 1 * (wb_u[:, 0:1] - self.w_u).pow(2).mean() / self.w_scale + \
              1 * (wb_u[:, 1:2] - self.b_u).pow(2).mean() / self.b_scale
        return mse
    
class UninformedStationaryPINN(StationaryPINN):
    def phys_loss(self, wb):
        return torch.zeros_like(wb).sum()
        
    def equation_string(self):
        return ''
    
class LinearDiffusionStationaryPINN(StationaryPINN):
    def phys_loss(self, wb):
        y, x = self.y_u, self.x_u
        
        D_wb = linear_diffusion(wb, y, x, self.coefs)
                
        return D_wb.pow(2).mean(dim=0).sum()
        
    def equation_string(self):
        coefs = self.coefs[:, 0].detach().cpu().numpy()
        eq = '\n'
        eq += f'\tdt P_w = {coefs[0]:.3f} grad^2 P_w + {coefs[1]:.3f} grad^2 P_b = 0\n'
        eq += f'\tdt P_b = {coefs[2]:.3f} grad^2 P_w + {coefs[3]:.3f} grad^2 P_b = 0\n'
        return eq
    
class CubicDiffusionStationaryPINN(StationaryPINN):
    def phys_loss(self, wb):
        y, x = self.y_u, self.x_u
        
        D_wb = cubic_diffusion(wb, y, x, self.coefs)
                
        return D_wb.pow(2).mean(dim=0).sum()
        
    def equation_string(self):
        coefs = self.coefs.detach().cpu().numpy()
        facs = ['', 'P_w', 'P_b', 'P_w^2', 'P_b^2', 'P_w P_b']
        eq = '\n'
        eq +=  f'dt P_w = div( D_wi grad(P_i)) = 0\n'
        eq += '\tD_ww =  ' + ' + '.join([f'{c:.3f} {f}' for c, f in zip(coefs[0], facs)]) + '\n'
        eq += '\tD_wb =  ' + ' + '.join([f'{c:.3f} {f}' for c, f in zip(coefs[1], facs)]) + '\n'
        
        eq += f'dt P_b = div( D_bi grad(P_i)) = 0\n'
        eq += '\tD_bw =  ' + ' + '.join([f'{c:.3f} {f}' for c, f in zip(coefs[2], facs)]) + '\n'
        eq += '\tD_bb =  ' + ' + '.join([f'{c:.3f} {f}' for c, f in zip(coefs[3], facs)]) + '\n'
            
        return eq