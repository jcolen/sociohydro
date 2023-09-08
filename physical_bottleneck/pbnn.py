import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from data_processing import scalar_img_to_mesh, mesh_to_scalar_img

import dolfin as dlf
import dolfin_adjoint as d_ad
from tqdm.auto import trange, tqdm

'''
Training scripts
'''
def pretrain(model, dataset, n_epochs, batch_size, device, savedir='dynamic',
             pretrain_model=None,):
    '''
    Pretrain a model to produce reasonable or fixed parameters
    '''
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    loss_history = []

    if pretrain_model is None:
        gammas_init = torch.zeros(2, dtype=torch.float, device=device)
    else:
        gammas_init = pretrain_model.gammas.detach()

    print(f'Pretraining with target Gamma = {gammas_init.cpu().numpy()}')

    step = 0
    batch_size = 8
    idxs = np.arange(len(dataset), dtype=int)

    with tqdm(total=n_epochs*len(dataset)) as ebar:
        for epoch in range(n_epochs):
            np.random.shuffle(idxs)
            d_ad.set_working_tape(d_ad.Tape())
            for i in range(len(dataset)):
                batch = dataset[idxs[i]]
                batch['wb0'] = batch['wb0'].to(device)
                batch['mask'] = torch.BoolTensor(batch['mask']).to(device)

                with d_ad.stop_annotating(), torch.no_grad():
                    if pretrain_model is None:
                        batch['Dij'] = batch['Dij'].to(device)
                        batch['gammas'] = gammas_init
                    else:
                        Dij, _, gammas = pretrain_model.forward(batch['wb0'][None])
                        batch['Dij'] = Dij
                        batch['gammas'] = gammas

                loss, Dij, gammas = model.pretrain(batch)
                loss_history.append(loss.item())
                step += 1
                ebar.update()

                if step % batch_size == 0:
                    ebar.set_postfix(
                        loss=np.mean(loss_history[-len(dataset):]),
                        gammas=gammas)

                    opt.step()
                    d_ad.set_working_tape(d_ad.Tape())
                    opt.zero_grad()


            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'loss_history': loss_history,
                },
                f'{savedir}/{model.__class__.__name__}_pretrain.ckpt')

            sch.step()
            
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

                    Dij, Dij_mesh, constants, J, dJdD = model.training_step(batch)
                    train_loss.append(J)
                    step += 1
                    ebar.update()

                    if step % batch_size == 0:
                        ebar.set_postfix(
                            loss=np.mean(train_loss[-batch_size:]),
                            gammas=constants.detach().cpu().numpy())

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

                        Dij, Dij_mesh, constants, J = model.validation_step(batch)
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
        self.read_out = nn.Conv2d(hidden_size, 4, kernel_size=1)
        self.cnn = nn.ModuleList()
        for i in range(N_hidden):
            self.cnn.append(ConvNextBlock(hidden_size, hidden_size, kernel_size))
            
        self.blur = transforms.GaussianBlur(kernel_size, sigma=1)
        self.device = torch.device('cpu')        
    
    def pretrain(self, sample):        
        Dij, _, gammas = self.forward(sample['wb0'][None])
        if 'mask' in sample:
            mask = sample['mask']
            loss = F.mse_loss(Dij[:, mask], sample['Dij'][:, mask])
        else:
            loss = F.mse_loss(Dij, sample['Dij'])
        
        loss += F.mse_loss(gammas, sample['gammas'])
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
        D = self.read_out(D).squeeze()
        
        Dij = self.blur(D)
        gammas = self.gammas.exp() #Gammas are positive
                
        if FctSpace is not None:
            assert xy is not None
            Dij_mesh = torch.zeros([Dij.shape[0], FctSpace.dim()], dtype=Dij.dtype, device=Dij.device)
            for i in range(Dij.shape[0]):
                Dij_mesh[i] = scalar_img_to_mesh(Dij[i], *xy, FctSpace, vals_only=True)
                
            return Dij, Dij_mesh, gammas
    
        return Dij, None, gammas
    
    def simulate(self, sample, mesh, tmax, dt=0.1):
        FctSpace = sample['FctSpace']
        N = FctSpace.dim()
        Dww = d_ad.Function(FctSpace)
        Dwb = d_ad.Function(FctSpace)
        Dbw = d_ad.Function(FctSpace)
        Dbb = d_ad.Function(FctSpace)
        w0 = scalar_img_to_mesh(sample['wb0'][0], sample['x'], sample['y'], sample['FctSpace'])
        b0 = scalar_img_to_mesh(sample['wb0'][1], sample['x'], sample['y'], sample['FctSpace'])
        
        Dt = d_ad.Constant(dt)
        x_verts = mesh.coordinates()[0]
        y_verts = mesh.coordinates()[1]
        x_img = sample['x']
        y_img = sample['y']
        mask = sample['mask']
        
        #Interpolate w0, b0 to grid
        w_img = mesh_to_scalar_img(w0, mesh, x_img, y_img, mask)
        b_img = mesh_to_scalar_img(b0, mesh, x_img, y_img, mask)
        wb = torch.FloatTensor(np.stack([w_img, b_img])).to(sample['wb0'].device)
        
        steps = int(np.round(tmax / dt))
        for i in trange(steps):
            #Run PBNN to get Dij, constants and assign to variational parameters
            _, Dij, gammas = self.forward(wb[None], sample['FctSpace'], (sample['x'], sample['y']))
            Dij = Dij.detach().cpu().numpy()
            gammas = gammas.detach().cpu().numpy()
            Dww.vector()[:] = Dij[0]
            Dwb.vector()[:] = Dij[1]
            Dbw.vector()[:] = Dij[2]
            Dbb.vector()[:] = Dij[3]
            GammaW = d_ad.Constant(gammas[0])
            GammaB = d_ad.Constant(gammas[1])
                                               
            #Solve variational problem and update w0, b0
            wb, _, _ = sample['pde_forward'](Dww, Dwb, Dbw, Dbb, GammaW, GammaB, Dt, w0, b0, mesh)
            w, b = wb.split(True)
            w0.assign(w)
            b0.assign(b)
            
            #Interpolate w0, b0 to grid
            w_img = mesh_to_scalar_img(w0, mesh, x_img, y_img, mask)
            b_img = mesh_to_scalar_img(b0, mesh, x_img, y_img, mask)
            wb = torch.FloatTensor(np.stack([w_img, b_img])).to(sample['wb0'].device)
                                    
        return w0, b0, wb
        
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
        sqAB = torch.sqrt(D[0].exp() * D[3].exp())
        Dij[1] += D[2].tanh() * sqAB # |D_+| < sqrt(D_a D_b)
        Dij[2] += D[2].tanh() * sqAB # |D_+| < sqrt(D_a D_b)
        Dij = self.blur(Dij)
        
        gammas = self.gammas * 0
                
        if FctSpace is not None:
            assert xy is not None
            Dij_mesh = torch.zeros([Dij.shape[0], FctSpace.dim()], dtype=Dij.dtype, device=Dij.device)
            for i in range(Dij.shape[0]):
                Dij_mesh[i] = scalar_img_to_mesh(Dij[i], *xy, FctSpace, vals_only=True)
                
            return Dij, Dij_mesh, gammas
    
        return Dij, None, gammas
    
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
        wb0[wb0.isnan()] = 0
        D = self.read_in(wb0)
        for cell in self.cnn:
            D = D + cell(D)
        D = self.read_out(D).squeeze()
        Dij = torch.zeros_like(D)

        #Eigenvalues must be positive
        '''
        Dij[3] += D[3] #D_{BB} can be anything
        Dij[0] += D[0].exp() - D[3] #D_{AA} runs from -D_{BB} to infinity

        bound1 = D[3] * (D[0].exp() - D[3]) #D_{AA} * D_{BB}
        bound2 = -0.25 * (D[0].exp() - 2*D[3]).pow(2) #-(D_{AA} - D_{BB})^2 / 4
        
        #D_{BW} \times D_{WB} lies between bound1 and bound2
        lower = torch.min(torch.stack([bound1, bound2]), dim=0)[0]
        upper = torch.max(torch.stack([bound1, bound2]), dim=0)[0]
        prod = lower + D[1].tanh() * (upper - lower) #tanh satisfies boundaries
        Dij[1] += prod / D[2]
        Dij[1, D[2] == 0] = prod[D[2] == 0]
        Dij[2] += D[2]
        '''
        Dij += D

        Dij = self.blur(Dij)
        
        gammas = self.gammas * 0
                
        if FctSpace is not None:
            assert xy is not None
            Dij_mesh = torch.zeros([Dij.shape[0], FctSpace.dim()], dtype=Dij.dtype, device=Dij.device)
            for i in range(Dij.shape[0]):
                Dij_mesh[i] = scalar_img_to_mesh(Dij[i], *xy, FctSpace, vals_only=True)
                
            return Dij, Dij_mesh, gammas
    
        return Dij, None, gammas