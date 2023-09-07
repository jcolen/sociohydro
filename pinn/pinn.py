import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

'''
Building blocks
'''
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
def init_weights(m):
    '''
    Initialization of weights in linear layers using xavier uniform
    '''
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
        
def gradient(x, *y):
    '''
    Gradient of x wrt each element of y
    Returns [*x.shape, len(y)] array
    '''
    x0 = x.view([x.shape[0], -1])
    dxdy = []
    for i in range(x0.shape[1]):
        for j in range(len(y)):
            xi = x0[..., i]
            grad_outputs = torch.ones_like(xi)
            grad = autograd.grad(xi, y[j], grad_outputs=grad_outputs, create_graph=True)[0]
            dxdy.append(grad)
    dxdy = torch.stack(dxdy, dim=-1)
    dxdy = dxdy.reshape([*x.shape, len(y)])
    return dxdy

def div(x, *y):
    '''
    Divergence of vector x in coordinate system given by y
    Returns [*x.shape] array
    '''
    x0 = x.view([x.shape[0], -1, len(y)])
    div = torch.zeros_like(x0[..., 0])
    for i in range(x0.shape[1]):
        for j in range(len(y)):
            xij = x0[..., i, j]
            grad_outputs = torch.ones_like(xij)
            grad = autograd.grad(xij, y[j], grad_outputs=grad_outputs, create_graph=True)[0]
            div[..., i:i+1] += grad
    div = div.view(x.shape[:-1])
    return div

class PINN(nn.Module):
    def setup_model(self):
        
        layers = [self.n_inputs,] + [self.N_h,]*self.N_l + [self.n_outputs,]
        print(layers)
        lst = []
        for i in range(len(layers)-2):
            lst.append(nn.Linear(layers[i], layers[i+1]))
            lst.append(self.act())
        lst.append(nn.Linear(layers[-2], layers[-1]))
        self.model = nn.Sequential(*lst)
                
        self.gammas = nn.Parameter(torch.zeros(2, dtype=torch.float), requires_grad=True)
        self.model.register_parameter('gammas', self.gammas)
        
        self.apply(init_weights)
    
    def setup_optimizers(self):
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.lbfgs_lr,
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-9, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.lbfgs_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.lbfgs, gamma=0.95)
        
        self.adam = torch.optim.Adam(self.model.parameters(), lr=self.adam_lr)
        self.adam_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.adam, 
                                                                         patience=self.adam_patience,
                                                                         factor=0.5,
                                                                         min_lr=1e-5,
                                                                         verbose=True)

    def __init__(self, 
                 x_u, y_u, t_u, #Coordinates
                 w_u, b_u,      #Populations
                 N_h=256,       #Hidden width
                 N_l=5,         #Hidden layers
                 act=Sin,
                 lbfgs_lr=1e0,
                 adam_lr=3e-4,
                 adam_patience=500,
                 print_every=1000,
                 save_every=5000):
        super().__init__()
        self.n_inputs = 3
        self.n_outputs = 6 #[W, B, 4 Dij components]
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
        self.t_u = nn.Parameter(t_u[:, None], requires_grad=True)
        self.w_u = nn.Parameter(w_u[:, None], requires_grad=True)
        self.b_u = nn.Parameter(b_u[:, None], requires_grad=True)
        
        self.w_scale = self.w_u.pow(2).mean().item()
        self.b_scale = self.b_u.pow(2).mean().item()
        
        X = torch.stack([t_u, y_u, x_u], dim=-1)
        self.lb = nn.Parameter(X.min(0)[0], requires_grad=False)
        self.ub = nn.Parameter(X.max(0)[0], requires_grad=False)
        self.ub[self.ub == self.lb] += 1e-3 #Remove divide by zero error in scaling
        
        self.setup_model()
        self.setup_optimizers()
        
    def print(self, loss=None, mse=None, phys=None):
        outstr = ''
        if loss is None:
            mse, phys = self.training_step()
            loss = mse + phys
        outstr += f'{self.__class__.__name__} Iter = {self.iter}\tLoss = {loss.item():.3e}, MSE = {mse.item():.3e}, Phys = {phys.item():.3e}'
        outstr += self.equation_string()
        print(outstr, flush=True)
        
    def train(self, n_lbfgs=0, n_adam=500000):        
        if n_lbfgs > 0:
            self.optimizer = self.lbfgs
            self.scheduler = self.lbfgs_scheduler
            print('Starting L-BFGS optimization', flush=True)
            for i in range(3):
                self.optimizer.step(self.loss_func)
                self.scheduler.step()
            print(f'Done after {self.iter}', flush=True)
        if n_adam > 0:
            self.optimizer = self.adam
            self.scheduler = self.adam_scheduler
            print('Starting Adam optimization', flush=True)
            for i in range(n_adam):
                loss = self.loss_func()
                self.optimizer.step()
                self.scheduler.step(loss)
            print('Done', flush=True)
        
        self.print()
        self.save()
        
    def save(self, mse=None, phys=None):
        if mse is None:
            mse, phys = self.training_step()
        torch.save(dict(state_dict=self.model.state_dict(),
                hparams=dict(N_h=self.N_h, 
                             N_l=self.N_l, 
                             lbfgs_lr=self.lbfgs_lr, 
                             adam_lr=self.adam_lr),
                mse=mse,
                phys=phys,
                iteration=self.iter),
           f'./{self.__class__.__name__}')

    def forward(self, X):
        H = 2. * (X - self.lb) / (self.ub - self.lb) - 1.0
        y = self.model(H)
        return y
        
    def loss_func(self):
        mse, phys = self.training_step()
        loss = mse + phys
        self.optimizer.zero_grad()
        loss.backward()
        
        self.iter += 1
        if self.iter % self.print_every == 0:
            self.print(loss, mse, phys)
        
        if self.iter % self.save_every == 0:
            self.save(mse, phys)
        
        return loss
    
    def mse_loss(self, wb_u):
        mse = (wb_u[:, 0:1] - self.w_u).pow(2).mean() / self.w_scale + \
              (wb_u[:, 1:2] - self.b_u).pow(2).mean() / self.b_scale
        return mse
    
    def training_step(self):
        wb_u = self(torch.cat([self.t_u, self.y_u, self.x_u], dim=-1))
        mse = self.mse_loss(wb_u)
        phys = self.phys_loss(wb_u)
                
        return mse, phys
    
    def phys_loss(self, wbD):
        t, y, x = self.t_u, self.y_u, self.x_u
        
        dt_wb = gradient(wbD[:, 0:2], t)[..., 0] #[N, 2]
        D_wb = self.diffusion(wbD, y, x) #[N, 2]
        G_wb = self.gamma_term(wbD[:, 0:2], y, x)
        
        f_wb = dt_wb - D_wb + G_wb
        
        return f_wb.pow(2).mean(dim=0).sum()
    
    def diffusion(self, wbD, y, x):
        wb = wbD[:, 0:2] #[N, 2]
        Dij = wbD[:, 2:6] #[N, 4]  
        #No restrictions on Diagonal coefficients
        
        grad_wb = gradient(wb, y, x) #[N, 2, 2]
        lapl_wb = div(grad_wb, y, x) #[N, 2]
        
        grad_Dij = gradient(Dij, y, x)#[N, 4, 2]
        
        D_w = torch.einsum('bj,bj->b', Dij[:,0:2], lapl_wb) + \
              torch.einsum('bij,bij->b', grad_Dij[:,0:2], grad_wb)
        D_b = torch.einsum('bj,bj->b', Dij[:,2:4], lapl_wb) + \
              torch.einsum('bij,bij->b', grad_Dij[:,2:4], grad_wb)
        
        return torch.stack([D_w, D_b], dim=1)
    
    def gamma_term(self, wb, y, x):
        '''
        Use the full gamma term Gamma_i \nabla \cdot [(1 - \sum_j \phi_j) \phi_i \nabla^3 \phi_i]
        '''
        grad_wb = gradient(wb, y, x) #[N, 2, 2]
        lapl_wb = div(grad_wb, y, x) #[N, 2]
        d3_wb = gradient(lapl_wb, y, x) #[N, 2, 2]
        bihr_wb = div(d3_wb, y, x) #[N, 2]

        pref = 1 - torch.sum(wb, dim=1, keepdims=True) #[N, 1]
        grad_pref = -torch.sum(grad_wb, dim=1, keepdims=True) #[N, 1, 2]

        G_wb  = pref * wb * bihr_wb #[N, 2]
        G_wb += pref * torch.einsum('bij,bij->bi', grad_wb, d3_wb) #[N, 2]
        G_wb += torch.einsum('bkj,bi,bij->bki', grad_pref, wb, d3_wb)[:, 0] #[N, 2]

        gammas = self.gammas.exp()[None] #[1, 2]

        return gammas * G_wb
            
    def equation_string(self):
        gammas = self.gammas.exp().detach().cpu().numpy()

        eq = '\n'
        eq +=  f'  dt P_w = div( D_wi grad(P_i) - {gammas[0]:.3g} (1 - P) P_w grad^3 P_w )\n'  
        eq +=  f'  dt P_b = div( D_bi grad(P_i) - {gammas[1]:.3g} (1 - P) P_b grad^3 P_b )\n'
            
        return eq
    
class LinearGammaPINN(PINN):
    def gamma_term(self, wb, y, x):
        '''
        Use only the linear gamma term Gamma_i \nabla^4 \phi_i
        gamma = [2]
        '''
        grad_wb = gradient(wb, y, x) #[N, 2, 2]
        lapl_wb = div(grad_wb, y, x) #[N, 2]
        d3_wb = gradient(lapl_wb, y, x) #[N, 2, 2]
        bihr_wb = div(d3_wb, y, x) #[N, 2]

        gammas = self.gammas.exp()[None] #[1, 2]

        return gammas * G_wb
    
class DiagonalOnlyPINN(PINN):
    '''
    Model local diffusion matrix with a NN
    No off-diagonal components, no Gamma terms
    Linear stability requires both diagonal diffusion coefficients are positive
    '''
    def get_diffusion(self, D):
        #Diagonal coefficients are positive
        Dij = torch.zeros_like(D)
        Dij[:, 0] += D[:, 0].exp()
        Dij[:, 3] += D[:, 3].exp()
        return Dij
    
    def diffusion(self, wbD, y, x):
        wb = wbD[:, 0:2] #[N, 2]
        D = wbD[:, 2:6] #[N, 4]
        
        #Diagonal coefficients are positive
        Dij = torch.zeros_like(D)
        Dij[:, 0] += D[:, 0].exp()
        Dij[:, 3] += D[:, 3].exp()
        
        grad_wb = gradient(wb, y, x) #[N, 2, 2]
        lapl_wb = div(grad_wb, y, x) #[N, 2]
        
        grad_Dij = gradient(Dij, y, x)#[N, 4, 2]
        
        D_w = torch.einsum('bj,bj->b', Dij[:,0:2], lapl_wb) + \
              torch.einsum('bij,bij->b', grad_Dij[:,0:2], grad_wb)
        D_b = torch.einsum('bj,bj->b', Dij[:,2:4], lapl_wb) + \
              torch.einsum('bij,bij->b', grad_Dij[:,2:4], grad_wb)
        
        return torch.stack([D_w, D_b], dim=1)
    
    def gamma_term(self, wb, y, x):
        return torch.zeros_like(wb)
    
    def equation_string(self):
        eq = '\n'
        eq +=  f'  dt P_a = div( D_aa grad(P_a))\n'
        eq +=  f'  dt P_b = div( D_bb grad(P_b))\n'
            
        return eq
    
class SymmetricCrossDiffusionPINN(PINN):
    '''
    Model local diffusion matrix with a NN
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
    def get_diffusion(self, D):
        #Diagonal coefficients are positive
        Dij = torch.zeros_like(D)
        Dij[:, 0] += D[:, 0].exp()
        Dij[:, 3] += D[:, 3].exp()
        sqAB = torch.sqrt(D[:,0].exp() * D[:,3].exp()) #[N,]
        Dij[:, 1] += D[:, 2].tanh() * sqAB # |D_+| < sqrt(D_a D_b)
        Dij[:, 2] += D[:, 2].tanh() * sqAB # |D_+| < sqrt(D_a D_b)
        return Dij
    
    def diffusion(self, wbD, y, x):
        wb = wbD[:, 0:2] #[N, 2]
        D = wbD[:, 2:6] #[N, 4]
        
        #Diagonal coefficients are positive
        Dij = torch.zeros_like(D)
        Dij[:, 0] += D[:, 0].exp()
        Dij[:, 3] += D[:, 3].exp()
        sqAB = torch.sqrt(D[:,0].exp() * D[:,3].exp()) #[N,]
        Dij[:, 1] += D[:, 2].tanh() * sqAB # |D_+| < sqrt(D_a D_b)
        Dij[:, 2] += D[:, 2].tanh() * sqAB # |D_+| < sqrt(D_a D_b)
        
        grad_wb = gradient(wb, y, x) #[N, 2, 2]
        lapl_wb = div(grad_wb, y, x) #[N, 2]
        
        grad_Dij = gradient(Dij, y, x)#[N, 4, 2]
        
        D_w = torch.einsum('bj,bj->b', Dij[:,0:2], lapl_wb) + \
              torch.einsum('bij,bij->b', grad_Dij[:,0:2], grad_wb)
        D_b = torch.einsum('bj,bj->b', Dij[:,2:4], lapl_wb) + \
              torch.einsum('bij,bij->b', grad_Dij[:,2:4], grad_wb)
        
        return torch.stack([D_w, D_b], dim=1)
    
    def gamma_term(self, wb, y, x):
        return torch.zeros_like(wb)
    
    def equation_string(self):
        eq = '\n'
        eq +=  f'  dt P_a = div( D_aa grad(P_a) + D_+ grad(P_b))\n'
        eq +=  f'  dt P_b = div( D_bb grad(P_b) + D_+ grad(P_b))\n'
            
        return eq