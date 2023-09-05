import torch
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
import numpy as np
import h5py

import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad

dlf.set_log_level(40)


def smooth_with_fill(arr, sigma=3):
    msk = np.isnan(arr)
    mask = np.where(~msk)

    interp = NearestNDInterpolator(np.transpose(mask), arr[mask])
    arr = interp(*np.indices(arr.shape))
    arr = gaussian_filter(arr, sigma=sigma)
    arr[msk] = np.nan
    return arr

def scalar_img_to_mesh(img, x, y, FctSpace, vals_only=False):
    '''
    from github.com/schmittms/physical_bottleneck
    '''
    dof_coords = FctSpace.tabulate_dof_coordinates().copy() # Shape (Nnodes, 2)
    
    if isinstance(img, np.ndarray):
        mask = ~np.isnan(img)
        fct_vals = griddata((x[mask], y[mask]), img[mask], dof_coords)
    else:
        dof_coords = torch.tensor(dof_coords, dtype=img.dtype, device=img.device)
        xmin, ymin = x.min(), y.min()
        dx = x[1,1]-x[0,0]
        dy = y[1,1]-y[0,0]
        dof_x = (dof_coords[:, 0] - xmin) / dx
        dof_y = (dof_coords[:, 1] - ymin) / dy
        
        fct_vals = img[dof_y.long(), dof_x.long()]
        fct_vals[fct_vals.isnan()] = fct_vals[~fct_vals.isnan()].mean()
                        
    if vals_only:
        return fct_vals
    else:
        meshfct = d_ad.Function(FctSpace)
        meshfct.vector()[:] = fct_vals
        return meshfct

class CensusDataset(torch.utils.data.Dataset):
    def __init__(self,
                 county='cook_IL',
                 spatial_scale=1e3,):
        self.county = county
        self.spatial_scale = spatial_scale
        self.train = True
        self.init_data()
        
    def training(self):
        self.train = True
    
    def validate(self):
        self.train = False
    
    def init_data(self):
        with h5py.File(f'/home/jcolen/sociohydro/data/{self.county}.hdf5', 'r') as d:
            x_grid = d["x_grid"][:] / self.spatial_scale
            y_grid = d["y_grid"][:] / self.spatial_scale
            w_grid = d["w_grid_array_masked"][:].transpose(2, 0, 1)
            b_grid = d["b_grid_array_masked"][:].transpose(2, 0, 1)
            for i in range(5):
                w_grid[i] = smooth_with_fill(w_grid[i])
                b_grid[i] = smooth_with_fill(b_grid[i])

        self.x = x_grid
        self.y = y_grid
        self.t = np.array([1980, 1990, 2000, 2010, 2020], dtype=float)
        
        #Convert population to occupation fraction
        wb = np.stack([w_grid, b_grid], axis=1)
        max_grid = np.sum(wb, axis=1).max(axis=0)
        wb /= max_grid
        
        self.mask = np.all(~np.isnan(wb), axis=(0, 1))
        self.wb = interp1d(self.t, wb, axis=0, fill_value='extrapolate')
        self.housing = max_grid
        
        self.mesh = d_ad.Mesh(f'/home/jcolen/sociohydro/data/{self.county}_mesh.xml')
            
    def get_dolfin(self, sample):
        '''
        Generate reduced functional for calculating dJ/dD
        Following github.com/schmittms/physical_bottleneck
        '''
        D_init = np.ones_like(self.x)
        el = ufl.FiniteElement('CG', self.mesh.ufl_cell(), 1)
        V  = dlf.FunctionSpace(self.mesh, el)    #W/B function space
        VV = dlf.FunctionSpace(self.mesh, el*el)
        
        def forward(Dww, Dwb, Dbw, Dbb, GammaW, GammaB, dt, w0, b0, mesh):
            '''Solve forward problem for given diffusion coefficients and Gamma terms'''
            w, b = dlf.TrialFunctions(VV)
            u, v = dlf.TestFunctions(VV)
            
            #dt(\phi_i) - grad(Dij grad(phi_j)) = 0
            #Note that Danny's paper uses the opposite sign on Dij
            a = w*u*ufl.dx + b*v*ufl.dx
            a += dt * ufl.dot(Dww * ufl.grad(w), ufl.grad(u)) * ufl.dx
            a += dt * ufl.dot(Dwb * ufl.grad(b), ufl.grad(u)) * ufl.dx
            a += dt * ufl.dot(Dbw * ufl.grad(w), ufl.grad(v)) * ufl.dx
            a += dt * ufl.dot(Dbb * ufl.grad(b), ufl.grad(v)) * ufl.dx
            
            a -= dt * GammaW * ufl.dot((1 - w0 - b0) * w0 * ufl.grad(ufl.div(ufl.grad(w))), ufl.grad(u)) * ufl.dx
            a -= dt * GammaB * ufl.dot((1 - w0 - b0) * b0 * ufl.grad(ufl.div(ufl.grad(b))), ufl.grad(v)) * ufl.dx
            
            L = w0*u*ufl.dx + b0*v*ufl.dx
            
            wb = d_ad.Function(VV)
            d_ad.solve(a == L, wb)
            return wb, a, L
            
        GammaW = d_ad.Constant(0., name='gammaW')
        GammaB = d_ad.Constant(0., name='gammaB')
        dt = d_ad.Constant(sample['dt'], name='dt')
        
        Dij_init = np.zeros([4, *self.x.shape])
        Dij_init[0, sample['mask']] = 1
        Dij_init[3, sample['mask']] = 1

        Dww = scalar_img_to_mesh(Dij_init[0], self.x, self.y, V)
        Dwb = scalar_img_to_mesh(Dij_init[1], self.x, self.y, V)
        Dbw = scalar_img_to_mesh(Dij_init[2], self.x, self.y, V)
        Dbb = scalar_img_to_mesh(Dij_init[3], self.x, self.y, V)
        w0 = scalar_img_to_mesh(sample['wb0'][0], self.x, self.y, V)
        b0 = scalar_img_to_mesh(sample['wb0'][1], self.x, self.y, V)
        
        w1 = scalar_img_to_mesh(sample['wb1'][0], self.x, self.y, V)
        b1 = scalar_img_to_mesh(sample['wb1'][1], self.x, self.y, V)
        
        wb, a, L = forward(Dww, Dwb, Dbw, Dbb, GammaW, GammaB, dt, w0, b0, self.mesh)
        w, b = wb.split(True)
        J = d_ad.assemble(ufl.dot(w - w1, w - w1) * ufl.dx + \
                          ufl.dot(b - b1, b - b1) * ufl.dx)
        
        #Build controls to allow for updating values
        control_Dww = d_ad.Control(Dww) 
        control_Dwb = d_ad.Control(Dwb) 
        control_Dbw = d_ad.Control(Dbw) 
        control_Dbb = d_ad.Control(Dbb)
        control_GammaW = d_ad.Control(GammaW)
        control_GammaB = d_ad.Control(GammaB)
        control_dt = d_ad.Control(dt)
        control_w0 = d_ad.Control(w0)
        control_b0 = d_ad.Control(b0)
        
        controls = [control_Dww, control_Dwb, control_Dbw, control_Dbb,
                    control_GammaW, control_GammaB, control_dt, control_w0, control_b0]
        Jhat_np = pyad.ReducedFunctionalNumPy(J, controls)
        Jhat =    pyad.ReducedFunctional(J, controls)
        
        control_arr = [p.data() for p in Jhat_np.controls]
        control_arr = Jhat_np.obj_to_array(control_arr)
        
        return {
            'Jhat': Jhat_np,
            'control_arr': control_arr,
            'pde_forward': forward,
            'FctSpace': V,
            'base_RF': Jhat,
            'J': J,
            'Dij': torch.FloatTensor(Dij_init),
        }
    
    
    def get_time(self, t, dt=1):
        wb0 = self.wb(t)
        wb1 = self.wb(t+dt)
        
        sample = {
            't': t,
            'x': self.x,
            'y': self.y,
            'mask': self.mask,
            'dt': dt,
            'wb0': self.wb(t),
            'wb1': self.wb(t+dt),
        }
        pde = self.get_dolfin(sample)
        return {**sample, **pde}
    
    def __len__(self):
        return int(np.ptp(self.t))
    
    def __getitem__(self, idx):
        t0 = self.t[0] + idx
        dt = 1
        if self.train:
            t0 += np.random.random()
            dt *= np.random.random()
            
        sample = self.get_time(t0, dt)
        sample['wb0'] = torch.FloatTensor(sample['wb0'])
                
        return sample

class StationaryDataset(CensusDataset):
    def get_dolfin(self, sample):
        '''
        Generate reduced functional for calculating dJ/dD
        Following github.com/schmittms/physical_bottleneck
        '''
        D_init = np.ones_like(self.x)
        el = ufl.FiniteElement('CG', self.mesh.ufl_cell(), 1)
        V  = dlf.FunctionSpace(self.mesh, el)    #W/B function space
        VV = dlf.FunctionSpace(self.mesh, el*el)
        
        def forward(Dww, Dwb, Dbw, Dbb, GammaW, GammaB, dt, w0, b0, mesh):
            '''Solve forward problem for given diffusion coefficients and Gamma terms'''
            w, b = dlf.TrialFunctions(VV)
            u, v = dlf.TestFunctions(VV)
            
            #dt(\phi_i) - grad(Dij grad(phi_j)) = 0
            #Note that Danny's paper uses the opposite sign on Dij
            #Stationary problem requires symmetric Dij
            #To prevent Dij -> 0, we define diagonal components are positive and set DAA = 1 + DAA
            Dpl = (Dwb + Dbw) / 2
            a = w*u*ufl.dx + b*v*ufl.dx
            a += dt * ufl.dot((1+Dww) * ufl.grad(w), ufl.grad(u)) * ufl.dx
            a += dt * ufl.dot(Dpl * ufl.grad(b), ufl.grad(u)) * ufl.dx
            a += dt * ufl.dot(Dpl * ufl.grad(w), ufl.grad(v)) * ufl.dx
            a += dt * ufl.dot(Dbb * ufl.grad(b), ufl.grad(v)) * ufl.dx
            
            a -= dt * GammaW * ufl.dot((1 - w0 - b0) * w0 * ufl.grad(ufl.div(ufl.grad(w))), ufl.grad(u)) * ufl.dx
            a -= dt * GammaB * ufl.dot((1 - w0 - b0) * b0 * ufl.grad(ufl.div(ufl.grad(b))), ufl.grad(v)) * ufl.dx
            
            L = w0*u*ufl.dx + b0*v*ufl.dx
            
            wb = d_ad.Function(VV)
            d_ad.solve(a == L, wb)
            return wb, a, L
            
        GammaW = d_ad.Constant(0., name='gammaW')
        GammaB = d_ad.Constant(0., name='gammaB')
        dt = d_ad.Constant(sample['dt'], name='dt')
        
        Dij_init = np.zeros([4, *self.x.shape])
        Dij_init[0, sample['mask']] = 1
        Dij_init[3, sample['mask']] = 1

        Dww = scalar_img_to_mesh(Dij_init[0], self.x, self.y, V)
        Dwb = scalar_img_to_mesh(Dij_init[1], self.x, self.y, V)
        Dbw = scalar_img_to_mesh(Dij_init[2], self.x, self.y, V)
        Dbb = scalar_img_to_mesh(Dij_init[3], self.x, self.y, V)
        w0 = scalar_img_to_mesh(sample['wb0'][0], self.x, self.y, V)
        b0 = scalar_img_to_mesh(sample['wb0'][1], self.x, self.y, V)
        
        wb, a, L = forward(Dww, Dwb, Dbw, Dbb, GammaW, GammaB, dt, w0, b0, self.mesh)
        w, b = wb.split(True)
        J = d_ad.assemble(ufl.dot(w - w0, w - w0) * ufl.dx + \
                          ufl.dot(b - b0, b - b0) * ufl.dx)
        
        #Build controls to allow for updating values
        control_Dww = d_ad.Control(Dww) 
        control_Dwb = d_ad.Control(Dwb) 
        control_Dbw = d_ad.Control(Dbw) 
        control_Dbb = d_ad.Control(Dbb)
        control_GammaW = d_ad.Control(GammaW)
        control_GammaB = d_ad.Control(GammaB)
        control_dt = d_ad.Control(dt)
        control_w0 = d_ad.Control(w0)
        control_b0 = d_ad.Control(b0)
        
        controls = [control_Dww, control_Dwb, control_Dbw, control_Dbb,
                    control_GammaW, control_GammaB, control_dt, control_w0, control_b0]
        Jhat_np = pyad.ReducedFunctionalNumPy(J, controls)
        Jhat =    pyad.ReducedFunctional(J, controls)
        
        control_arr = [p.data() for p in Jhat_np.controls]
        control_arr = Jhat_np.obj_to_array(control_arr)
        
        return {
            'Jhat': Jhat_np,
            'control_arr': control_arr,
            'pde_forward': forward,
            'FctSpace': V,
            'base_RF': Jhat,
            'J': J,
            'Dij': torch.FloatTensor(Dij_init),
        }
    
    
    def get_time(self, t):
        wb0 = self.wb(t)
        
        sample = {
            't': t,
            'x': self.x,
            'y': self.y,
            'mask': self.mask,
            'dt': 1,
            'wb0': self.wb(t),
        }
        pde = self.get_dolfin(sample)
        return {**sample, **pde}
    
    def __getitem__(self, idx):
        t0 = self.t[0] + idx
        if self.train:
            t0 += np.random.random()
            
        sample = self.get_time(t0)
        sample['wb0'] = torch.FloatTensor(sample['wb0'])
                
        return sample