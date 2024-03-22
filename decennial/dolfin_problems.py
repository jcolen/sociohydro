import torch
import numpy as np
import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy

from scipy.interpolate import griddata

dlf.set_log_level(40)


def plot_mesh(ax, f, mesh, lw=False, scale=1, vmin=0, vmax=1, **kwargs):
    x, y = mesh.coordinates()[:, 0], mesh.coordinates()[:, 1]
    tri = mesh.cells()
    if lw:
        kwargs['edgecolors'] = 'black'
        kwargs['linewidth'] = 0.05
    if not isinstance(f, np.ndarray):
        ax.tripcolor(x, y, f.compute_vertex_values() * scale, triangles=tri, vmin=vmin, vmax=vmax, **kwargs)
    else:
        ax.tripcolor(x, y, f * scale, triangles=tri, vmin=vmin, vmax=vmax, **kwargs)
    ax.set(xticks=[], yticks=[])

def mesh_to_scalar_img(f, mesh, x, y, mask):
    f_verts = f.compute_vertex_values()
    x_verts = mesh.coordinates()[:, 0]
    y_verts = mesh.coordinates()[:, 1]
    
    f_img = griddata((x_verts, y_verts), f_verts, (x, y))
    f_img[~mask] = 0
    return f_img

def scalar_img_to_mesh(img, x, y, FctSpace, vals_only=False):
    '''
    from github.com/schmittms/physical_bottleneck
    '''
    dof_coords = FctSpace.tabulate_dof_coordinates().copy() # Shape (Nnodes, 2)
        
    if isinstance(img, np.ndarray):
        mask = ~np.isnan(img)
        fct_vals = griddata((x[mask], y[mask]), img[mask], dof_coords, method='nearest')
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

        if not isinstance(img, np.ndarray):
            meshfct.vector()[:] = fct_vals.detach().cpu().numpy()
        else:
            meshfct.vector()[:] = fct_vals
        return meshfct
    
class SociohydrodynamicsProblem:
    '''
    Generate reduced functional for calculating dJ/dD
    Following github.com/schmittms/physical_bottleneck

    Using dt_phi = div(J) from Danny's paper
    '''    
    def __init__(self, dataset, sample):
        el = ufl.FiniteElement('CG', dataset.mesh.ufl_cell(), 1)
        self.FctSpace = dlf.FunctionSpace(dataset.mesh, el)
        self.VV = dlf.FunctionSpace(dataset.mesh, el*el)
       
        self.Gammas = [d_ad.Constant(0.) for i in range(2)]
        self.Dij = [d_ad.Function(self.FctSpace) for i in range(4)]
        self.Si =  [d_ad.Function(self.FctSpace) for i in range(2)]
        
        for i in range(len(self.Dij)):
            self.Dij[i].vector()[:] = 0.
        for i in range(len(self.Si)):
            self.Si[i].vector()[:]  = 0.
        
        args = (dataset.x, dataset.y, self.FctSpace)
        self.dt = d_ad.Constant(sample['dt'], name='dt')
        self.wb0 = [scalar_img_to_mesh(sample['wb0'][i], *args) for i in range(2)]
        wb1 = [scalar_img_to_mesh(sample['wb1'][i], *args) for i in range(2)]

        wb = self.forward()
        w, b = wb.split(True)
        J = d_ad.assemble(ufl.dot(w - wb1[0], w - wb1[0]) * ufl.dx + \
                          ufl.dot(b - wb1[1], b - wb1[1]) * ufl.dx)

        #Build controls to allow for updating values
        control_Dij    = [d_ad.Control(self.Dij[i]) for i in range(4)]
        control_Si     = [d_ad.Control(self.Si[i]) for i in range(2)]
        control_Gammas = [d_ad.Control(self.Gammas[i]) for i in range(2)]

        controls = [*control_Dij, *control_Si, *control_Gammas]
        self.Jhat = ReducedFunctionalNumPy(J, controls)

        control_arr = [p.data() for p in self.Jhat.controls]
        self.controls = self.Jhat.obj_to_array(control_arr)
        
    def residual(self):
        return self.Jhat(self.controls)
    
    def grad(self, device):
        dJdD = self.Jhat.derivative(self.controls, forget=True, project=False)
        dJdD = torch.tensor(dJdD, device=device)
        N = self.FctSpace.dim()
        return {
            'DS':     dJdD[:-2],
            'Gammas': dJdD[-2:],
        }
    
    def forward(self):
        '''Solve forward problem for given diffusion coefficients and Gamma terms'''
        w, b = dlf.TrialFunctions(self.VV)
        u, v = dlf.TestFunctions(self.VV)
        
        Dww, Dwb, Dbw, Dbb = self.Dij
        Sw, Sb = self.Si
        GammaW, GammaB = self.Gammas
        w0, b0 = self.wb0
        dt = self.dt

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
        L -= dt * Sw * u * ufl.dx
        L -= dt * Sb * v * ufl.dx

        wb = d_ad.Function(self.VV)
        d_ad.solve(a == L, wb)
        return wb
    
    def set_params(self, Dij=None, Si=None, Gammas=None, **kwargs):
        N = self.FctSpace.dim()
        if Dij is not None:
            s = np.s_[:N*4]
            if not isinstance(Dij, np.ndarray):
                self.controls[s] = Dij.detach().cpu().numpy().flatten()
            else:
                self.controls[s] = Dij.flatten()
            for i in range(4):
                self.Dij[i].vector()[:] = self.controls[i*N:(i+1)*N]
                
        if Si is not None:
            s = np.s_[N*4:N*6]
            if not isinstance(Si, np.ndarray):
                self.controls[s] = Si.detach().cpu().numpy().flatten()
            else:
                self.controls[s] = Si.flatten()
            for i in range(2):
                self.Si[i].vector()[:] = self.controls[(4+i)*N:(5+i)*N]
        
        if Gammas is not None:
            s = np.s_[-2:]
            if not isinstance(Gammas, np.ndarray):
                self.controls[s] = Gammas.detach().cpu().numpy().flatten()
            else:
                self.controls[s] = Gammas.flatten()
            self.Gammas[0].value = self.controls[-2]
            self.Gammas[1].value = self.controls[-1]

class TwoDemographicsDynamics:
    '''
    Generate reduced functional for calculating dJ/dD
    Following github.com/schmittms/physical_bottleneck

    Using dt_phi = S_i
    '''    
    def __init__(self, dataset, sample):
        el = ufl.FiniteElement('CG', dataset.mesh.ufl_cell(), 1)
        self.mesh_area = dataset.mesh_area
        self.FctSpace = dlf.FunctionSpace(dataset.mesh, el)
        self.VV = dlf.FunctionSpace(dataset.mesh, dlf.MixedElement([el,el]))
       
        self.Si =  [d_ad.Function(self.FctSpace) for i in range(2)]
        for i in range(len(self.Si)):
            self.Si[i].vector()[:]  = 0.
        
        args = (dataset.x, dataset.y, self.FctSpace)
        self.dt = d_ad.Constant(sample['dt'], name='dt')
        self.wb0 = [scalar_img_to_mesh(sample['wb0'][i], *args) for i in range(2)]
        wb1 = [scalar_img_to_mesh(sample['wb1'][i], *args) for i in range(2)]

        scale = 1. / dataset.mesh_area
        
        #print(f'Scale = {scale}')
        
        wb = self.forward()
        w, b= wb.split(True)
        J = scale * d_ad.assemble(
            ufl.dot(w - wb1[0], w - wb1[0]) * ufl.dx + \
            ufl.dot(b - wb1[1], b - wb1[1]) * ufl.dx)

        #Build controls to allow for updating values
        controls = [d_ad.Control(self.Si[i]) for i in range(2)]
        self.Jhat = ReducedFunctionalNumPy(J, controls)

        control_arr = [p.data() for p in self.Jhat.controls]
        self.controls = self.Jhat.obj_to_array(control_arr)
        
    def residual(self):
        return self.Jhat(self.controls)
    
    def grad(self, device):
        dJdD = self.Jhat.derivative(self.controls, forget=True, project=False)
        return {
            'Si': torch.tensor(dJdD, device=device),
        }
    
    def forward(self):
        '''Solve forward problem for given diffusion coefficients and Gamma terms'''
        w, b = dlf.TrialFunctions(self.VV)
        u, v = dlf.TestFunctions(self.VV)
        
        Sw, Sb = self.Si
        w0, b0 = self.wb0
        dt = self.dt

        #dt(\phi_i) - grad(Dij grad(phi_j)) = S_i
        #Note that Danny's paper uses the opposite sign on Dij
        a = w*u*ufl.dx + b*v*ufl.dx

        L = w0*u*ufl.dx + b0*v*ufl.dx
        L -= dt * Sw * u * ufl.dx
        L -= dt * Sb * v * ufl.dx

        wb = d_ad.Function(self.VV)
        d_ad.solve(a == L, wb)
        return wb
    
    def set_params(self, Si=None, **kwargs):
        N = self.FctSpace.dim()
        if Si is not None:
            if not isinstance(Si, np.ndarray):
                self.controls[:] = Si.detach().cpu().numpy().flatten()
            else:
                self.controls[:] = Si.flatten()
            for i in range(2):
                self.Si[i].vector()[:] = self.controls[i*N:(1+i)*N]            
            
class ThreeDemographicsDynamics:
    '''
    Generate reduced functional for calculating dJ/dD
    Following github.com/schmittms/physical_bottleneck

    Using dt_phi = S_i
    '''    
    def __init__(self, dataset, sample):
        el = ufl.FiniteElement('CG', dataset.mesh.ufl_cell(), 1)
        self.mesh_area = dataset.mesh_area
        self.FctSpace = dlf.FunctionSpace(dataset.mesh, el)
        self.VVV = dlf.FunctionSpace(dataset.mesh, dlf.MixedElement([el,el,el]))
       
        self.Si =  [d_ad.Function(self.FctSpace) for i in range(3)]
        for i in range(len(self.Si)):
            self.Si[i].vector()[:]  = 0.
        
        args = (dataset.x, dataset.y, self.FctSpace)
        self.dt = d_ad.Constant(sample['dt'], name='dt')
        self.wb0 = [scalar_img_to_mesh(sample['wb0'][i], *args) for i in range(3)]
        wb1 = [scalar_img_to_mesh(sample['wb1'][i], *args) for i in range(3)]

        scale = 1. / dataset.mesh_area
        
        #print(f'Scale = {scale}')
        
        wb = self.forward()
        w, b, h = wb.split(True)
        J = scale * d_ad.assemble(
            ufl.dot(w - wb1[0], w - wb1[0]) * ufl.dx + \
            ufl.dot(b - wb1[1], b - wb1[1]) * ufl.dx + \
            ufl.dot(h - wb1[2], h - wb1[2]) * ufl.dx)

        #Build controls to allow for updating values
        controls = [d_ad.Control(self.Si[i]) for i in range(3)]
        self.Jhat = ReducedFunctionalNumPy(J, controls)

        control_arr = [p.data() for p in self.Jhat.controls]
        self.controls = self.Jhat.obj_to_array(control_arr)
        
    def residual(self):
        return self.Jhat(self.controls)
    
    def grad(self, device):
        dJdD = self.Jhat.derivative(self.controls, forget=True, project=False)
        return {
            'Si': torch.tensor(dJdD, device=device),
        }
    
    def forward(self):
        '''Solve forward problem for given diffusion coefficients and Gamma terms'''
        w, b, h = dlf.TrialFunctions(self.VVV)
        u, v, x = dlf.TestFunctions(self.VVV)
        
        Sw, Sb, Sh = self.Si
        w0, b0, h0 = self.wb0
        dt = self.dt

        #dt(\phi_i) - grad(Dij grad(phi_j)) = S_i
        #Note that Danny's paper uses the opposite sign on Dij
        a = w*u*ufl.dx + b*v*ufl.dx + h*x*ufl.dx

        L = w0*u*ufl.dx + b0*v*ufl.dx + h0*x*ufl.dx
        L -= dt * Sw * u * ufl.dx
        L -= dt * Sb * v * ufl.dx
        L -= dt * Sh * x * ufl.dx

        wb = d_ad.Function(self.VVV)
        d_ad.solve(a == L, wb)
        return wb
    
    def set_params(self, Si=None, **kwargs):
        N = self.FctSpace.dim()
        if Si is not None:
            if not isinstance(Si, np.ndarray):
                self.controls[:] = Si.detach().cpu().numpy().flatten()
            else:
                self.controls[:] = Si.flatten()
            for i in range(3):
                self.Si[i].vector()[:] = self.controls[i*N:(1+i)*N]