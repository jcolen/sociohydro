import torch
import numpy as np
import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy

from scipy.interpolate import griddata

dlf.set_log_level(40)

def mesh_to_scalar(f, mesh):
    return f.compute_vertex_values()

def scalar_to_mesh(f, FctSpace, vals_only=False):    
    d2v = dlf.dof_to_vertex_map(FctSpace).astype(np.int64)

    if not isinstance(f, np.ndarray):
        d2v = torch.tensor(d2v, dtype=torch.long, device=f.device)
    fct_vals = f[d2v]
        
    if vals_only:
        return fct_vals
    else:
        meshfct = d_ad.Function(FctSpace)

        if not isinstance(f, np.ndarray):
            meshfct.vector()[:] = fct_vals.detach().cpu().numpy()
        else:
            meshfct.vector()[:] = fct_vals
        return meshfct
    
class SimulationProblem:
    '''
    Generate reduced functional for calculating dJ/dD
    Following github.com/schmittms/physical_bottleneck
    '''    
    def __init__(self, dataset, sample):
        el = ufl.FiniteElement('CG', dataset.mesh.ufl_cell(), 1)
        self.FctSpace = dlf.FunctionSpace(dataset.mesh, el)
        self.VV = dlf.FunctionSpace(dataset.mesh, el*el)
       
        self.Si =  [d_ad.Function(self.FctSpace) for i in range(2)]
        for i in range(len(self.Si)):
            self.Si[i].vector()[:]  = 0.
        
        args = (dataset.x, self.FctSpace)
        self.dt = d_ad.Constant(sample['dt'], name='dt')
        self.ab0 = [scalar_to_mesh(sample['ab0'][i], self.FctSpace) for i in range(2)]
        ab1 = [scalar_to_mesh(sample['ab1'][i], self.FctSpace) for i in range(2)]

        ab = self.forward()
        a, b = ab.split(True)
        J = d_ad.assemble(ufl.dot(a - ab1[0], a - ab1[0]) * ufl.dx + \
                          ufl.dot(b - ab1[1], b - ab1[1]) * ufl.dx)

        #Build controls to allow for updating values
        control_Si     = [d_ad.Control(self.Si[i]) for i in range(2)]

        controls = [*control_Si]
        self.Jhat = ReducedFunctionalNumPy(J, controls)

        control_arr = [p.data() for p in self.Jhat.controls]
        self.controls = self.Jhat.obj_to_array(control_arr)
        
    def residual(self):
        return self.Jhat(self.controls)
    
    def grad(self, device):
        dJdD = self.Jhat.derivative(self.controls, forget=True, project=False)
        dJdD = torch.tensor(dJdD, device=device)
        return {
            'Si':     dJdD,
        }
    
    def forward(self):
        '''Solve forward problem for given diffusion coefficients and Gamma terms'''
        a, b = dlf.TrialFunctions(self.VV)
        u, v = dlf.TestFunctions(self.VV)
        
        Sa, Sb = self.Si
        a0, b0 = self.ab0
        dt = self.dt

        #dt(\phi_i) - grad(Dij grad(phi_j)) = 0
        #Note that Danny's paper uses the opposite sign on Dij
        a = a*u*ufl.dx + b*v*ufl.dx

        L = a0*u*ufl.dx + b0*v*ufl.dx
        L -= dt * Sa * u * ufl.dx
        L -= dt * Sb * v * ufl.dx

        ab = d_ad.Function(self.VV)
        d_ad.solve(a == L, ab)
        return ab
    
    def set_params(self, Si=None, **kwargs):
        N = self.FctSpace.dim()
        if Si is not None:
            if not isinstance(Si, np.ndarray):
                self.controls[:] = Si.detach().cpu().numpy().flatten()
            else:
                self.controls[:] = Si.flatten()
            
            self.Si[0].vector()[:] = self.controls[:N]
            self.Si[1].vector()[:] = self.controls[N:]