import torch
import numpy as np
import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy

from scipy.interpolate import griddata

dlf.set_log_level(40)

def mesh_to_scalar_img(f, mesh, x, y, mask):
    f_verts = f.compute_vertex_values()
    x_verts = mesh.coordinates()[:, 0]
    y_verts = mesh.coordinates()[:, 1]
    
    f_img = griddata((x_verts, y_verts), f_verts, (x, y))
    f_img[~mask] = 0

    # Handle remaining NaNs using nearest_neighbor interpolation
    if np.sum(np.isnan(f_img)) > 0:
        msk = np.isnan(f_img)
        f_msk = griddata((x_verts, y_verts), f_verts, (x[msk], y[msk]), method='nearest')
        f_img[msk] = f_msk
    
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
        self.dt = d_ad.Constant(sample['dt'], name='dt')

        self.Si = []  # Source terms
        self.wb0 = [] # Initial conditions
        self.wb1 = [] # Target values
        for i in range(2):
            Si = d_ad.Function(self.FctSpace)
            Si.vector()[:] = 0.
            self.Si.append(Si)

            self.wb0.append(scalar_img_to_mesh(sample['wb0'][i], dataset.x, dataset.y, self.FctSpace))
            self.wb1.append(scalar_img_to_mesh(sample['wb1'][i], dataset.x, dataset.y, self.FctSpace))

        # Normalize residual by county area
        # This prevents giant counties like Maricopa or San Bernardino from dominating the loss
        scale = 1. / dataset.mesh_area
        
        # Build the loss functional as the mean squared error between prediction and self.wb1
        wb = self.forward()
        w, b = wb.split(True)
        J = scale * d_ad.assemble(
            ufl.dot(w - self.wb1[0], w - self.wb1[0]) * ufl.dx + \
            ufl.dot(b - self.wb1[1], b - self.wb1[1]) * ufl.dx)

        #Build controls to update source term values
        controls = [d_ad.Control(self.Si[i]) for i in range(2)]
        self.Jhat = ReducedFunctionalNumPy(J, controls)

        control_arr = [p.data() for p in self.Jhat.controls]
        self.controls = self.Jhat.obj_to_array(control_arr)
        
    def residual(self):
        return self.Jhat(self.controls)
    
    def grad(self, device):
        dJdD = self.Jhat.derivative(self.controls, forget=True, project=False)
        return torch.tensor(dJdD, device=device)
    
    def forward(self):
        '''Solve forward problem for given source terms'''
        w, b = dlf.TrialFunctions(self.VV)
        u, v = dlf.TestFunctions(self.VV)
        
        Sw, Sb = self.Si
        w0, b0 = self.wb0
        dt = self.dt

        # dt(phi_i) = S_i 
        # phi_i (t0 + dt) = phi_i (t0) + dt * S_i
        a = w*u*ufl.dx + b*v*ufl.dx

        L = w0*u*ufl.dx + b0*v*ufl.dx
        L -= dt * Sw * u * ufl.dx
        L -= dt * Sb * v * ufl.dx

        wb = d_ad.Function(self.VV)
        d_ad.solve(a == L, wb)
        return wb
    
    def set_params(self, Si=None, **kwargs):
        N = self.FctSpace.dim()
        self.controls[:] = Si.flatten()
        for i in range(2):
            self.Si[i].vector()[:] = Si[i].flatten()
        return


        if Si is not None:
            if not isinstance(Si, np.ndarray):
                self.controls[:] = Si.detach().cpu().numpy().flatten()
            else:
                self.controls[:] = Si.flatten()
            for i in range(2):
                self.Si[i].vector()[:] = self.controls[i*N:(1+i)*N]