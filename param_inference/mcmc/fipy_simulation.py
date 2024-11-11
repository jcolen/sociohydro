import sys
sys.path.insert(0, '..')

from fvm_utils import *
import pymc
import json

from fipy.tools.dump import read
from fipy_dataset import FipyDataset
import glob
from argparse import Namespace

class FipyLongTimeDataset(FipyDataset):
    """ Dataset that loads longer time sequences from fipy """
    def __init__(self, path="../data/Georgia_Fulton_small/fipy_output", tmax=5):
        super().__init__(path=path)
        self.tmax = tmax

        param_file = glob.glob(f'{path}/../*.json')[0]
        with open(param_file) as f:
            self.params = json.load(f)

    def __len__(self):
        return len(self.files) - self.tmax

    def __getitem__(self, idx):
        if self.data:
            W0, B0, t0 = self.data[idx]
            W1, B1, t1 = self.data[idx+self.tmax]
        else:
            W0, B0, t0 = read(self.files[idx])
            W1, B1, t1 = read(self.files[idx+self.tmax])

        sample = {
            'W0_mesh': W0,
            'B0_mesh': B0,
            'W1_mesh': W1,
            'B1_mesh': B1,
            't0': t0,
            't1': t1
        }

        return sample

def init_variables(sample):
    mesh = sample['W0_mesh'].mesh
    ϕW = sample['W0_mesh']
    μW = fp.CellVariable(name=r"muW", mesh=mesh, hasOld=True)
    ϕB = sample['B0_mesh']
    μB = fp.CellVariable(name=r"muB", mesh=mesh, hasOld=True)
    ϕ0 = 1 - ϕW - ϕB

    return ϕW, μW, ϕB, μB, ϕ0

def init_equations(ϕW, μW, ϕB, μB, ϕ0, params):
    args = Namespace(**params)
    # Unpack parameters and variables
    TW   = args.tempW
    ΓW   = args.gammaW
    κWW  = args.kWW
    κWB  = args.kWB
    νWWW = args.nuWWW
    νWWB = args.nuWWB
    νWBB = args.nuWBB
    rW = args.growthW

    TB   = args.tempB
    ΓB   = args.gammaB
    κBB  = args.kBB
    κBW  = args.kBW
    νBBB = args.nuBBB
    νBWB = args.nuBWB
    νBWW = args.nuBWW
    rB = args.growthB

    # Build intermediate terms
    mobilityW = ϕW * ϕ0
    πW = κWW * ϕW + κWB * ϕB + νWWW * ϕW * ϕW + νWWB * ϕW * ϕB + νWBB * ϕB * ϕB
    dπWdϕW = κWW + 2 * νWWW * ϕW + νWWB * ϕB
    μW_taylorExpand = -πW + TW * (np.log(ϕW) - np.log(ϕ0))
    dμWdϕW = -dπWdϕW + TW * (1 - ϕB) / (ϕW * ϕ0)
    #SW = rW[0] + rW[1] * ϕW + rW[2] * ϕB + rW[3] * ϕW * ϕW + rW[4] * ϕW * ϕB + rW[5] * ϕB * ϕB
    SW = rW

    mobilityB = ϕB * ϕ0
    πB = κBW * ϕW + κBB * ϕB + νBWW * ϕW * ϕW + νBWB * ϕW * ϕB + νBBB * ϕB * ϕB
    dπBdϕB = κBB + νBWB * ϕW + 2 * νBBB * ϕB
    μB_taylorExpand = -πB + TB * (np.log(ϕB) - np.log(ϕ0))
    dμBdϕB = -dπBdϕB + TB * (1 - ϕW) / (ϕB * ϕ0)
    #SB = rB[0] + rB[1] * ϕW + rB[2] * ϕB + rB[3] * ϕW * ϕW + rB[4] * ϕW * ϕB + rB[5] * ϕB * ϕB
    SB = rB
    
    eqW_1 = (fp.TransientTerm(var=ϕW) == fp.DiffusionTerm(coeff=mobilityW, var=μW) + SW)
    eqW_2 = (fp.ImplicitSourceTerm(coeff=1, var=μW)
             == fp.ImplicitSourceTerm(coeff=dμWdϕW, var=ϕW)
             - dμWdϕW * ϕW + μW_taylorExpand
             - fp.DiffusionTerm(coeff=ΓW, var=ϕW))
    
    eqB_1 = (fp.TransientTerm(var=ϕB) == fp.DiffusionTerm(coeff=mobilityB, var=μB) + SB)
    eqB_2 = (fp.ImplicitSourceTerm(coeff=1, var=μB)
             == fp.ImplicitSourceTerm(coeff=dμBdϕB, var=ϕB)
             - dμBdϕB * ϕB + μB_taylorExpand
             - fp.DiffusionTerm(coeff=ΓB, var=ϕB))
    
    eq = eqW_1 & eqW_2 & eqB_1 & eqB_2
    return eq

def run_simulation(sample, params):
    ϕW, μW, ϕB, μB, ϕ0 = init_variables(sample)
    equations = init_equations(ϕW, μW, ϕB, μB, ϕ0, params)

    t0, t1 = sample['t0'], sample['t1']
    dt = params.get('dt', 0.1)
    t_curr = t0

    while t_curr < t1:
        for var in [ϕW, μW, ϕB, μB]:
            var.updateOld()
        res = 1e10
        while res > 1e-5:
            res = equations.sweep(dt=dt)
        t_curr += dt
    
    mseB = np.mean((sample['B1_mesh'] - ϕB.value)**2)
    mseW = np.mean((sample['W1_mesh'] - ϕW.value)**2)

    return ϕB, ϕW, mseB, mseW

from time import time
if __name__ == '__main__':
    dataset = FipyLongTimeDataset(tmax=5)
    sample = dataset[10]

    t0, t1 = sample['t0'], sample['t1']
    print(f'Simulating from {t0:.2g} to {t1:.2g}')

    t = time()
    ϕB, ϕW, mseB, mseW = run_simulation(sample, dataset.params)
    
    print(f'Simulation finished in {time()-t:.3g}s with MSE {mseB:.3g}, {mseW:.3g}')
