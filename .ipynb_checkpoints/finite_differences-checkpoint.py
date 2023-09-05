import numpy as np
from numba import jit

def rk4(wb, dt, func):
    k1 = dt * func(wb)
    k2 = dt * func(wb + 0.5 * k1)
    k3 = dt * func(wb + 0.5 * k2)
    k4 = dt * func(wb + k3)
    return wb + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def build_masks(mask, mw=3):
    '''
    Figure out where we need to use forward, backward, and central differences
    mask: [Y, X] bool array of where the population is NOT
    mw: half-width of the finite-difference operator
    '''
    #Default behavior is to use central differences
    central_x  = np.ones(mask.shape, dtype=bool)
    central_y  = np.ones_like(central_x)
    forward_x  = np.zeros_like(central_x)
    forward_y  = np.zeros_like(central_x)
    backward_x = np.zeros_like(central_x)
    backward_y = np.zeros_like(central_x)

    #Cannot use central differences at image edge 
    central_y[:mw+1, :] = False
    forward_y[:mw+1, :] = True

    central_x[:, :mw+1] = False
    forward_x[:, :mw+1] = True
    
    central_y[ -mw:,:] = False
    backward_y[-mw:,:] = True

    central_x[ :,-mw:] = False
    backward_x[:,-mw:] = True

    #Look for points that are outside the region
    for y in range(mw, mask.shape[0]-mw):
        for x in range(mw, mask.shape[1]-mw):          
            if np.any(mask[y:y+mw+1, x]):
                central_y[y,x] = False
                backward_y[y,x] = True
            if np.any(mask[y-mw:y+1, x]):
                central_y[y,x]= False
                forward_y[y,x] = True

            if np.any(mask[y,x:x+mw+1]):
                central_x[y,x] = False
                backward_x[y,x]= True
            if np.any(mask[y,x-mw:x+1]):
                central_x[y,x] = False
                forward_x[y,x] = True

    #Make sure that these are all False wherever the population is missing
    central_x = np.logical_and(central_x, ~mask)
    central_y = np.logical_and(central_y, ~mask)
    forward_x = np.logical_and(forward_x, ~mask)
    forward_y = np.logical_and(forward_y, ~mask)
    backward_x = np.logical_and(backward_x, ~mask)
    backward_y = np.logical_and(backward_y, ~mask)

    return (mask, central_y, central_x, forward_y, forward_x, backward_y, backward_x)

@jit(nopython=True)
def calculate_grad1(wb, grad1_wb, dd, fd_1, masks):
    '''
    Calculate gradient
    wb: [Y, X, 2] occupation fraction of white/black populations
    grad1_wb: empty array of size [2, Y, X, 2] to be filled
    dd:  pixel size
    fd_1: finite difference filters
    masks: series of central, forward, or backward difference masks
    '''
    mask, central_y, central_x, forward_y, forward_x, backward_y, backward_x = masks
    central_1, forward_1, backward_1 = fd_1
    mw = forward_1.shape[0] - 1
    for y in range(wb.shape[0]):
        for x in range(wb.shape[1]):
            if mask[y,x]: continue
            
            #gradient in y direction
            if central_y[y,x]:
                for i in range(wb.shape[2]):
                    grad1_wb[0,y,x,i] = np.sum(central_1 * wb[y-mw:y+mw+1,x,i])
            elif forward_y[y,x]:
                for i in range(wb.shape[2]):
                    grad1_wb[0,y,x,i] = np.sum(forward_1 * wb[y:y+mw+1,x,i])
            elif backward_y[y,x]:                
                for i in range(wb.shape[2]):
                    grad1_wb[0,y,x,i] = np.sum(backward_1 * wb[y-mw:y+1,x,i])

            #gradient in x direction
            if central_x[y,x]:
                for i in range(wb.shape[2]):
                    grad1_wb[1,y,x,i] = np.sum(central_1 * wb[y,x-mw:x+mw+1,i])
            elif forward_x[y,x]:
                for i in range(wb.shape[2]):
                    grad1_wb[1,y,x,i] = np.sum(forward_1 * wb[y,x:x+mw+1,i])
            elif backward_x[y,x]:
                for i in range(wb.shape[2]):
                    grad1_wb[1,y,x,i] = np.sum(backward_1 * wb[y,x-mw:x+1,i])
            
            grad1_wb[:,y,x,:] = grad1_wb[:,y,x,:] / dd
            
@jit(nopython=True)
def calculate_div(J, div_J, dd, fd_1, masks):
    '''
    Calculate divergence of J
    J: [2, Y, X, 2] 
    div_J: empty array of size [Y, X, 2] to be filled
    dd:  pixel size
    fd_1: finite difference filters
    masks: series of central, forward, or backward difference masks
    '''
    mask, central_y, central_x, forward_y, forward_x, backward_y, backward_x = masks
    central_1, forward_1, backward_1 = fd_1
    mw = forward_1.shape[0] - 1

    for y in range(J.shape[1]):
        for x in range(J.shape[2]):
            if mask[y,x]: continue
            if central_y[y,x]:
                for i in range(J.shape[3]):
                    div_J[y,x,i] = np.sum(central_1 * J[0,y-mw:y+mw+1,x,i])
            elif forward_y[y,x]:
                for i in range(J.shape[3]):
                    div_J[y,x,i] = np.sum(forward_1 * J[0,y:y+mw+1,x,i])
            elif backward_y[y,x]:                
                for i in range(J.shape[3]):
                    div_J[y,x,i] = np.sum(backward_1 * J[0,y-mw:y+1,x,i])

            #gradient in x direction
            if central_x[y,x]:
                for i in range(J.shape[3]):
                    div_J[y,x,i] += np.sum(central_1 * J[1,y,x-mw:x+mw+1,i])
            elif forward_x[y,x]:
                for i in range(J.shape[3]):
                    div_J[y,x,i] += np.sum(forward_1 * J[1,y,x:x+mw+1,i])
            elif backward_x[y,x]:
                for i in range(J.shape[3]):
                    div_J[y,x,i] += np.sum(backward_1 * J[1,y,x-mw:x+1,i])
            
            div_J[y,x,:] = div_J[y,x,:] / dd

@jit(nopython=True)
def calculate_lapl(wb, lapl_wb, dd, fd_2, masks):
    '''
    Calculate laplacian of wb
    wb: [Y, X, 2] 
    lapl_wb: empty array of size [Y, X, 2] to be filled
    dd:  pixel size
    fd_2: finite difference filters of order 2
    masks: series of central, forward, or backward difference masks
    '''
    mask, central_y, central_x, forward_y, forward_x, backward_y, backward_x = masks
    central_2, forward_2, backward_2 = fd_2
    mw = forward_2.shape[0] - 1

    for y in range(wb.shape[0]):
        for x in range(wb.shape[1]):
            if mask[y,x]: continue
            
            #gradient in y direction
            if central_y[y,x]:
                for i in range(wb.shape[2]):
                    lapl_wb[y,x,i] = np.sum(central_2 * wb[y-mw:y+mw+1,x,i])
            elif forward_y[y,x]:
                for i in range(wb.shape[2]):
                    lapl_wb[y,x,i] = np.sum(forward_2 * wb[y:y+mw+1,x,i])
            elif backward_y[y,x]:                
                for i in range(wb.shape[2]):
                    lapl_wb[y,x,i] = np.sum(backward_2 * wb[y-mw:y+1,x,i])

            #gradient in x direction
            if central_x[y,x]:
                for i in range(wb.shape[2]):
                    lapl_wb[y,x,i] += np.sum(central_2 * wb[y,x-mw:x+mw+1,i])
            elif forward_x[y,x]:
                for i in range(wb.shape[2]):
                    lapl_wb[y,x,i] += np.sum(forward_2 * wb[y,x:x+mw+1,i])
            elif backward_x[y,x]:
                for i in range(wb.shape[2]):
                    lapl_wb[y,x,i] += np.sum(backward_2 * wb[y,x-mw:x+1,i])
            
            lapl_wb[y,x,:] = lapl_wb[y,x,:] / dd**2
    
@jit(nopython=True)
def calculate_grad3(wb, grad3_wb, dd, fd_1, fd_2, masks):
    '''
    Calculate grad^3 of wb
    wb: [Y, X, 2] 
    grad3_wb: empty array of size [2, Y, X, 2] to be filled
    dd:  pixel size
    fd_1: finite difference filters (order 1)
    fd_2: finite difference filters (order 2)
    masks: series of central, forward, or backward difference masks
    '''
    lapl_wb = np.zeros_like(wb)
    calculate_lapl(wb, lapl_wb, dd, fd_2, masks)
    calculate_grad1(lapl_wb, grad3_wb, dd, fd_1, masks)
                
@jit(nopython=True)
def calculate_Dij(wb, cij, Dij, mask):
    for i in range(Dij.shape[0]):
        for y in range(wb.shape[0]):
            for x in range(wb.shape[1]):
                if mask[y,x]: continue
                
                Dij[i,y,x]  = cij[i,0] 
                Dij[i,y,x] += cij[i,1] * wb[y,x,0] + cij[i,2] * wb[y,x,1]
                Dij[i,y,x] += cij[i,3] * wb[y,x,0]**2 + cij[i,4] * wb[y,x,1]**2
                Dij[i,y,x] += cij[i,5] * wb[y,x,0] * wb[y,x,1]