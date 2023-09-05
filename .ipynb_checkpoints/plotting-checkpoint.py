import matplotlib
import matplotlib.pyplot as plt
import numpy as np

lw = 1
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['axes.linewidth'] = lw
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = lw
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.width'] = lw
plt.rcParams['font.size'] = 6
plt.rcParams['legend.framealpha'] = 0.
plt.rcParams['legend.handlelength'] = 1.25
plt.rcParams['image.origin'] = 'upper'
plt.rcParams['pcolor.shading'] = 'auto'
plt.rcParams['figure.dpi'] = 250

def plot_white_black_Dij(fig, ax, w_grid, b_grid, D_grid, year, x_grid, y_grid, vmax=2, title=False):
    
    c0 = ax[0].pcolormesh(x_grid, y_grid, w_grid, cmap="Blues", vmin=0, vmax=1)
    cax0 = ax[0].inset_axes([1.05, 0, 0.05, 1])
    fig.colorbar(c0, ax=ax[0], cax=cax0)

    c1 = ax[1].pcolormesh(x_grid, y_grid, b_grid, cmap="Reds", vmin=0, vmax=1)
    cax1 = ax[1].inset_axes([1.05, 0, 0.05, 1])
    fig.colorbar(c1, ax=ax[1], cax=cax1)

    if title:
        ax[0].set_title(f'$\\phi_W$')
        ax[1].set_title(f'$\\phi_B$')
        
    toplot = {
        '$D_{WW}$': np.exp(D_grid[..., 0]),
        '$D_{BB}$': np.exp(D_grid[..., 3]),
        '$D_+$': 0.5 * (D_grid[..., 1] + D_grid[..., 2]),
        '$D_-$': 0.5 * (D_grid[..., 1] - D_grid[..., 2]),
    }
    
    for i, key in enumerate(toplot.keys()):
        ci = ax[i+2].pcolormesh(x_grid, y_grid, toplot[key],
                                cmap='PiYG', vmin=-vmax, vmax=vmax)
        if title:
            ax[i+2].set_title(key)
        caxi = ax[i+2].inset_axes([1.05, 0, 0.05, 1])
        fig.colorbar(ci, ax=ax[i+2], cax=caxi)
    
    for a in ax.ravel():
        a.set_aspect(1)
        a.set(xticks=[], yticks=[])
        
def plot_white_black_seg(fig, ax, w_grid, b_grid, year, x_grid, y_grid, 
                         wmax=None, bmax=None, smax=None, max_grid=None):
    fac = 1 if max_grid is None else max_grid
    c0 = ax[0].pcolormesh(x_grid, y_grid, w_grid * fac, cmap="Blues", vmin=0, vmax=wmax)
    cax0 = ax[0].inset_axes([1.05, 0, 0.05, 1])
    fig.colorbar(c0, ax=ax[0], cax=cax0)

    c1 = ax[1].pcolormesh(x_grid, y_grid, b_grid * fac, cmap="Reds", vmin=0, vmax=bmax)
    cax1 = ax[1].inset_axes([1.05, 0, 0.05, 1])
    fig.colorbar(c1, ax=ax[1], cax=cax1)

    pop = np.array([(w_grid*fac).ravel(), (b_grid*fac).ravel()]).T
    h, H = entropy_index(pop)
    h_grid = h.reshape(x_grid.shape)
    
    c2 = ax[2].pcolormesh(x_grid, y_grid, h_grid, cmap="viridis", vmin=0, vmax=smax)
    cax2 = ax[2].inset_axes([1.05, 0, 0.05, 1])
    fig.colorbar(c2, ax=ax[2], cax=cax2)
    
    for a in ax.ravel():
        a.set_aspect(1)
        a.ticklabel_format(axis='both', scilimits=(0,0))
        a.set_xlabel('$x$ ($10^5$ m)')
        
    ax[0].set_ylabel('$y$ ($10^5$ m)')
    #ax[0].set_yticks([4.8e5, 5.0e5, 5.2e5, 5.4e5])

    ax[0].set_title(f'White population in {year}')
    ax[1].set_title(f'Black population in {year}')
    ax[2].set_title(f'Segregation in {year}')
    
from scipy import stats
def entropy_index(pop):
    '''
    Measures entropy index of population
    
    Inputs
    ------
    pop : 2D array-like
        Each row indexes neighborhood (i.e. census tract, interpolated grid point).
        Each column gives different population type (i.e. black, hispanic, white)

    Outputs
    -------
    h : 2D numpy array
        Spatial entropy index measure in the same format as pop
    H : float
        Integrated entropy index for entire dataset
    '''
    pop = np.asarray(pop)
    
    T = np.nansum(pop)  # total population in region
    ti = np.nansum(pop, axis=1)  # total population at each neighborhood
    tm = np.nansum(pop, axis=0)  # total population of each type in region 
    pim = pop / ti[:, np.newaxis]  # proportion of each type in each neighborhood
    pm = tm / T  # proportion of each type in region
    
    E = stats.entropy(pm)
    norm = T * E

    plnp = stats.entropy(pim, qk=np.broadcast_to(pm, pim.shape), axis=1)
    h = plnp / E
    
    H = np.nansum(ti * h) / T
    
    return h, H


def seg(w, b):
    pop = np.array([w.ravel(), b.ravel()]).T
    h, H = entropy_index(pop)
    h = h.reshape(b.shape)
    return h
