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