## activating `fipy` environment

`conda activate /project/vitelli/dsseara/anaconda/fipy/`

## working with `fipy` variables

[`fipy`](https://pages.nist.gov/fipy/en/latest/index.html) is a Finite Volume Method solver, that discretizes space in cells to numerically solve PDEs.

Given a `fipy` variable `var`, note the following useful methods:
- `var.value` returns a numpy array of numerical values for the field inside each cell
- `var.mesh` gives access to the mesh object
    - `var.mesh.cellCenters.value` gives access to the (x, y) coordinates of the center of each cell
 
If you want to plot a scalar field given by the fipy variable `var`, use the helper function in `plot_mesh` found in `fvm_utils` in this directory. For example:
```python
fig, ax = plt.subplots()
plot_mesh(var, var.mesh, ax, cmap=plt.cm.Reds, vmin=0, vmax=1)
```