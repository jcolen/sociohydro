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

## the machine learning gizmos

I've added three files and a jupyter notebook implementing the hybrid parameter inference module. 
These follow the structure of the other ML subdirectories within this repo

- `fipy_dataset.py` - defines data loading and transformation
- `fipy_nn.py` - defines the hybrid neural network models for parameter inference
- `train_fipy_nn.py` - defines the training loop
- `HybridInferenceTesting.ipynb` - tests each module and roughly shows how it works
- `HybridInferenceAnalysis.ipynb` - demonstrates the analysis pipeline for trained models

Training a model can be done with the command `python train_fipy_nn.py`. 
The default configuration uses the "sociohydrodynamic" grouped terms and applies them to Fulton County, GA.
The county can be changed with the keyword argument `--county`. 
Model sare saved in the `models` folder of this subdirectory.
For now I've hardcoded the data paths so feel free to rewrite them on whatever machine or file structure you're using.
