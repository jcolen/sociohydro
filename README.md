# Physical Bottleneck approach to Sociohydrodynamics problem

This repository contains a few notebooks and supporting files for experimenting with the sociohydrodynamics dataset. 
Specifically, we use Matthew Schmitt's physical bottleneck method to learn to predict time derivatives of US census data subject to physical hypotheses.
Dynamics are implemented using Fenics/Dolfin and the neural network is trained using dolfin-adjoint. 
This can be run inside the Anaconda environment in `/project/vitelli/ml_venv`

The notebooks contained in this repo are:
- DecennialPBNN.ipynb - The big one, where I test recently trained PBNN networks and evaluate them using their built-in "simulate" method, and also compute their saliency maps
- DecennialSaliency.ipynb - Evaluating saliency plots for the Decennial PBNN data
- YearlyPBNN.ipynb - Same as PBNN.ipynb but evaluating on yearly data
- YearlySaliency.ipynb - Evaluating saliency plots for the Yearly PBNN data
- MakeDilatedMeshes.ipynb - mesh making for the yearly saliency models
