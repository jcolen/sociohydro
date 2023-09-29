# Physical Bottleneck approach to Sociohydrodynamics problem

This repository contains a few notebooks and supporting files for experimenting with the sociohydrodynamics dataset. 
Specifically, we use Matthew Schmitt's physical bottleneck method to learn to predict time derivatives of US census data subject to physical hypotheses.
Dynamics are implemented using Fenics/Dolfin and the neural network is trained using dolfin-adjoint. 
This can be run inside the Anaconda environment in `/project/vitelli/ml_venv`

The notebooks contained in this repo are:
- Fenics_data.ipynb - The one where all of the example Fenics implementations are for running on Census data. Does not require any NNs to use.
- physical_bottleneck.ipynb - The big one, where I test recently trained PBNN networks and evaluate them using their built-in "simulate" method
- IntegrablePBNN.ipynb - Testing whether a PBNN subject to no physical hypotheses could predict $\partial_t \phi_i$ well enough to be integrated using a Fenics solver. Here $\partial_t \phi_i = NN(\phi_i)$
