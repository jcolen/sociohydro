# Sociohydrodynamics: data-driven modelling of social behavior

This repository contains the code for training and analyzing machine learning models in [ArXiv:2312.17627](https://arxiv.org/abs/2312.17627). 
The **decennial** folder contains models trained using US Census data from 1980-2020. Due to the irregular geometries of each county, we adapted
the physical bottleneck method introduced in [this paper](https://www.cell.com/cell/fulltext/S0092-8674(23)01331-4) [(Github)](https://github.com/schmittms/physical_bottleneck).
The neural networks predict the time derivatives of US census data and these are used to evolve the system in time using Fenics/Dolfin. The network itself is trained using dolfin-adjoint.

The **simulations** folder contains models trained using coarse-grained data from simulations of an agent-based model [(Danny's github here)](). 
This is used to validate the methodology and demonstrate how network saliency can reflect utility functions. 

### TODO: Old README that I need to move into the decennial folder
The notebooks contained in this repo are:
- DecennialPBNN.ipynb - The big one, where I test recently trained PBNN networks and evaluate them using their built-in "simulate" method, and also compute their saliency maps
- DecennialSaliency.ipynb - Evaluating saliency plots for the Decennial PBNN data
- YearlyPBNN.ipynb - Same as PBNN.ipynb but evaluating on yearly data
- YearlySaliency.ipynb - Evaluating saliency plots for the Yearly PBNN data
- MakeDilatedMeshes.ipynb - mesh making for the yearly saliency models
