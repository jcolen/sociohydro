# Sociohydrodynamics: data-driven modelling of social behavior

This repository contains the code for training and analyzing machine learning models in [ArXiv:2312.17627](https://arxiv.org/abs/2312.17627). The **census_gridded** folder contains models trained using US Census data from 1980-2020 as well as saliency analyses of these models. This is the data to be used in the revision of the paper. 

The **nonlinear_simulations** folder contains models trained using coarse-grained data from simulations of an agent-based model [(Danny's github here)](). 
This is used to validate the methodology and demonstrate how network saliency can reflect utility functions. 

## Legacy code

In addition to the above two folders, we include two additional folders for legacy purposes which will be removed in a future release. These formed the basis of preliminary investigations but the data should not be considered final.

The **census_mesh** folder contains models trained using US Census data from 1980-2020. Due to the irregular geometries of each county, we adapted
the physical bottleneck method introduced in [this paper](https://www.cell.com/cell/fulltext/S0092-8674(23)01331-4) [(Github)](https://github.com/schmittms/physical_bottleneck).
The neural networks predict the time derivatives of US census data and these are used to evolve the system in time using Fenics/Dolfin. The network itself is trained using dolfin-adjoint. This was used during the first draft of the paper, but further analysis during the revision stage revealed some concerns related to the meshing and Fenics pipelines that limited our ability to analyze the network saliencies.

The **linear_simulations** folder contains models trained using coarse-grained data from simulations of agents with *linear* utility functions. This was used to validate the methodology, but we eventually moved to evaluating nonlinear utility functions.
