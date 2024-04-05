# Sociohydrodynamics: data-driven modelling of social behavior

This repository contains the code for training and analyzing machine learning models in [ArXiv:2312.17627](https://arxiv.org/abs/2312.17627). The **census_gridded** folder contains models trained using US Census data from 1980-2020 as well as saliency analyses of these models. This is the data to be used in the revision of the paper. 

The **nonlinear_simulations** folder contains models trained using coarse-grained data from simulations of an agent-based model [(Github)](https://github.com/dsseara/sociohydro). 
This is used to validate the methodology and demonstrate how network saliency can reflect utility functions. 

## Legacy code

In addition to the above two folders, we include two additional folders for legacy purposes. These formed the basis of preliminary investigations but the data should not be considered final.

The **linear_simulations** folder contains models trained using coarse-grained data from simulations of agents with *linear* utility functions. This was used to validate the methodology, but we eventually moved to evaluating nonlinear utility functions.

The **census_mesh** folder contains models trained using US Census data from 1980-2020. Due to the irregular geometries of each county, we adapted
the physical bottleneck method introduced in [this paper](https://www.cell.com/cell/fulltext/S0092-8674(23)01331-4) [(Github)](https://github.com/schmittms/physical_bottleneck).
The neural networks predict the time derivatives of US census data and these are used to evolve the system in time using Fenics/Dolfin. The network itself is trained using dolfin-adjoint. 
This approach to learning and predicting dynamics was used during the initial draft of the paper.
During the revision stage, we noticed potential issues with the meshing procedure applied to each US county, as well as the process for using the models to predict dynamics (the physical bottleneck method was previously only used to predict static images).
Because these concerns limited our confidence in drawing conclusions about the network saliencies, we changed our machine learning pipeline to that used in census_gridded.
