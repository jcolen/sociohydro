#!/bin/bash

#cp slurm_python.slurm job.slurm
#sed -i "s/AAAA/socio_NODE/" job.slurm
#sed -i "s/MM/conv/" job.slurm
#sbatch job.slurm

#cp slurm_python.slurm job.slurm
#sed -i "s/AAAA/socio_NN/" job.slurm
#sed -i "s/MM/lstm/" job.slurm
#sbatch job.slurm

#cp slurm_python.slurm job.slurm
#sed -i "s/AAAA/socio_NN/" job.slurm
#sed -i "s/MM/fcn/" job.slurm
#sbatch job.slurm

models=("Uninformed" "LinearDiffusion" "QuadraticDiffusion" "CubicDiffusion")
n_points=5000
county="harris_TX"
loaddir="."
#models=("LinearDiffusionLinear" "QuadraticDiffusionLinear" "CubicDiffusionLinear" "Sociohydrodynamics")
models=("DiffusionNN_" "DiffusionNN_Linear" "DiffusionNN_FullSocio")
models=("DiffusionNN_" "DiffusionDiagonalNN_")
models=("DiffusionNN_Linear" "DiffusionNN_FullSocio")
n_points=1000
county="cook_IL"
#loaddir="data\/pinn_08252023/"
for model in ${models[@]}
do
	echo $model

	cp slurm_python.slurm job.slurm
	sed -i "s/AAAA/PINN/" job.slurm
	sed -i "s/MM/${model}/" job.slurm
	sed -i "s/OPTS/--num_points ${n_points} --loaddir ${loaddir} --county ${county}/" job.slurm
	sbatch job.slurm
done

rm job.slurm
