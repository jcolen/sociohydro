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

models=("LinearDiffusion" "QuadraticDiffusion" "CubicDiffusion")
models=("LinearDiffusionLinear" "QuadraticDiffusionLinear" "CubicDiffusionLinear" "Sociohydrodynamics")
for model in ${models[@]}
do
	echo $model

	cp slurm_python.slurm job.slurm
	sed -i "s/AAAA/PINN/" job.slurm
	sed -i "s/MM/${model} --num_points 1000/" job.slurm
	sbatch job.slurm
done

rm job.slurm
