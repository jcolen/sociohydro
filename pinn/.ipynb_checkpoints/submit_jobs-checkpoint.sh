#!/bin/bash

n_points=5000
loaddir="."
models=("DiagonalOnly" "SymmetricCrossDiffusion")
models=("SymmetricCrossDiffusion")

county="cook_IL"
for model in ${models[@]}
do
	echo $model

	cp slurm_python.slurm job.slurm
	sed -i "s/MM/${model}/" job.slurm
	sed -i "s/OPTS/--num_points ${n_points} --loaddir ${loaddir} --county ${county}/" job.slurm
	sbatch job.slurm
done

rm job.slurm
