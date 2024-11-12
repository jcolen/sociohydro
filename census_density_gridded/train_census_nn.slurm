#!/bin/bash
#
#SBATCH --job-name=all_counties_census
#SBATCH --output=/home/jcolen/scratch/%x_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --ntasks=2
#SBATCH --mem=8G

cd /home/jcolen/sociohydro/census_density_gridded
module load container_env pytorch-gpu/2.1.0

# Check python
crun -p /home/jcolen/envs/sociohydro python -c "print('Hello world')"

model_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_all_counties"
echo "STARTING TO TRAIN" ${model_id}

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    args=""
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
    args="--use_max_scaling"
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then
    args="--use_fill_frac"
fi

crun -p /home/jcolen/envs/sociohydro python train_census_nn.py --model_id ${model_id} ${args}

echo "STARTING TO COMPUTE SALIENCY" ${model_id}

crun -p /home/jcolen/envs/sociohydro python compute_census_saliency.py --model_id ${model_id} ${args}
