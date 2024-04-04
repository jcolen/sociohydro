#!/bin/bash
#
#SBATCH --job-name=all_counties_PBNN
#SBATCH --output=/home/jcolen/scratch/%x_%A.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --mem=8G

cd /home/jcolen/sociohydro/decennial
module load container_env pytorch-gpu/2.1.0

# Check python
crun -p /home/jcolen/envs/sociohydro python -c "print('Hello world')"

model_id="${SLURM_ARRAY_JOB_ID}_all_counties"
echo "STARTING TO TRAIN" ${model_id}

crun -p /home/jcolen/envs/sociohydro python train_decennial_PBNN.py \
    --use_all_counties \
    --model_id ${model_id}

echo "STARTING TO COMPUTE SALIENCY" ${model_id}

crun -p /home/jcolen/envs/sociohydro python compute_decennial_saliency.py \
    --model_id ${model_id}