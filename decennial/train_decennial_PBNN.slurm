#!/bin/bash
#
#SBATCH --job-name=decennial_PBNN
#SBATCH --output=/home/jcolen/scratch/%x_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --ntasks=2
#SBATCH --mem=8G

cd /home/jcolen/sociohydro/decennial
module load container_env pytorch-gpu/2.1.0

# Check python
crun -p /home/jcolen/envs/sociohydro python -c "print('Hello world')"

model_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "STARTING TO TRAIN" ${model_id}

crun -p /home/jcolen/envs/sociohydro python train_decennial_PBNN.py \
    --n_epochs 100 \
    --model_id ${model_id} \
    --val_tmax 20 \
    --num_train_counties 4 \
    --random_seed ${SLURM_ARRAY_TASK_ID}

echo "STARTING TO COMPUTE SALIENCY" ${model_id}

crun -p /home/jcolen/envs/sociohydro python compute_decennial_saliency.py \
    --model_id ${model_id}