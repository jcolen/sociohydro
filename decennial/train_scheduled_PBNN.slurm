#!/bin/bash
#
#SBATCH --job-name=scheduled_PBNN
#SBATCH --output=/home/jcolen/scratch/%x_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --ntasks=2
#SBATCH --mem=16G

echo "STARTING" $housing $county $model

cd /home/jcolen/sociohydro/decennial
module load container_env pytorch-gpu/2.1.0

# Check python
crun -p /home/jcolen/envs/sociohydro python -c "print('Hello world')"
echo "STARTING" ${SLURM_ARRAY_TASK_ID}

crun -p /home/jcolen/envs/sociohydro python train_scheduled_PBNN.py \
    --n_epochs 200 \
    --model_id ${SLURM_ARRAY_TASK_ID}