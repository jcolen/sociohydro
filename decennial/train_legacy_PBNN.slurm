#!/bin/bash
#
#SBATCH --job-name=decennial_PBNN
#SBATCH --output=/home/jcolen/scratch/%x_%A.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --mem=4G

echo "STARTING" $housing $county $model

cd /home/jcolen/sociohydro/decennial
module load container_env pytorch-gpu/2.1.0

# Check python
crun -p /home/jcolen/envs/sociohydro python -c "print('Hello world')"

crun -p /home/jcolen/envs/sociohydro python train_decennial_PBNN.py \
    --n_epochs 200 \
    --model_id legacy