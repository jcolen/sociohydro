#!/bin/bash
#
#SBATCH --job-name=yearly_PBNN
#SBATCH --output=/home/jcolen/scratch/%x_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-1
#SBATCH --ntasks=2
#SBATCH --mem=4G

cd /home/jcolen/sociohydro
module load container_env pytorch-gpu/2.1.0

# Check python
crun -p /home/jcolen/envs/sociohydro python -c "print('Hello world')"

housing="constant"

echo "STARTING" $housing $model ${SLURM_ARRAY_TASK_ID}

crun -p /home/jcolen/envs/sociohydro python train_simulation_PBNN.py \
    --n_epochs 200 \
    --savename models/simulation/SimulationPBNN_${SLURM_ARRAY_TASK_ID}

