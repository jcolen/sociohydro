#!/bin/bash
#
#SBATCH --job-name=decennial_PBNN
#SBATCH --output=/home/jcolen/scratch/%x_%A.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --mem=4G


housing="constant"
county="cook_IL fulton_GA harris_TX la_CA"
model="DecennialPBNN"

echo "STARTING" $housing $county $model

cd /home/jcolen/sociohydro/
module load container_env pytorch-gpu/2.1.0

# Check python
crun -p /home/jcolen/envs/sociohydro python -c "print('Hello world')"

crun -p /home/jcolen/envs/sociohydro python train_decennial_PBNN.py \
    --modeltype ${model} \
    --county ${county} \
    --housing ${housing} \
    --n_epochs 200 \
    --savename validation/decennial/DecennialPBNN