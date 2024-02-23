#!/bin/bash
#
#SBATCH --job-name=pipeline_PBNN
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/%x_%A.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/%x_%A.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294
#SBATCH --ntasks=1
#SBATCH --mem=4G

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /home/jcolen/sociohydro/physical_bottleneck

housing="constant"
county="cook_IL fulton_GA harris_TX la_CA"
model="SourcedOnlyPBNN"

echo "STARTING" $housing $county $model

python train_PBNN.py \
--modeltype ${model} \
--county ${county} \
--housing ${housing} \
--n_epochs 200 \
--savename validation/decennial/SourcedOnlyPBNN