#!/bin/bash
#
#SBATCH --job-name=dynamic_PBNN
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/dynamic_PBNN_%A.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/dynamic_PBNN_%A.out
#SBATCH --partition=vitelli-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294
#SBATCH --ntasks=1
#SBATCH --mem=4G

echo "Starting"

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /home/jcolen/sociohydro/physical_bottleneck
python train_PBNN.py --modeltype DiagonalOnlyPBNN --dynamic
