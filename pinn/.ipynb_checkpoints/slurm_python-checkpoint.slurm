#!/bin/bash
#
#SBATCH --job-name=run_MM
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/run_MM_%A.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/run_MM_%A.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294,midway3-[0277-0281]
#SBATCH --ntasks=1
#SBATCH --mem=4G

echo "Starting"

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /home/jcolen/sociohydro/pinn
python train_PINN.py --model MM OPTS
