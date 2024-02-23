#!/bin/bash
#
#SBATCH --job-name=pipeline_PBNN
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/pipeline_PBNN_%A_%a.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/pipeline_PBNN_%A_%a.out
#SBATCH --partition=vitelli-gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-3
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294,midway3-[0277-0281]
#SBATCH --ntasks=1
#SBATCH --mem=24G

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /home/jcolen/sociohydro

housing="constant"
model="SourcedOnlyPBNN"
i=1

savedir="yearly/pipeline_${SLURM_ARRAY_JOB_ID}"
mkdir -p ${savedir}

echo "STARTING" $housing $model ${SLURM_ARRAY_TASK_ID}

ln --symbolic /scratch/midway3/jcolen/sbatch_outputs/pipeline_PBNN_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ${savedir}/slurm_log_${SLURM_ARRAY_TASK_ID}.out

python train_yearly_PBNN.py \
    --modeltype ${model} \
    --housing ${housing} \
    --n_epochs 200 \
    --savename ${savedir}/${model}_${SLURM_ARRAY_TASK_ID}

