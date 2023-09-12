#!/bin/bash
#
#SBATCH --job-name=pipeline_PBNN
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/pipeline_PBNN_%A_%a.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/pipeline_PBNN_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-10
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294
#SBATCH --ntasks=1
#SBATCH --mem=4G

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /home/jcolen/sociohydro/physical_bottleneck

housings=("constant" "varying")
counties=("cook_IL")
models=("DiagonalOnlyPBNN" "SymmetricCrossDiffusionPBNN" "CrossDiffusionPBNN" "SourcedSymmetricPBNN" "SourcedDiffusionPBNN")
i=1

savedir="pipeline_${SLURM_ARRAY_JOB_ID}"
mkdir -p ${savedir}

for housing in "${housings[@]}"; do
    for county in "${counties[@]}"; do
        for model in "${models[@]}"; do 
        if (( $i == $SLURM_ARRAY_TASK_ID )); then
                echo "STARTING" $i $housing $county $model

                ln --symbolic /scratch/midway3/jcolen/sbatch_outputs/pipeline_PBNN_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ${savedir}/slurm_log_${SLURM_ARRAY_TASK_ID}.out

                python train_PBNN.py \
                    --modeltype ${model} \
                    --county ${county} \
                    --housing ${housing} \
                    --n_epochs 200 \
                    --savedir ${savedir}
            fi
            i=$(($i+1))
        done
    done
done
