#!/bin/bash
#
#SBATCH --job-name=pipeline_PBNN
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/pipeline_PBNN_%A_%a.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/pipeline_PBNN_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-6
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294
#SBATCH --ntasks=1
#SBATCH --mem=4G

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /home/jcolen/sociohydro/physical_bottleneck

housings=("constant" "varying")
counties=("cook_IL" "harris_TX" "cook_IL harris_TX")
i=1

for housing in "${housings[@]}"; do
    for county in "${counties[@]}"; do
        if (( $i == $SLURM_ARRAY_TASK_ID )); then
            echo "STARTING" $i $housing $county
            savedir="pipeline_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
            mkdir -p ${savedir}

            ln --symbolic /scratch/midway3/jcolen/sbatch_outputs/pipeline_PBNN_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ${savedir}/slurm_log.out

            python run_PBNN_pipeline.py \
                --county ${county} \
                --housing ${housing} \
                --n_epochs 200 \
                --pretrain_epochs 100 \
                --savedir ${savedir}
        fi
        i=$(($i+1))
    done
done