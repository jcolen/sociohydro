#!/bin/bash
#
#SBATCH --job-name=saliency_PBNN
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/saliency_PBNN_%A_%a.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/saliency_PBNN_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
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

logname="/scratch/midway3/jcolen/sbatch_outputs/saliency_PBNN_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

for dir in yearly/pipeline_10012023; do
    if (( $i == $SLURM_ARRAY_TASK_ID )); then
		echo "STARTING" $dir
		ln --symbolic ${logname}  ${dir}/saliency_log.out
		python get_yearly_saliency.py ${dir}
	fi
	i=$(($i+1))
done
