#!/bin/bash
#SBATCH --job-name=$1_$2$3
#SBATCH --output=/scratch/midway3/dsseara/$1_$2$3.out
#SBATCH --account=pi-vitelli
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --mem=8G

module load python/anaconda-2022.05
source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda deactivate
conda deactivate
conda activate /project/vitelli/dsseara/anaconda/fipy+pytorch

cd /project/vitelli/dsseara/schelling/jcolen/sociohydro/param_inference

datafolder=/project/vitelli/dsseara/schelling/data/yearly/raw/meshed
savefolder=models_census
sigma=10.0
county=$1
alpha=$2
beta=$3

python train_census_nn.py --datafolder $datafolder --savefolder $savefolder --sigma $sigma --county $county --alpha $alpha --beta $beta