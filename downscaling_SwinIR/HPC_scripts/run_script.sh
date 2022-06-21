#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-out.%j
#SBATCH --error=train-err.%j
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

#source ../venv_booster/bin/activate

#module purge
module load Stages/2022 GCCcore/.11.2.0
module load PyTorch/1.11-CUDA-11.5
module load torchvision
#ml SciPy-bundle/2021.10
source ../venv_booster/bin/activate

python ../main_scripts/main_train_precip.py  
