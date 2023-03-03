#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-diffusion-out.%j
#SBATCH --error=train-diffusion-err.%j
#SBATCH --time=23:40:00
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

#module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision
#ml SciPy-bundle/2021.10
source ../env_setup/venv_booster/bin/activate 

stat_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/train

srun --overlap python sr.py -p train -stat_dir ${stat_dir} -c config/sr_sr3_prep.json 
