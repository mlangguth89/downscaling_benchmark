#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=val-diffusion-out.%j
#SBATCH --error=val-diffusion-err.%j
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.ji@fz-juelich.de

#module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision
#ml SciPy-bundle/2021.10
source ../virtual_envs/env_booster/bin/activate
stat_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/preprocessing/preprocessed_ifs_radklim_full_disk/train

#srun --overlap python sr.py -p val -c config/sr_sr3_16_128.json
srun --overlap python sr.py -p val -stat_dir ${stat_dir} -c config/sr_sr3_prep.json 
