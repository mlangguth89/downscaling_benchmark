#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=preprocess_ifs_radklim_2018-out.%j
#SBATCH --error=preprocess_ifs_radklim_2018-err.%j
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:0
#SBATCH --partition=gpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.langguth@fz-juelich.de
##jutil env activate -p cjjsc42

srun python ../preprocess/preprocess_radklim_ifs.py

