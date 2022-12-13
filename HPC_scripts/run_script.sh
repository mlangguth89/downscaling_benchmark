#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-out.%j
#SBATCH --error=train-err.%j
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.ji@fz-juelich.de

#source ../venv_booster/bin/activate

#module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision
#ml SciPy-bundle/2021.10
source ../venv_booster/bin/activate



train_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/preprocessing/preprocessed_ifs_radklim_full_disk/train
test_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/preprocessing/preprocessed_ifs_radklim_full_disk/test
save_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/results/Unet_on_the_fly
epochs=60
model_type=swinir

python ../main_scripts/main_train.py --train_dir ${train_dir} --test_dir ${test_dir} --save_dir ${save_dir} --epochs ${epochs} --model_type ${model_type}> output.test #python ../main_scripts/dataset_prep.py
#python ../main_scripts/main_train_precip.py  
