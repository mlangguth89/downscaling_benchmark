#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-swinIR-out.%j
#SBATCH --error=train-swinIR-err.%j
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

#module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision
module load  netcdf4-python/1.5.7
#ml SciPy-bundle/2021.10
source ../env_setup/venv_booster/bin/activate

train_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/t2m/all_files/downscaling_tier2_train.nc
val_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/t2m/all_files/downscaling_tier2_val.nc
save_dir=../results/exp_20230403_swinIR_temp
# save_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/results/swinUnet_exp1017_origin_booster_3years
#train_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/train
#val_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/val
#save_dir=../results/exp_20230310_unet_booster

epochs=10
dataset_type=temperature
#dataset_type=precipitation
#model_type=vitSR
model_type=swinIR
#model_type=unet
#model_type=swinUnet
srun --overlap python ../main_scripts/main_train.py --dataset_type ${dataset_type} --train_dir ${train_dir} --val_dir ${val_dir} --save_dir ${save_dir} --epochs ${epochs} --model_type ${model_type} 
#srun --overlap python ../main_scripts/dataset_temp.py
