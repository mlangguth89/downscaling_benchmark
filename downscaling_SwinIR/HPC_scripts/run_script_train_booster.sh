#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-out.%j
#SBATCH --error=train-err.%j
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maximbr@post.bgu.ac.il

#source ../venv_booster/bin/activate

module purge
module load Stages/2022 GCCcore/.11.2.0 GCC/11.2.0
ml OpenMPI 4.1.2
ml netCDF/4.8.1
module load dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision
module load xarray/0.20.1
module load Cartopy/0.20.0
module load matplotlib/3.4.3


source ../venv_booster/bin/activate

train_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/downscaling_tier2_train.nc
test_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/downscaling_tier2_val.nc
epochs=3
save_dir=../saves/unet
checkpoint_dir=../results/exp_test/unet

python ../main_scripts/main_train.py --train_dir ${train_dir} --test_dir ${test_dir} --epochs ${epochs} --save_dir ${save_dir}
