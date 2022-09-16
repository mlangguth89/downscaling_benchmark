#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-out.%j
#SBATCH --error=train-err.%j
#SBATCH --time=02:00:00
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
source ../venv_booster/bin/activate

train_data=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/preproc_era5_crea6_train.nc
test_data=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/preproc_era5_crea6_val.nc
save_dir=../results/exp_test
epochs=30
#model_type=vitS
#model_type=swinSR
model_type=unet
python ../main_scripts/main_train_test_temp.py --train_data ${train_data} --test_data ${test_data} --save_dir ${save_dir} --epochs ${epochs} --model_type ${model_type} > output_l1.test

