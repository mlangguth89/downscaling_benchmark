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
moudle load Cartopy/0.20.0


source ../venv_booster/bin/activate




# data-directories
# Note template uses Tier2-dataset. Adapt accordingly for other datasets.
datadir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/downscaling_tier2_train.nc
model_basedir=../model_base_dir/
outdir=../output/
model_name=generator_step735000.pth
exp_name=wgan
dataset=downscaling_tier2_train

# run job
python ../main_scripts/main_postprocessing.py -data_dir ${datadir} -model_base_dir ${model_basedir} \
                                                                    -exp_name ${exp_name} -dataset ${dataset} -model_name ${model_name} \
                                                                    -output_base_directory ${outdir}

