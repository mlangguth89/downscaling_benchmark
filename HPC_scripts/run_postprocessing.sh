#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=postprocessing-out.%j
#SBATCH --error=postprocessing-err.%j
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

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

# data-directories
# Note template uses Tier2-dataset. Adapt accordingly for other datasets.
datadir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessedi_era5_crea6/t_2m/all_files/
model_basedir=../results/exp_20230310_swinIR_booster
outdir=../exp_20230310_swinIR_booster/postprocessing
model_name=117580_G.pth
exp_name=swinIR
dataset=downscaling_tier2_test

# run job
python ../main_scripts/main_postprocessing.py -data_dir ${datadir} -model_base_dir ${model_basedir} \
                                                                    -exp_name ${exp_name} -dataset ${dataset} -model_name ${model_name} \
                                                                    -output_base_dir ${outdir}

