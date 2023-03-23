#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=inference_simple_ap5-out.%j
#SBATCH --error=inference_simple_ap5-err.%j
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
##SBATCH --partition=develbooster
##SBATCH --partition=batch
#SBATCH --mail-type=ALL

# runscript directories
WORK_DIR=$(pwd)
BASE_DIR=$(dirname "${WORK_DIR}")

ml --force purge
ml use $OTHERSTAGES
ml Stages/2022

ml GCCcore/.11.2.0
ml GCC/11.2.0
ml ParaStationMPI/5.5.0-1
ml netcdf4-python/1.5.7-serial
ml SciPy-bundle/2021.10
ml xarray/0.20.1
ml dask/2021.9.1
ml TensorFlow/2.6.0-CUDA-11.5

# parsing arguments -> PLEASE ADAPT TO YOUR DIRECTORY STRUCTURE
model_dir=${BASE_DIR}/trained_models/unet_test
data_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_tier2/  

srun --overlap python3 ./inference_simple.py -data_dir ${data_dir} -model_dir ${model_dir}
