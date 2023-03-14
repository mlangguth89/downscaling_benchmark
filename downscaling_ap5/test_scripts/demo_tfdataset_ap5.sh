#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --output=demo_tfdataset_ap5-out.%j
#SBATCH --error=demo_tfdataset_ap5-err.%j
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
##SBATCH --partition=develbooster
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.langguth@fz-juelich.de

# basic directories
WORK_DIR=$(pwd)
BASE_DIR=$(dirname "${WORK_DIR}")

ml GCCcore/.11.2.0
ml GCC/11.2.0
ml ParaStationMPI/5.5.0-1
ml netcdf4-python/1.5.7-serial
ml SciPy-bundle/2021.10
ml xarray/0.20.1
ml dask/2021.9.1
ml TensorFlow/2.6.0-CUDA-11.5

source ${BASE_DIR}/virtual_envs/venv_hdfml/bin/activate
export MALLOC_MMAP_MAX_=40960
# data-directories

srun --overlap python3 ${BASE_DIR}/jupyter_notebooks_test/demo_tfdataset_ap5.py -lprefetch
