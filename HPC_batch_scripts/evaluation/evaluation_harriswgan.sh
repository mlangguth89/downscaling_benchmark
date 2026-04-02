#!/bin/bash -x
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=%x-out.%j
#SBATCH --error=%x-err.%j
#SBATCH --time=02:00:00
##SBATCH --gres=gpu:1
#SBATCH --partition=batch
##SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

# environmental variables to support cpus_per_task with Slurm>22.05
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
# basic directories
WORK_DIR=$(pwd)
BASE_DIR=$(dirname $(dirname "${WORK_DIR}"))
###############################################################################
###############################################################################
### PLEASE ADAPT THE FOLLOWING LINES FOR YOUR VENV AND PER VARIABLE AND PATHS

# Name of virtual environment 
VENV_DIR=${BASE_DIR}/virtual_envs/
VIRT_ENV_NAME=VENV_NAME  # PLEASE ADAPT

## model and variable specifics
var_name=t2m
# var_name=ws100m
# var_name=glob_rad
model_name="Harris WGAN"
model_name_file="harris_wgan"
if [[ "${var_name}" == "ws100m" ]]; then
   exp_name=harris_wgan_benchmark_cs256_4gpus_layernorm_recon100_added_norm_det
elif [[ "${var_name}" == "glob_rad" ]]; then
   exp_name=harris_wgan_benchmark_cs256_4gpus_layernorm_recon100_added_norm_det
else
   exp_name=harris_wgan_benchmark_cs256_4gpus_layernorm_added_norm_det
fi

# data-directories
### output directory for evaluation results (metrics, plots, etc.)
outdir=/path/to/evaluation_directory
### resultsdir is where the downscaled netcdf files are located from inference
resultsdir=/path/to/results_directory

### DO NOT ADAPT BELOW UNLESS YOU KNOW WHAT YOU ARE DOING
###############################################################################
###############################################################################
###############################################################################

# environmental variables to support cpus_per_task with Slurm>22.05
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# Loading modules
source ${BASE_DIR}/env_setup/modules_jsc_evaluation.sh
# Activate virtual environment if needed (and possible)
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ${VENV_DIR}/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ${VENV_DIR}/${VIRT_ENV_NAME}/bin/activate
   else
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi

## further variables, no need to adapt
config_evaluation=${BASE_DIR}/config/evaluation/config_evaluation_$var_name.json
## results file
results_nc=${resultsdir}/downscaled_$var_name\_$model_name_file.nc

srun --overlap python3 ${BASE_DIR}/main_scripts/main_evaluation.py \
    --output_base_directory ${outdir} \
    --configuration_evaluation ${config_evaluation} \
    -exp_name ${exp_name} provided_results \
    --results_netcdf ${results_nc} \
    --model_name "${model_name}"
