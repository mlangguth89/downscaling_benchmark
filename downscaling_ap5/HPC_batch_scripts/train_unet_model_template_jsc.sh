#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=train_unet-model-out.%j
#SBATCH --error=train_unet-model-err.%j
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=batch
##SBATCH --partition=gpus
#SBATCH --partition=develgpus
##SBATCH --partition=booster
##SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXX@fz-juelich.de

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# environmental variables to support cpus_per_task with Slurm>22.05
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# basic directories
WORK_DIR=$(pwd)
BASE_DIR=$(dirname "${WORK_DIR}")

# Name of virtual environment
VENV_DIR=${BASE_DIR}/virtual_envs/
VIRT_ENV_NAME=<my_venv>

# Loading mouldes
source ../env_setup/modules_jsc.sh
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


# data-directories
# Adapt accordingly to your dataset
indir=<training_data_dir>
outdir=${BASE_DIR}/trained_models/
js_model_conf=${BASE_DIR}/config/config_unet.json
js_ds_conf=${BASE_DIR}/config/config_ds_tier2.json

model=unet
dataset=tier2

exp_name=<my_exp>

# run job
srun --overlap python3 ${BASE_DIR}/main_scripts/main_train.py -in ${indir} -out ${outdir} -model ${model} -dataset ${dataset} \
	                                                           -conf_ds ${js_ds_conf} -conf_md ${js_model_conf} -exp_name ${exp_name} -id ${SLURM_JOBID}

