#!/bin/bash -x
#SBATCH --account=maelstrom
#SBATCH --partition=i-gpu-a100
##SBATCH --partition=a-gpu-mi100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128Gb
#SBATCH --gres=gpu:1
##SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=train_wgan-model-out.%j
#SBATCH --error=train_wgan-model-err.%j

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# basic directories
WORK_DIR=$(pwd)
BASE_DIR=$(dirname "${WORK_DIR}")

# Name of virtual environment
VENV_DIR=/opt/share/users/maelstrom/
VIRT_ENV_NAME=venv-rocm

# Loading mouldes
module purge
ml slurm 

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

export PYTHONPATH=${BASE_DIR}:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/utils:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/handle_data:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/models:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/postprocess:$PYTHONPATH
echo ${PYTHONPATH}

# data-directories 
# Adapt accordingly for your dataset
indir=<my_input_dir>
outdir=${BASE_DIR}/trained_models/
js_model_conf=${BASE_DIR}/config/config_wgan.json
js_ds_conf=${BASE_DIR}/config/config_ds_tier2.json

model=wgan
dataset=tier2

exp_name=<my_exp>

# run job
srun --overlap python3 ${BASE_DIR}/main_scripts/main_train.py -in ${indir} -out ${outdir} -model ${model} -dataset ${dataset} \
	                                                           -conf_ds ${js_ds_conf} -conf_md ${js_model_conf} -exp_name ${exp_name} -id ${SLURM_JOBID}

