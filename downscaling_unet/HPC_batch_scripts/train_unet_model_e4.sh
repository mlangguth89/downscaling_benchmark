#!/bin/bash -x
#SBATCH --partition=ice-nc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=train_unet-model-out.%j
#SBATCH --error=train_unet-model-err.%j

# auxiliary variables
WORK_DIR="$(pwd)"
BASE_DIR=$(dirname "$WORK_DIR")

# Name of virtual environment
VIRT_ENV_NAME="test"

# Loading mouldes
#source ../env_setup/modules_train.sh
ml slurm
ml nvidia/cuda-11.2

export PYTHONPATH=${BASE_DIR}:$PYTHONPATH 
export PYTHONPATH=${BASE_DIR}/utils:$PYTHONPATH 
export PYTHONPATH=${BASE_DIR}/handle_data:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/models:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/postprocess:$PYTHONPATH
echo ${PYTHONPATH}

# Activate virtual environment if needed (and possible)
#if [ -z ${VIRTUAL_ENV} ]; then
#   if [[ -f ../virtual_envs/${VIRT_ENV_NAME}/bin/activate ]]; then
#      echo "Activating virtual environment..."
#      source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate
#   else
#      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
#      exit 1
#   fi
# fi

# declare directory-variables which will be modified by config_runscript.py
source_dir=/data/maelstrom/langguth1/
destination_dir=/data/maelstrom/langguth1/trained_models/


srun python3 ../main_scripts/main_train.py -in ${source_dir} -out ${destination_dir} -id ${SLURM_JOBID}
