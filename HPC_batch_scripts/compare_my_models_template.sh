#!/bin/bash -x
#SBATCH --account=hclimrep
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --output=compare_my_models.%j-out
#SBATCH --error=compare_my_models.%j-err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
##SBATCH --mail-type=ALL
##SBATCH --mail-user=XXX@fz-juelich.de

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

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


model_basedir=${BASE_DIR}/results/
outdir=${BASE_DIR}/results/
datasets= <my_daasets>  # comma seperated, for example: wind,t2m
model_types=<my_models>  # comma seperated, for example: deepru,sha_wgan,sha_unet


# run job
# data inference with trained model of this framework
srun --overlap python3 ${BASE_DIR}/postprocess/compareModels.py --datasets ${datasets}  --model-types ${model_types} --output-dir ${outdir}  --model-dir ${model_basedir}

