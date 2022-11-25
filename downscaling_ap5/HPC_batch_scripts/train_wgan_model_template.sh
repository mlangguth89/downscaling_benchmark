#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=train_wgan-model-out.%j
#SBATCH --error=train_wgan-model-err.%j
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXX@fz-juelich.de

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
source ../env_setup/modules.sh
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
# Note template uses Tier2-dataset. Adapt accordingly for other datasets.
indir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/
outdir=<my_outdir>
js_model_conf=${WORK_DIR}/config_wgan.json
js_ds_conf=${WORK_DIR}/config_ds_tier2.json

model=wgan
dataset=tier2

exp_name=<my_exp>

# run job
srun --overlap python3 ${BASE_DIR}/main_scripts/main_train.py -in ${indir} -out ${outdir} -model ${model} -dataset ${dataset} \
	                                                           -conf_ds ${js_ds_conf} -conf_md ${js_model_conf} -exp_name ${exp_name} -id ${SLURM_JOBID}

