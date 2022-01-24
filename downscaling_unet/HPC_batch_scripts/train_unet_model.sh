#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=train_unet-model-out.%j
#SBATCH --error=train_unet-model-err.%j
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.langguth@fz-juelich.de

# Name of virtual environment
VIRT_ENV_NAME="test"

# Loading mouldes
source ../env_setup/modules_train.sh
# Activate virtual environment if needed (and possible)
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ../virtual_envs/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate
   else
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi

# declare directory-variables which will be modified by config_runscript.py
source_dir=/p/project/deepacf/maelstrom/data/downscaling_unet/
destination_dir=/p/project/deepacf/maelstrom/langguth1/downscaling_jsc_repo/downscaling_unet/trained_models/

srun python3 ../main_scripts/main_train.py -in ${source_dir} -out ${destination_dir} -job_id ${SLURM_JOBID}
