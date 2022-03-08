#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=postprocess-out.%j
#SBATCH --error=postprocess-err.%j
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

# Name of virtual environment
VIRT_ENV_NAME=venv_juwels

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
model_path=/p/home/jusers/gong1/juwels/bing_folder/downscaling_maelstrom/downscaling_unet/HPC_batch_scripts/results/trained_downscaling_unet_t2m_hour12_exp13.h5 
destination_dir=/p/home/jusers/gong1/juwels/bing_folder/downscaling_maelstrom/downscaling_unet/HPC_batch_scripts
opt_norm_path=/p/home/jusers/gong1/juwels/bing_folder/downscaling_maelstrom/downscaling_unet/HPC_batch_scripts/results/opt_norm.json


srun python3 ../main_scripts/main_postprocess.py -in ${source_dir} -out ${destination_dir} -model ${model_path} -opt_norm ${opt_norm_path} -id ${SLURM_JOBID}