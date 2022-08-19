#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=pre_rad_ifs_out.%j
#SBATCH --error=pre_rad_ifs_err.%j
#SBATCH --time=06:00:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.ji@fz-juelich.de
##jutil env activate -p cdeepacf

### Batch script for performing tests of script develoments ###

# Enter the name of virtual environment here!
VIRT_ENV_NAME="venv_hdfml"

# load necessary modules
source ../env_setup/modules.sh

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

# Run data extraction
srun python preprocess_radklim_ifs.py --years 2020

