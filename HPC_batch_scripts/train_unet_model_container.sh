#!/bin/bash -x
#SBATCH --partition=casc-hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=train_unet-model-out.%j
#SBATCH --error=train_unet-model-err.%j

# auxiliary variables
WORK_DIR="$(pwd)"
BASE_DIR=$(dirname "$WORK_DIR")

# Name of virtual environment
VIRT_ENV_NAME="test"

CONTAINER_IMG="${BASE_DIR}/env_setup/tensorflow_22.02-tf2-py3.sif"
WRAPPER="${BASE_DIR}/env_setup/wrapper_container.sh"

# sanity checks
if [[ ! -f ${CONTAINER_IMG} ]]; then
  echo "ERROR: Cannot find required TF2.7.0 container image '${CONTAINER_IMG}'."
  exit 1
fi

if [[ ! -f ${WRAPPER} ]]; then
  echo "ERROR: Cannot find wrapper-script '${WRAPPER}' for TF2.7.0 container image."
  exit 1
fi

# purge modules to rely on settings in container
module purge
ml slurm 
ml go-1.17.6/singularity-3.9.5 

# declare directory-variables which will be modified by config_runscript.py
source_dir=/p/project/deepacf/maelstrom/data/downscaling_unet/
destination_dir=/p/project/deepacf/maelstrom/langguth1/downscaling_jsc_repo/downscaling_unet/trained_models/

#srun --mpi=pspmix --cpu-bind=none \
srun --cpu-bind=none \
     singularity exec --nv "${CONTAINER_IMG}" "${WRAPPER}" ${VIRT_ENV_NAME} \
     python3 ../main_scripts/main_train.py -in ${source_dir} -out ${destination_dir} -id ${SLURM_JOBID}
