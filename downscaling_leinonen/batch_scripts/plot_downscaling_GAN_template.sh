#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=train_downscaling_GAN-out.%j
#SBATCH --error=train_downscaling_GAN-err.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.langguth@fz-juelich.de

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# set some paths and variables
DATE_NOW=$(date +%Y%m%dT%H%M%S)
WORK_DIR=`pwd`
BASE_DIR=$(dirname "$WORK_DIR")
# Name of virtual environment
VIRT_ENV_NAME="${WORK_DIR}/virtual_envs/<my_venv>"
# Name of container image (must be available in working directory)
CONTAINER_IMG="${WORK_DIR}/tensorflow_21.09-tf1-py3.sif"

# simple sanity checks
if ! [[ -f ${VIRT_ENV_NAME}/bin/activate ]]; then
  echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
  exit 1
fi

if ! [[ -f ${CONTAINER_IMG} ]]; then
  echo "ERROR: Required singularity containr ${CONTAINER_IMG} not found..."
  exit 1
fi

# clean-up modules to avoid conflicts between host and container settings
module purge

# variables for settings
data_dir="<path_to_data>"
mchrzc_file="mchrzc_samples-2018-128x128.nc"
goes_file="goes_samples-2019-128x128.nc"

# Please uncomment the following CUDA configuration
export CUDA_VISIBLE_DEVICES=1

# run training
srun --mpi=pspmix --cpu-bind=none \
singularity exec --nv ${CONTAINER_IMG} ./wrapper_container.sh ${VIRT_ENV_NAME}  python3 ${BASE_DIR}/dsrnngan/main.py plot \
                                      --mchrzc_data_file=${data_dir}/${mchrzc_file} --goescod_data_file=${data_dir}/${goes_file}
