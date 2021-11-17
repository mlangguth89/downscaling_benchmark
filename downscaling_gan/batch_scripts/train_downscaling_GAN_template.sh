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

# set some paths
WORK_DIR=`pwd`
BASE_DIR=$(dirname "$WORK_DIR")
# Name of virtual environment
VIRT_ENV_NAME="/${WORK_DIR}/<my_venv>/<my_venv>"
# Name of container image (must be available in working directory)
CONTAINER_IMG="${WORK_DIR}/tensorflow_21.09-tf1-py3.sif"

DATE_NOW=$(date +%Y%m%dT%H%M%S)
# clean-up modules to avoid conflicts between host and container settings
module purge

# variables for settings
application="mchrzc"
data_in="<path_to_samples-2018-128x128.nc>"
dest_file_wgt="<path_to_store_weights>"
dest_file_log="<path_to_store_weights>/logging_${DATE_NOW}"

# Please uncomment the following CUDA configuration
export CUDA_VISIBLE_DEVICES=1

# run training
srun --mpi=pspmix --cpu-bind=none \
singularity exec --nv ${CONTAINER_IMG} ./wrapper_container.sh ${VIRT_ENV_NAME}  python3 ../dsrnngan/main_train.py \
                                      --application=${application} --data_file=${data_in} \
                                      --save_weights_root=${dest_file_wgt} --log_path ${dest_file_log} 
