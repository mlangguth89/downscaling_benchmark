#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=train_downscaling_GAN-out.%j
#SBATCH --error=train_downscaling_GAN-err.%j
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.langguth@fz-juelich.de
##jutil env activate -p cjjsc42

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment 
#VIRT_ENV_NAME="my_venv"

# Loading mouldes
source ../env_setup/modules_train.sh
# Activate virtual environment if needed (and possible)
#if [ -z ${VIRTUAL_ENV} ]; then
#   if [[ -f ../${VIRT_ENV_NAME}/bin/activate ]]; then
#      echo "Activating virtual environment..."
#      source ../${VIRT_ENV_NAME}/bin/activate
#   else 
#      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
#      exit 1
#   fi
#fi

# variables for settings
application="mchrzc"
data_in=""
dest_file_wgt=""
dest_file_log=""

# Please uncomment the following CUDA configuration
#CUDA_VISIBLE_DEVICES=1

# run training
srun python ../dsrnngan/main_train.py --application=${application} --data_file=${data_in} \
                                      --save_weights_root=${dest_file_wgt} --log_path ${dest_file_log} 



 
