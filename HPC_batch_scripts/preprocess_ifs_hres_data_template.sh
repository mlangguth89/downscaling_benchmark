#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=12
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=preprocess_ifs_hres_data-out.%j
#SBATCH --error=preprocess_ifs_hres_data-err.%j
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.langguth@fz-juelich.de
##jutil env activate -p cjjsc42

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment
VIRT_ENV_NAME="venv_juwels"

# Activate virtual environment if needed (and possible)
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ../virtual_envs/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate
   else 
      echo "ERROR: Cannot find requested virtual environment ${VIRT_ENV_NAME}..."
      exit 1
   fi
fi
# Loading mouldes
source ../env_setup/modules.sh

# set variables to be parsed 
src_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ifs_hres/orig/
out_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ifs_hres/preprocessed/
grid_des_tar=../grid_des/ifs_hres_grid_tar

years=( 2016 2017 2018 2019 2020 )
months=( 4 5 6 7 8 9 )
method=Unet_Tier1

srun python -m mpi4py ../main_scripts/main_preprocessing.py -src_dir ${src_dir} -out_dir ${out_dir} -grid_des_tar ${grid_des_tar} \
                                                            -y "${years[@]}" -m "${months[@]}" -method ${method}


