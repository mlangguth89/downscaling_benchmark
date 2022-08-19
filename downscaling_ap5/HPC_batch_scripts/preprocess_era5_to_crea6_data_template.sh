#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=12
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=preprocess_era5_to_crea6_data-out.%j
#SBATCH --error=preprocess_era5_to_crea6_data-err.%j
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
VIRT_ENV_NAME="my_venv"

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

# path to data directories and files
src_dir_era5=/path/to/era5/data
src_dir_crea6=/path/to/cosmo/rea6/data
const_file_era5=/path/to/invariant/file/of/era5
const_file_crea6=/path/to/invariant/file/of/cosmo/rea6
grid_des_tar=/path/to/grid/description/
out_dir=/path/to/output/directory/
# selection of predictors and predictands
predictors='{"fc_sf": {"2t": "", "10u": "", "10v": "", "blh": "", "z": "", "sshf": "", "slhf": ""}, "fc_pl": {"t": ["p85000","p92500"]}}'
predictands='{"sf": {"t_2m": ""}, "invar": {"hsurf": ""}}'

# time request
years=( 2016 2017 2018 )
# months="all"             # still hard-coded
# preprocessing method
method=ERA5_to_CREA6

srun --overlap python -m mpi4py ../main_scripts/main_preprocessing.py \
  -in_datadir ${src_dir_era5} -tar_datadir ${src_dir_crea6} -out_dir ${out_dir} -grid_des_tar ${grid_des_tar} \
  -in_constfile ${const_file_era5} -tar_constfile ${const_file_crea6} -predictors "${predictors}" -predictands "${predictands}" \
  -y "${years[@]}" -method ${method}


