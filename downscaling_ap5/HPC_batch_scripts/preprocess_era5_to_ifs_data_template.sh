#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=37
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=preprocess_era5_to_ifs_data-out.%j
#SBATCH --error=preprocess_era5_to_ifs_data-err.%j
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:0
#SBATCH --partition=batch
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

# set variables to be parsed 
# input data directory and output directory for preprocessed data
src_dir_era5=/path/to/era5/data
src_dir_ifs=/path/to/ifs/data
out_dir=/path/to/output/directory/
invar_file=/path/to/invariant/file/
# predictors, predictands and grid description file for target domain
predictors='{"fc_sf": {"2t": "", "10u": "", "10v": "", "blh": "", "z": "", "sshf": "", "slhf": ""}, "fc_pl": {"t": ["p85000","p92500"]}}' # default
predictands='{"sf": {"t2m": "", "z": ""}}'                                                                                                # default
grid_des_tar=/path/to/grid/description

years=( YYYY YYYY YYYY YYYY)
# months=( 4 5 6 7 8 9 )        # "all" is still hard-coded
method=ERA5_to_IFS

srun --overlap python -m mpi4py ../main_scripts/main_preprocessing.py \
   -in_datadir ${src_dir_era5} -tar_datadir ${src_dir_ifs} -out_dir ${out_dir} -grid_des_tar ${grid_des_tar} -in_constfile ${invar_file} \
   -predictors "${predictors}" -predictands "${predictands}" -y "${years[@]}" -m "${months[@]}" -method ${method}


