#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=train_wgan-model-out.%j
#SBATCH --error=train_wgan-model-err.%j
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXX@fz-juelich.de

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment
VIRT_ENV_NAME=<my_venv>

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

# data-directories
indir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_ifs/netcdf_data/all_files/
outdir=/p/project/deepacf/maelstrom/langguth1/downscaling_jsc_repo/downscaling_ap5/trained_models/

# declare directory-variables which will be modified by config_runscript.py
nepochs=30
lr_gen=5.e-05
lr_critic=1.e-06
lr_end=5.e-06
lr_decay=True
model_name=my_wgan_model

srun --overlap python3 ../main_scripts/main_train_wgan.py -in ${indir} -out ${outdir} -lr_gen ${lr_gen} -lr_critic ${lr_critic} -lr_gen_end ${lr_end} \
                                                          -nepochs ${nepochs} -lr_decay -model_name ${model_name} -id ${SLURM_JOBID}

