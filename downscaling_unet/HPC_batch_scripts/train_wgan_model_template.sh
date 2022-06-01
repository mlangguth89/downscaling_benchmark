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
#SBATCH --mail-user=m.langguth@fz-juelich.de

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment
VIRT_ENV_NAME="venv_juwels"

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
nepochs=30
lr_gen=5.e-05
lr_critc=1.e-06
lr_end=5.e-06
lr_decay=True
model_name=wgan_lr1e-05_epochs30_opt_split

srun python3 ../main_scripts/wgan_model.py -lr_gen ${lr_gen} -lr_critic ${lr_critic} -lr_gen_end ${lr_end} -nepochs ${nepochs}
                                           -lr_decay -model_name ${model_name} -id ${SLURM_JOBID}
