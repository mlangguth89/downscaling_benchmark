#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=postprocess_my-model-out.%j
#SBATCH --error=postprocess_my-model-err.%j
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXX@fz-juelich.de

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# basic directories
WORK_DIR=$(pwd)
BASE_DIR=$(dirname "${WORK_DIR}")

# Name of virtual environment
VENV_DIR=${BASE_DIR}/virtual_envs/
VIRT_ENV_NAME=<my_venv>

# Loading mouldes
source ../env_setup/modules_jsc.sh
# Activate virtual environment if needed (and possible)
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ${VENV_DIR}/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ${VENV_DIR}/${VIRT_ENV_NAME}/bin/activate
   else
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi


# data-directories
# Note template uses Tier2-dataset. Adapt accordingly for other datasets.
datadir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/benchmark_t2m/all_files/
model_basedir=../trained_models/
outdir=../results/

exp_name=<my_exp>
dataset=benchmark_t2m

# run job
srun --overlap python3 ${BASE_DIR}/main_scripts/main_postprocess.py -data_dir ${datadir} -model_base_dir ${model_basedir} \
                                                                    -output_base_dir ${outdir} -exp_name ${exp_name} -dataset ${dataset}

