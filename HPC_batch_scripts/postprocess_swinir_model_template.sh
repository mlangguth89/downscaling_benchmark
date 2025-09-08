#!/bin/bash -x
#SBATCH --account=hclimrep
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=postprocess_swinir-model-out.%j
#SBATCH --error=postprocess_swinir-model-err.%j
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.patnala@fz-juelich.de

######### Template identifier (don't remove) #########
#echo "Do not run the template scripts"
#exit 99
######### Template identifier (don't remove) #########

# basic directories
WORK_DIR=$(pwd)
BASE_DIR=$(dirname "${WORK_DIR}")

# Name of virtual environment
VENV_DIR=${BASE_DIR}/virtual_envs/
VIRT_ENV_NAME=jwb_venv

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
datadir=/p/scratch/hclimrep/maelstrom/downscaling/downscaling_benchmark_dataset/benchmark_t2m/dataset/hres_input
model_basedir=../trained_models/
outdir=${BASE_DIR}/trained_models/
config_postprocess=../config/postprocessing/config_postprocess_t2m_test.json
ckpt=epoch=599-step=151200.ckpt

exp_name=ankit_swinir_new_lr_12_wihtout_zbranch_biasfree_conv_128
dataset=benchmark_t2m

# run job
# run job
# data inference with trained model of this framework
srun --overlap python3 ${BASE_DIR}/main_scripts/main_postprocess_lightning.py --output_base_directory ${outdir} --configuration_postprocess ${config_postprocess} -exp_name ${exp_name} inference \
                                                                   -model_base_dir ${model_basedir} -data_dir ${datadir} -dataset ${dataset} -epoch ${ckpt}

## data provided from netcdf-file
#results_nc=<path_results>
#
#srun --overlap python3 ${BASE_DIR}/main_scripts/main_postprocess.py --output_base_directory ${outdir} --configuration_postprocess ${config_postprocess} -exp_name ${exp_name} provided_results \
#                                                                    --results_netcdf ${results_nc} --model_name "${model_name}"
#                                                                    -output_base_dir ${outdir} -exp_name ${exp_name} -dataset ${dataset}
