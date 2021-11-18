#!/usr/bin/env bash
#
# __authors__ = Michael Langguth
# __date__  = '2021_03_25'
# __update__= '2021-11-17'
#
# **************** Description ****************
# This script can be used for setting up the virtual environment needed for downscaling with the GAN-network
# developped by Leinonen et al., 2020 (DOI: https://doi.org/10.1109/TGRS.2020.3032790)
# **************** Description ****************
#
### auxiliary-function S ###
check_argin() {
# Handle input arguments and check if one is equal to -lcontainer (not needed currently)
# Can also be used to check for non-positional arguments (such as -exp_id=*, see commented lines)
    for argin in "$@"; do
        # if [[ $argin == *"-exp_id="* ]]; then
        #  exp_id=${argin#"-exp_id="}
        if [[ $argin == *"-lcontainer"* ]]; then
	        bool_container=1
        fi  
    done
    if [[ -z "${bool_container}" ]]; then
        bool_container=0
    fi
}
### auxiliary-function E ###

### MAIN S ###
#set -eu              # enforce abortion if a command is not re

## some first sanity checks
# script is sourced?
if [[ ${BASH_SOURCE[0]} == "${0}" ]]; then
  echo "ERROR: 'create_env.sh' must be sourced, i.e. execute by prompting 'source create_env.sh [virt_env_name]'"
  exit 1
fi


# from now on, just return if something unexpected occurs instead of exiting
# as the latter would close the terminal including logging out
if [[ -z "$1" ]]; then
  echo "ERROR: Provide a name to set up the virtual environment, i.e. execute by prompting 'source create_env.sh [virt_env_name]"
  return
fi

# set some variables
SCR_NAME="%create_env.sh:"
HOST_NAME=$(hostname)
ENV_NAME=$1
SETUP_DIR=$(pwd)
SETUP_DIR_NAME="$(basename "${SETUP_DIR}")"
BASE_DIR="$(dirname "${SETUP_DIR}")"
VENV_DIR="${BASE_DIR}/virtual_envs/${ENV_NAME}"
TF_CONTAINER="${SETUP_DIR}/tensorflow_21.09-tf1-py3.sif"

## perform sanity checks
# * ensure availability of singularity container
# * check if script is called from env_setup-directory
# * check if virtual env has already been set up
# Check if the required TF1.15-container is available
  if [[  ! -f "${TF_CONTAINER}" ]]; then
    echo "ERROR: Could not found required TensorFlow 1.15-container under ${TF_CONTAINER}"
    return
  fi

# script is called from env_setup-directory?
if [[ "${SETUP_DIR_NAME}" != "env_setup"  ]]; then
  echo "${SCR_NAME} ERROR: Execute 'create_env.sh' from the env_setup-subdirectory only!"
  echo ${SETUP_DIR_NAME}
  return
fi

# virtual environment already set-up?
if [[ -d ${VENV_DIR} ]]; then
  echo "${SCR_NAME} Virtual environment has already been set up under ${VENV_DIR} and is ready to use."
  echo "NOTE: If you wish to set up a new virtual environment, delete the existing one or provide a different name."
  ENV_EXIST=1
else
  ENV_EXIST=0
fi

## check integratability of operating system
if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *jwlogin* ]]; then
  # unset PYTHONPATH to ensure that system-realted paths are not set (container-environment should be used only)
  unset PYTHONPATH
else
  echo "${SCR_NAME} ERROR: Model only runs on HDF-ML and Juwels (Booster) so far."
  return
fi

## set up virtual environment
if [[ "$ENV_EXIST" == 0 ]]; then
  # Install virtualenv-package and set-up virtual environment with required additional Python packages.
  echo "${SCR_NAME} Configuring and activating virtual environment on ${HOST_NAME}"

  singularity exec --nv "${TF_CONTAINER}" ./install_venv_container.sh "${VENV_DIR}"

  info_str="Virtual environment ${VENV_DIR} has been set up successfully."
elif [[ "$ENV_EXIST" == 1 ]]; then
  # simply activate virtual environment
  info_str="Virtual environment ${VENV_DIR} has already been set up before. Nothing to be done."
fi

echo "${info_str}"
### MAIN E ###
