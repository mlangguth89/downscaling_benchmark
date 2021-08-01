#!/usr/bin/env bash
#
# __authors__ = Michael Langguth
# __date__  = '2021_08_01'
#
# **************** Description ****************
# This script can be used for setting up the virtual environment needed for
# the downscaling application in scope of the MAELSTROM-project.
# **************** Description ****************
#
### auxiliary-function ###
check_argin() {
# Handle input arguments and check if one is equal to -lcontainer
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
### auxiliary-function ###

# some first sanity checks
if [[ ${BASH_SOURCE[0]} == ${0} ]]; then
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
HOST_NAME=`hostname`
ENV_NAME=$1
ENV_SETUP_DIR=`pwd`
WORKING_DIR="$(dirname "$ENV_SETUP_DIR")"
EXE_DIR="$(basename "$ENV_SETUP_DIR")"
ENV_DIR=${WORKING_DIR}/${ENV_NAME}

## perform sanity checks

# * ensure execution from env_setup-directory
# * check if virtual env has already been set up

if [[ "${EXE_DIR}" != "env_setup"  ]]; then
  echo "${SCR_NAME} ERROR: Execute 'create_env.sh' from the env_setup-subdirectory only!"
  return
fi

if [[ -d ${ENV_DIR} ]]; then
  echo "${SCR_NAME} Virtual environment has already been set up under ${ENV_DIR} and is ready to use."
  echo "NOTE: If you wish to set up a new virtual environment, delete the existing one or provide a different name."
  
  ENV_EXIST=1
else
  ENV_EXIST=0
fi

## check integratability of modules
if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *jwlogin* ]]; then
  source modules_preprocess.sh
else
  echo "${SCR_NAME} ERROR: Model only runs on HDF-ML and Juwels (Booster) so far."
  return
fi

## set up virtual environment
if [[ "$ENV_EXIST" == 0 ]]; then
  # Activate virtual environment and install additional Python packages.
  echo "${SCR_NAME} Configuring and activating virtual environment on ${HOST_NAME}"

  python3 -m venv ${ENV_DIR}
  activate_virt_env=${ENV_DIR}/bin/activate

  source "${activate_virt_env}"
  echo "${SCR_NAME} Start installing additional Python modules with pip..."
  req_file=${ENV_SETUP_DIR}/requirements_preporcessing.txt
  pip3 install --no-cache-dir -r "${req_file}"
  
  # expand PYTHONPATH...
  export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH >> "${activate_virt_env}"
  export PYTHONPATH=${WORKING_DIR}/utils/:$PYTHONPATH >> "${activate_virt_env}"
  # ...and ensure that this also done when the 
  echo "" >> "${activate_virt_env}"
  echo "# Expand PYTHONPATH..." >> "${activate_virt_env}"
  echo "export PYTHONPATH=${WORKING_DIR}:\$PYTHONPATH" >> "${activate_virt_env}"
  echo "export PYTONPATH=${WORKING_DIR}/utils/:\$PYTHONPATH" >> "${activate_virt_env}"

  if [[ -f "${activate_virt_env}" ]]; then
    echo "${SCR_NAME} Virtual environment ${ENV_DIR} has been set up successfully."
  else
    echo "${SCR_NAME} ERROR: Creation of virtual environment was not successful. Check for preceiding error-messages!"
  fi
fi
