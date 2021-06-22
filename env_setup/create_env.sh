#!/usr/bin/env bash
#
# __authors__ = Michael Langguth
# __date__  = '2021_03_25'
#
# **************** Description ****************
# This script can be used for setting up the virtual environment needed for AMBS-project
# Add the flag -lcontainer for setting up the virtual environment inside a running cotainer environment.
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
HOST_NAME=`hostname`
ENV_NAME=$1
ENV_SETUP_DIR=`pwd`
WORKING_DIR="$(dirname "$ENV_SETUP_DIR")"
EXE_DIR="$(basename "$ENV_SETUP_DIR")"
ENV_DIR_BASE=${WORKING_DIR}/virtual_envs/
ENV_DIR=${ENV_DIR_BASE}/${ENV_NAME}

## perform sanity checks
# Check if singularity is running
if [[ -z "${SINGULARITY_NAME}" ]]; then
  echo "ERROR: create_env.sh must be executed in a running singularity on Juwels in conjuction with container-usage."
  echo "Thus, execute 'singularity shell [my_docker_image]' first!"
  return
fi

# further sanity checks:
# * ensure execution from env_setup-directory
# * check if virtual env has already been set up

if [[ "${EXE_DIR}" != "env_setup"  ]]; then
  echo "ERROR: Execute 'create_env.sh' from the env_setup-subdirectory only!"
  return
fi

if [[ -d ${ENV_DIR} ]]; then
  echo "Virtual environment has already been set up under ${ENV_DIR} and is ready to use."
  echo "NOTE: If you wish to set up a new virtual environment, delete the existing one or provide a different name."
  
  ENV_EXIST=1
else
  ENV_EXIST=0
fi

## check integratability of modules
if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *jwlogin* ]]; then
  # unset PYTHONPATH to ensure that system-realted paths are not set (container-environment should be used only)
  unset PYTHONPATH
else
  echo "ERROR: Model only runs on HDF-ML and Juwels (Booster) so far."
  return
fi

## set up virtual environment
if [[ "$ENV_EXIST" == 0 ]]; then
  # Activate virtual environment and install additional Python packages.
  echo "Configuring and activating virtual environment on ${HOST_NAME}"

  VIRTUAL_ENV_TOOL=${ENV_DIR_BASE}/virtualenv-\*dist-info
  if ! ls "$VIRTUAL_ENV_TOOL" 1> /dev/null 2<&1; then
    if [[ ! -d ${ENV_DIR_BASE} ]]; then
      mkdir "${ENV_DIR_BASE}"
    fi
    echo "Install virtualenv in base directory for virtual environments ${ENV_DIR_BASE}"
    pip install --target="${ENV_DIR_BASE}" virtualenv
  fi
  # create virtual environment and install missing required packages
  cd "${ENV_DIR_BASE}"
  echo "***** Create and activate virtual environment ${ENV_NAME}... *****"
  python -m virtualenv -p /usr/bin/python --system-site-packages "${ENV_NAME}"

  activate_virt_env=${ENV_DIR}/bin/activate
  source "${activate_virt_env}"
  echo "***** Start installing additional Python modules with pip... *****"
  req_file=${ENV_SETUP_DIR}/requirements_container.txt
  pip3 install --no-cache-dir -r "${req_file}"
  # expand PYTHONPATH...
  export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH >> "${activate_virt_env}"
  export PYTHONPATH=${ENV_DIR}/lib/python3.8/site-packages/:$PYTHONPATH >> "${activate_virt_env}"
  # ...and ensure that this also done when the 
  echo "" >> "${activate_virt_env}"
  echo "# Expand PYTHONPATH..." >> "${activate_virt_env}"
  echo "export PYTHONPATH=${WORKING_DIR}:\$PYTHONPATH" >> "${activate_virt_env}"
  echo "export PYTONPATH=${ENV_DIR}/lib/python3.8/site-packages/:\$PYTHONPATH" >> "${activate_virt_env}"

  if [[ -f "${activate_virt_env}" ]]; then
    echo "Virtual environment ${ENV_DIR} has been set up successfully."
  else
    echo "***** ERROR: Cretaion of virtual environment was not successful. Check for preceiding error-messages! *****"
  fi
fi
## finally, deactivate virtual environment and clean up loaded modules
deactivate
