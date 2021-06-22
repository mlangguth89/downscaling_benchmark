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

# init bool_container
bool_container=0
# some first sanity checks
if [[ ${BASH_SOURCE[0]} == ${0} ]]; then
  echo "ERROR: 'create_env.sh' must be sourced, i.e. execute by prompting 'source create_env.sh [virt_env_name]'"
  exit 1
fi

# from now on, just return if something unexpected occurs instead of exiting
# as the latter would close the terminal including logging out
if [[ ! -n "$1" ]]; then
  echo "ERROR: Provide a name to set up the virtual environment, i.e. execute by prompting 'source create_env.sh [virt_env_name]"
  return
fi

if [[ "$#" -gt 1 ]]; then
  check_argin ${@:2}                 # sets exp_id if provided, always sets l_container 
fi

# set some variables
HOST_NAME=`hostname`
ENV_NAME=$1
ENV_SETUP_DIR=`pwd`
WORKING_DIR="$(dirname "$ENV_SETUP_DIR")"
EXE_DIR="$(basename "$ENV_SETUP_DIR")"
ENV_DIR=${WORKING_DIR}/${ENV_NAME}

## perform sanity checks
# correct bool_container if host is Juwels Booster and ensure running singularity
if [[ "${bool_container}" == 0 ]] && [[ "${HOST_NAME}" == jwlogin2[1-4]* ]]; then
  echo "******************************************** NOTE ********************************************"
  echo "          Running on Juwels Booster is only possible inside a container environment.          "
  echo "******************************************** NOTE ********************************************"
  bool_container=1
fi

if [[ "${bool_container}" == 1 ]]; then
  echo "******************************************** NOTE ********************************************"
  echo "                Make use of dedicated Horovod-related working branches only!!!                "
  echo "******************************************** NOTE ********************************************"
  # Check if singularity is running
  if [[ -z "${SINGULARITY_NAME}" ]]; then
    echo "ERROR: create_env.sh must be executed in a running singularity on Juwels in conjuction with container-usage."
    echo "Thus, execute 'singularity shell [my_docker_image]' first!"
    return
  fi
fi

# further sanity checks:
# * ensure execution from env_setup-directory
# * check if virtual env has already been set up

if [[ "${EXE_DIR}" != "env_setup"  ]]; then
  echo "ERROR: Execute 'create_env.sh' from the env_setup-subdirectory only!"
  return
fi

if [[ -d ${ENV_DIR} ]]; then
  echo "Virtual environment has already been set up under ${ENV_DIR}. The present virtual environment is activated now."
  echo "NOTE: If you wish to set up a new virtual environment, delete the existing one or provide a different name."
  
  ENV_EXIST=1
else
  ENV_EXIST=0
fi

## check integratability of modules
if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *jwlogin* ]]; then
  if [[ "${bool_container}" > 0 ]]; then  
    # on Juwels Booster, we are in a container environment -> loading modules is not possible	  
    echo "***** Note for container environment! *****"
    echo "Already checked the required modules?"
    echo "To do so, run 'source modules_train.sh' after exiting the singularity."
    echo "***** Note for container environment! *****"
  else
    # load modules and check for their availability
    echo "***** Checking modules required during the workflow... *****"
    source ${ENV_SETUP_DIR}/modules_preprocess.sh purge
    source ${ENV_SETUP_DIR}/modules_train.sh purge
    source ${ENV_SETUP_DIR}/modules_postprocess.sh
  fi
else 
  # unset PYTHONPATH on every other machine that is not a known HPC-system	
  unset PYTHONPATH
fi

## set up virtual environment
if [[ "$ENV_EXIST" == 0 ]]; then
  # Activate virtual environment and install additional Python packages.
  echo "Configuring and activating virtual environment on ${HOST_NAME}"
    
  python3 -m venv $ENV_DIR
  
  activate_virt_env=${ENV_DIR}/bin/activate

  echo "Entering virtual environment ${ENV_DIR} to install required Python modules..."
  source ${activate_virt_env}
  
  # install some requirements and/or check for modules
  if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *jwlogin* ]]; then
    # Install packages depending on host 
    echo "***** Start installing additional Python modules with pip... *****"
    req_file=${ENV_SETUP_DIR}/requirements.txt 
    if [[ "${bool_container}" > 0 ]]; then req_file=${ENV_SETUP_DIR}/requirements_container.txt; fi
    
    pip3 install --no-cache-dir -r ${req_file}
  else
    echo "***** Start installing additional Python modules with pip... *****"
    pip3 install --upgrade pip
    pip3 install -r ${ENV_SETUP_DIR}/requirements.txt
    pip3 install  mpi4py 
    pip3 install netCDF4
    pip3 install  numpy
    pip3 install h5py
    pip3 install tensorflow-gpu==1.13.1
  fi

  # expand PYTHONPATH...
  export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH >> ${activate_virt_env}
  export PYTHONPATH=${WORKING_DIR}/utils:$PYTHONPATH >> ${activate_virt_env}
  export PYTHONPATH=${WORKING_DIR}/external_package/lpips-tensorflow:$PYTHONPATH >> ${activate_virt_env}
  export PYTHONPATH=${WORKING_DIR}/model_modules:$PYTHONPATH >> ${activate_virt_env}

  if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *jwlogin* ]]; then
     export PYTHONPATH=${ENV_DIR}/lib/python3.6/site-packages:$PYTHONPATH >> ${activate_virt_env}
     if [[ "${bool_container}" > 0 ]]; then  
       export PYTONPATH=/usr/locali/lib/python3.6/dist-packages:$PYTHONPATH
     fi
  fi
  # ...and ensure that this also done when the 
  echo "" >> ${activate_virt_env}
  echo "# Expand PYTHONPATH..." >> ${activate_virt_env}
  echo "export PYTHONPATH=${WORKING_DIR}:\$PYTHONPATH" >> ${activate_virt_env}
  echo "export PYTHONPATH=${WORKING_DIR}/utils/:\$PYTHONPATH" >> ${activate_virt_env}
  echo "export PYTHONPATH=${WORKING_DIR}/model_modules:$PYTHONPATH " >> ${activate_virt_env}
  echo "export PYTHONPATH=${WORKING_DIR}/external_package/lpips-tensorflow:\$PYTHONPATH" >> ${activate_virt_env}

  if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *juwels* ]]; then
    echo "export PYTHONPATH=${ENV_DIR}/lib/python3.6/site-packages:\$PYTHONPATH" >> ${activate_virt_env}
    if [[ "${bool_container}" > 0 ]]; then  
       echo "export PYTONPATH=/usr/locali/lib/python3.6/dist-packages:\$PYTHONPATH" >> ${activate_virt_env}
     fi
  fi
  info_str="Virtual environment ${ENV_DIR} has been set up successfully."
elif [[ "$ENV_EXIST" == 1 ]]; then
  # activating virtual env is suifficient
  source ${ENV_DIR}/bin/activate
  info_str="Virtual environment ${ENV_DIR} has been activated successfully."
fi

echo "******************************************** NOTE ********************************************"
echo ${info_str}
echo "Make use of config_runscript.py to generate customized runscripts of the workflow steps."
echo "******************************************** NOTE ********************************************"

## finally, deactivate virtual environment and clean up loaded modules
# (if we on an HPC-system and not inside a running singularity)
deactivate
if [[ "${HOST_NAME}" == *hdfml* || "${HOST_NAME}" == *jwlogin* ]] && [[ "${bool_container}" == 0 ]]; then
  module --force purge
fi
