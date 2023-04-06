#!/usr/bin/env bash
#
# __authors__ = Michael Langguth
# __date__  = '2022-01-21'
# __update__= '2023-03-02'
#
# **************** Description ****************
# This script can be used for setting up the virtual environment together with JUBE.
# The script either sets up a new virtual environment or activates an existing one
# after checking if the virtual environment according to the parsed name exists.
# **************** Description ****************
#
### auxiliary-function S ###
check_argin() {
# Handle input arguments and check if one is equal to -lcontainer (not needed currently)
# Can also be used to check for non-positional arguments (such as -exp_id=*, see commented lines)
# NOT USED YET!
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

SCR_SETUP="%setup_env_jube.sh: "

## some first sanity checks
# script is sourced?
if [[ ${BASH_SOURCE[0]} == "${0}" ]]; then
  echo "${SCR_SETUP}ERROR: 'create_env.sh' must be sourced, i.e. execute by prompting 'source create_env.sh [virt_env_name]'"
  exit 1
fi


# from now on, just return if something unexpected occurs instead of exiting
# as the latter would close the terminal including logging out
if [[ -z "$1" ]]; then
  echo "${SCR_SETUP}ERROR: Provide a name to set up the virtual environment, i.e. execute by prompting 'source create_env.sh [virt_env_name]"
  return
fi

# set some variables
HOST_NAME=$(hostname)
ENV_NAME=$1
SETUP_DIR=$(pwd)
SETUP_DIR_NAME="$(basename "${SETUP_DIR}")"
BASE_DIR="$(dirname "${SETUP_DIR}")"
VENV_DIR="${BASE_DIR}/virtual_envs/${ENV_NAME}"

## perform sanity checks
# * check if script is called from env_setup-directory
# * check if virtual env has already been set up
# * check if system is supported

# script is called from env_setup-directory?
if [[ "${SETUP_DIR_NAME}" != "env_setup"  ]]; then
  echo "${SCR_SETUP}ERROR: Execute 'create_env.sh' from the env_setup-subdirectory only!"
  echo "${SETUP_DIR_NAME}"
  return
fi

# virtual environment already set-up?
if [[ -d ${VENV_DIR} ]]; then
  echo "${SCR_SETUP}Virtual environment has already been set up under ${VENV_DIR} and is ready to use."
  echo "NOTE: If you wish to set up a new virtual environment, delete the existing one or provide a different name."
  ENV_EXIST=1
else
  ENV_EXIST=0
fi

## check operating system is
SUPPORTED_SYSTEMS=("HDFML" "JWB" "JRCMI200" "E4")
declare -A SYSTEM_HUMAN_NAMES
SYSTEM_HUMAN_NAMES["HDFML"]="HDF-ML"
SYSTEM_HUMAN_NAMES["JWB"]="Juwels (Booster)"
SYSTEM_HUMAN_NAMES["JRCMI200"]="Jureca (MI200 partition)"
SYSTEM_HUMAN_NAMES["E4"]="E4"
declare -A SYSTEM_HOSTNAME_PATTERNS
SYSTEM_HOSTNAME_PATTERNS["HDFML"]="hdfml"
SYSTEM_HOSTNAME_PATTERNS["JWB"]="jwlogin"
SYSTEM_HOSTNAME_PATTERNS["JRCMI200"]="jrlogin"
SYSTEM_HUMAN_PATTERNS["E4"]="ilnode"
declare -A SYSTEM_IS_JSC
SYSTEM_IS_JSC["HDFML"]=1
SYSTEM_IS_JSC["JWB"]=1
SYSTEM_IS_JSC["JRCMI200"]=1
SYSTEM_IS_JSC["E4"]=0

IS_SUPPORTED_SYSTEM=0
for SYSTEM in "${SUPPORTED_SYSTEMS[@]}"
do
  HNAME_PATTERN="${SYSTEM_HOSTNAME_PATTERNS[$SYSTEM]}"
  echo "Checking if ${HOST_NAME} matches pattern ${HNAME_PATTERN}"
  if [[ "${HOST_NAME}" == *"${HNAME_PATTERN}"* ]]; then
    echo "Match! System is supported"
    IS_SUPPORTED_SYSTEM=1
    SYSTEM_NOW=$SYSTEM
    break
  else
    echo "No match!"
  fi
done

if ! [[ $IS_SUPPORTED_SYSTEM == 1 ]]; then
  echo "${SCR_SETUP}ERROR: Current operating system ${HOST_NAME} is not supported."
  exit 1
fi

# On e4, a virtual environment is already provided. Thus, we just need to activate it and expand PYTONPATH
if [[ "${SYSTEM_NOW}" == "E4" ]]; then
  VENV_DIR=/opt/share/users/maelstrom/venv-rocm
  ENV_EXIST=1
  # expand PYTHONPATH
  export PYTHONPATH=${BASE_DIR}:$PYTHONPATH
  export PYTHONPATH=${BASE_DIR}/utils:$PYTHONPATH
  export PYTHONPATH=${BASE_DIR}/handle_data:$PYTHONPATH
  export PYTHONPATH=${BASE_DIR}/models:$PYTHONPATH
  export PYTHONPATH=${BASE_DIR}/postprocess:$PYTHONPATH
  export PYTHONPATH=${BASE_DIR}/preprocess:$PYTHONPATH
fi

activate_virt_env=${VENV_DIR}/bin/activate

## set up virtual environment
if [[ "$ENV_EXIST" == 0 ]]; then
  # Install virtualenv-package and set-up virtual environment with required additional Python packages.
  echo "${SCR_SETUP}Configuring and activating virtual environment on ${HOST_NAME}"

  python3 -m venv --system-site-packages "${VENV_DIR}"

  echo "${SCR_SETUP}Entering virtual environment ${VENV_DIR} to install required Python modules..."
  source "${activate_virt_env}"
 
  # get machine and PYTHON-version
  MACHINE=$(hostname -f | cut -d. -f2)
  if [[ "${HOST}" == jwlogin2[2-4] ]]; then
     MACHINE="juwelsbooster"
  fi
  PY_VERSION=$(python --version 2>&1 | cut -d ' ' -f2 | cut -d. -f1-2)
  echo "${SCR_SETUP}Appending PYTHONPATH on ${MACHINE} for Python version ${PY_VERSION} to ensure proper set-up..."

  req_file=${SETUP_DIR}/requirements.txt

  # Without the environmental variables set above, we need to install wheel and explictly set the target directory
  pip3 install --no-cache-dir -r "${req_file}"

  # expand PYTHONPATH
  export PYTHONPATH=${BASE_DIR}:$PYTHONPATH >> ${activate_virt_env} 
  export PYTHONPATH=${BASE_DIR}/utils:$PYTHONPATH >> ${activate_virt_env}
  export PYTHONPATH=${BASE_DIR}/handle_data:$PYTHONPATH >> ${activate_virt_env}
  export PYTHONPATH=${BASE_DIR}/models:$PYTHONPATH >> ${activate_virt_env}
  export PYTHONPATH=${BASE_DIR}/postprocess:$PYTHONPATH >> ${activate_virt_env}
  export PYTHONPATH=${BASE_DIR}/preprocess:$PYTHONPATH >> ${activate_virt_env}

  # ...and ensure that this also done when the
  echo "" >> ${activate_virt_env}
  echo "# Expand PYTHONPATH..." >> ${activate_virt_env}
  echo "export PYTHONPATH=${BASE_DIR}:\$PYTHONPATH" >> ${activate_virt_env}
  echo "export PYTHONPATH=${BASE_DIR}/utils/:\$PYTHONPATH" >> ${activate_virt_env}
  echo "export PYTHONPATH=${BASE_DIR}/models:\$PYTHONPATH " >> ${activate_virt_env}
  echo "export PYTHONPATH=${BASE_DIR}/handle_data:\$PYTHONPATH" >> ${activate_virt_env}
  echo "export PYTHONPATH=${BASE_DIR}/postprocess:\$PYTHONPATH" >> ${activate_virt_env}
  echo "export PYTHONPATH=${BASE_DIR}/preprocess:\$PYTHONPATH" >> ${activate_virt_env}

  info_str="Virtual environment ${VENV_DIR} has been set up successfully."
elif [[ "$ENV_EXIST" == 1 ]]; then
  # simply activate virtual environment
  info_str="Virtual environment ${VENV_DIR} has already been set up before. Activate it now."
  source "${activate_virt_env}"
fi

echo "${SCR_SETUP}${info_str}"
### MAIN E ###
