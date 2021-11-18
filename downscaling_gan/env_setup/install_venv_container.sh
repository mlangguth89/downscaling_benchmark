#!/usr/bin/env bash
#
# __authors__ = Bing Gong, Michael Langguth
# __date__  = '2021-10-28'
# __last_update__  = '2021-11-18' by Michael Langguth
#
# **************** Description ****************
# This auxiliary script sets up the virtual environment within a singularity container.
# **************** Description ****************

SCR_INSTALL="%install_venv_container.sh: "            # this script
# set some basic variables
BASE_DIR=$(pwd)
VENV_DIR=$1
VENV_NAME="$(basename "${VENV_DIR}")"
VENV_BASE="$(dirname "${VENV_DIR}")"

# sanity checks
# check if we are running in a container
if [ -z "${SINGULARITY_NAME}" ]; then
  echo "${SCR_INSTALL}ERROR: install_venv_container.sh must be called within a running singularity container."
  return
fi

# check if directory to virtual environment is parsed
if [ -z "$1" ]; then
  echo "${SCR_INSTALL}ERROR: Provide a name to set up the virtual environment."
  exit
fi

# check if virtual environment is not already existing
if [ -d "$1" ]; then
  echo "${SCR_INSTALL}ERROR: Target directory of virtual environment ${1} already exists. Choose another directory path."
  exit
fi

# check for requirement-file
if [ ! -f "${BASE_DIR}/requirements_container.txt" ]; then
  echo "${SCR_INSTALL}ERROR: Cannot find requirement-file ${BASE_DIR}/requirements_container.txt to set up virtual environment."
  exit
fi

# remove dependancies from system packages
export PYTHONPATH=

# create basic target directory for virtual environment
if ! [[ -d "${VENV_BASE}" ]]; then
  mkdir "${VENV_BASE}"
  # Install virtualenv in this directory
  echo "${SCR_INSTALL}Installing virtualenv under ${VENV_BASE}..."
  pip install --target="${VENV_BASE}/" virtualenv
  # Change into the base-directory of virtual environments...
  cd "${VENV_BASE}" || return
else
  # Change into the base-directory of virtual environments...
  cd "${VENV_BASE}" || return
  if ! python -m virtualenv --version >/dev/null; then
    echo "${SCR_INSTALL}ERROR: Base directory for virtual environment exists, but virtualenv-module is unavailable."
    exit
  fi
  echo "${SCR_INSTALL}Virtualenv is already installed."
fi
# ... and set-up virtual environment therein
#python -m virtualenv -p /usr/bin/python "${VENV_NAME}"
python -m virtualenv "${VENV_NAME}"
# Activate virtual environment and install required packages
echo "${SCR_INSTALL}Activating virtual environment ${VENV_NAME} to install required Python modules..."
source "${VENV_DIR}/bin/activate"
# set PYTHONPATH and install packages
export PYTHONPATH="/usr/local/lib/python3.8/dist-packages/"
echo 'export PYTHONPATH="/usr/local/lib/python3.8/dist-packages/"' >> "${VENV_DIR}/bin/activate"
pip install -r "${BASE_DIR}/requirements_container.txt"

# get back to basic directory
cd "${BASE_DIR}" || exit



