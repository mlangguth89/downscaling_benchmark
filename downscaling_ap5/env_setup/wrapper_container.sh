#!/usr/bin/env bash

# basic directory variables
ENV_SETUP_DIR=`pwd`
WORKING_DIR="$(dirname "$ENV_SETUP_DIR")"
EXE_DIR="$(basename "$ENV_SETUP_DIR")"
VENV_DIR=$WORKING_DIR/virtual_envs/$1
shift                     # replaces $1 by $2, so that $@ does not include the name of the virtual environment anymore

if ! [[ -d "${VENV_DIR}" ]]; then
   echo "ERROR: Could not found virtual environment under ${VENV_DIR}!"
   exit
fi

ml ml go-1.17.6/singularity-3.9.5

# unset PYTHONPATH and activate virtual environment
unset PYTHONPATH
source ${VENV_DIR}/bin/activate

# Control
echo "****** Check PYTHONPATH *****"
echo $PYTHONPATH
# MPI related environmental variables
export PMIX_SECURITY_MODE="native"     # default would include munge which is unavailable

$@

