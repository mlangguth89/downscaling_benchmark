#!/usr/bin/env bash

# basic directory variables
ENV_SETUP_DIR=`pwd`
WORKING_DIR="$(dirname "$ENV_SETUP_DIR")"
EXE_DIR="$(basename "$ENV_SETUP_DIR")"
VENV_DIR="${WORKING_DIR}/virtual_envs/$1"
shift                     # replaces $1 by $2, so that $@ does not include the name of the virtual environment anymore

# sanity checks
if [[ "${EXE_DIR}" != "batch_scripts"  ]]; then
  echo "ERROR: Run the HPC-batch script for the enviornment from the batch_scripts-directory!"
  exit
fi

if ! [[ -d "${VENV_DIR}" ]]; then
   echo "ERROR: Could not found virtual environment under ${VENV_DIR}!"
   exit
fi

# unset old PYTHONPATH and expand PYHTONPATH
# Include site-packages from virtual environment...
unset PYTHONPATH
export PYTHONPATH=${VENV_DIR}/lib/python3.8/site-packages/:$PYTHONPATH
# ... dist-packages from container singularity...
export PYTHONPATH=/usr/local/lib/python3.8/dist-packages:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH

# Control
echo "%wrapper_container.sh: Check PYTHONPATH below:"
echo $PYTHONPATH

# MPI related environmental variables
export PMIX_SECURITY_MODE="native"
export TF_XLA_FLAGS=--tf_xla_auto_jit=0      # disable XLA graph optimization
$@

