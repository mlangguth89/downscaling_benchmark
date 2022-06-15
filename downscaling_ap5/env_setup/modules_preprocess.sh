#!/usr/bin/env bash

# __author__ = Michael Langguth
# __date__  = '2020_08_01'

# This script loads the required modules for the preprocessing of IFS HRES data in scope of the
# downscaling application in scope of the MAELSTROM project on Juwels and HDF-ML.
# Note that some other packages have to be installed into a venv (see create_env.sh and requirements_preprocess.txt).

SCR_NAME_MOD="modules_preprocess.sh"
HOST_NAME=`hostname`

# start loading modules
echo "%${SCR_NAME_MOD}: Start loading modules on ${HOST_NAME} required for preprocessing IFS HRES data."

ml purge
ml use $OTHERSTAGES
ml Stages/2020

if [[ "${HOST_NAME}" == hdfml* ]]; then
  ml GCC/10.3.0
  ml GCCcore/.10.3.0
  ml ParaStationMPI/5.4.9-1
  ml CDO/2.0.0rc3
  ml NCO/4.9.5
  ml mpi4py/3.0.3-Python-3.8.5
  ml SciPy-Stack/2021-Python-3.8.5
  ml dask/2.22.0-Python-3.8.5
  ml TensorFlow/2.5.0-Python-3.8.5
elif [[ "${HOST_NAME}" == jw[b-c]* ]]; then
  ml GCC/9.3.0
  ml GCCcore/.9.3.0
  ml ParaStationMPI/5.4.7-1
  ml CDO/1.9.8
  ml NCO/4.9.5
  ml mpi4py/3.0.3-Python-3.8.5
  ml SciPy-Stack/2020-Python-3.8.5
  ml dask/2.22.0-Python-3.8.5
  ml TensorFlow/2.3.1-Python-3.8.5
else 
  echo "%${SCR_NAME_MOD}: Operating host system ${HOST_NAME} is unknown. Please work on HDF-ML or Juwels (Booster)..."
  exit
fi

