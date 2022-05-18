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

if [[ "${HOST_NAME}" == hdfml* ]] || [[ "${HOST_NAME}" == jw[b-c]* ]]; then
  ml purge
  ml use $OTHERSTAGES
  ml Stages/2022

  ml GCCcore/.11.2.0
  ml GCC/11.2.0
  ml ParaStationMPI/5.5.0-1
  ml mpi4py/3.1.3
  ml CDO/2.0.2
  ml NCO/5.0.3
  ml SciPy-bundle/2021.10
  ml xarray/0.20.1
  ml dask/2021.9.1
  ml TensorFlow/2.6.0-CUDA-11.5
else 
  echo "%${SCR_NAME_MOD}: Operating host system ${HOST_NAME} is unknown. Please work on HDF-ML or Juwels (Booster)..."
  return
  #exit
fi

