#!/bin/bash

ML_SCR=modules_train.sh

ML_COLL=downscaling_unet_v1_0

# check if module collection has been set-up once
#RES=$( { ml restore ${ML_COLL}; } 2<&1 )

#if [[ "${RES}" == *"error"* ]]; then
if [[ 0 == 0 ]]; then  # Restoring from model collection currently throws MPI-setting errors. Thus, this approach is disabled for now.
  
#  echo "%${ML_SCR}: Module collection ${ML_COLL} does not exist and will be set up."

  ml --force purge
  ml use $OTHERSTAGES
  ml Stages/2022

  ml GCCcore/.11.2.0
  ml GCC/11.2.0
  ml ParaStationMPI/5.5.0-1
  ml mpi4py/3.1.3
  ml git/2.33.1-nodocs
  ml CDO/2.0.2
  ml NCO/5.0.3
  ml netcdf4-python/1.5.7-serial
  ml SciPy-bundle/2021.10
  ml xarray/0.20.1
  ml dask/2021.9.1
  ml TensorFlow/2.6.0-CUDA-11.5
  ml Cartopy/0.20.0
  ml Graphviz/2.49.3

#  ml save ${ML_COLL}
#  echo "%${ML_SCR}: Module collection ${ML_COLL} created successfully."
else
  echo "%${ML_SCR}: Module collection ${ML_COLL} already exists and is restored."
  ml restore ${ML_COLL}
fi
