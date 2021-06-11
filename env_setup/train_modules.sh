#!/usr/bin/env bash

# __author__ = Michael Langguth
# __date__ = 2021-06-11
# __update__ = 2021-06-11

# This scripts loads all the modules required for training the GAN-downscaling model created by J.Leinonen.

HOST_NAME=`hostname` 

echo "%modules_train: Start loading modules for training on ${HOST_NAME}.."

module use $OTHERSTAGES
ml Stages/2020
ml GCC/9.3.0
ml GCCcore/.9.3.0
ml ParaStationMPI/5.4.7-1
ml cuDNN/8.0.2.39-CUDA-11.0
ml SciPy-Stack/2020-Python-3.8.5
ml TensorFlow/2.3.1-Python-3.8.5
ml netcdf4-python/1.5.4-Python-3.8.5
