#!/bin/bash

ml --force purge
ml use $OTHERSTAGES
ml Stages/2022

ml GCCcore/.11.2.0
ml GCC/11.2.0
#ml ParaStationMPI/5.5.0-1
ml OpenMPI/4.1.2
ml mpi4py/3.1.3
ml tqdm/4.62.3
ml CDO/2.0.2
ml NCO/5.0.3
ml netcdf4-python/1.5.7-serial
ml scikit-image/0.18.3
ml SciPy-bundle/2021.10
ml xarray/0.20.1
ml dask/2021.9.1
ml TensorFlow/2.6.0-CUDA-11.5
ml Horovod/0.24.3
ml Cartopy/0.20.0
ml Graphviz/2.49.3
