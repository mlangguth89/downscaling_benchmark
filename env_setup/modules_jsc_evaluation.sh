#!/bin/bash

ml --force purge
ml use $OTHERSTAGES
ml Stages/2025

ml GCCcore/.13.3.0
ml GCC/13.3.0
ml OpenMPI/5.0.5

ml numba/0.60.0
ml SciPy-bundle/2024.05
ml xarray/2024.9.0
ml matplotlib/3.9.2
ml dask/2024.9.1
ml netcdf4-python/1.7.1.post2
ml h5py/3.12.1

ml Cartopy/0.24.1
