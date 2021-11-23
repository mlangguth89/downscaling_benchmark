#!/bin/bash
#
# __authors__ = Michael Langguth
# __date__  = '2021_11_22'
# __update__= '2021-11-22'
#
# **************** Description ****************
# Jupyter Kernel including cartopy (available on the system) and 
# climetlab (must be installed with pip)
# **************** Description ****************

# Load required modules
module -q purge
module -q use $OTHERSTAGES
ml Stages/2020
ml GCCcore/.10.3.0
ml GCC/10.3.0
ml ParaStationMPI/5.4.9-1
ml netCDF/4.7.4
ml Cartopy/0.18.0-Python-3.8.5
ml SciPy-Stack/2021-Python-3.8.5
ml TensorFlow/2.5.0-Python-3.8.5

# Activate your Python virtual environment
source /p/project/deepacf/maelstrom/langguth1/jup_kernel_maelstrom/bin/activate 

# Ensure python packages installed in the virtual environment are always prefered
export PYTHONPATH=/p/project/deepacf/maelstrom/langguth1/jup_kernel_maelstrom/lib/python3.8/site-packages:${PYTHONPATH}

exec python -m ipykernel $@
