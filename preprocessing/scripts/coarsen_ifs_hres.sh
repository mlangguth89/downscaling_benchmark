#!/usr/bin/env bash

# __author__ = Michael Langguth
# __date__  = '2021_07_23'

# This script creates the input for the downscaling task tackled wit a Unet in the scope of the MAELSTROM project.
# The script requires that the netCDF-tools ncap2 and ncrea are available. Besides, CDO is required for remapping the data.
# For first-order conservative remapping, CDO must be installed with PROJ. 

# basic variables
scr_name="%coarsen_ifs_hres.sh"

HOST_NAME=`hostname`

# grid descriptions 
fine_grid_base_dscr="../grid_des/ifs_hres_grid_base"
fine_grid_tar_dscr="../grid_des/ifs_hres_grid_tar"
coarse_grid_dscr="../grid_des/ifs_hres_coarsened_grid"

# start and end coordinates of target domain (must correspond to grid descriptions!)
lon0="3.2"
lon1="17.5"
lat0="44.2"
lat1="55.3"
dx="0.8"

# some IFS-specific parameters (obtained from Chapter 12 in http://dx.doi.org/10.21957/efyk72kl)
cpd="1004.709"
g="9.80665"

### some sanity checks ###
if [[ ! -n "$1" ]]; then
  echo "${scr_name}: Pass a path to a netCDF-file to script."
fi

filename=$1
filename_base="${filename%.*}"

if ! command -v cdo &> /dev/null; then
  echo "${scr_name}: CDO is not available on ${HOST_NAME}";
  return 1
fi

if ! command -v ncap2 &> /dev/null; then
  echo "${scr_name}: ncap2 is not available on ${HOST_NAME}";
  return 1
fi

if ! command -v ncea &> /dev/null; then
  echo "${scr_name}: ncea is not available on ${HOST_NAME}";
  return 1
fi

if [[ ! -f "$1" ]]; then
  echo "${scr_name}: The file $1 does not exist.";
  return 1
fi

if [[ ! -f ${fine_grid_base_dscr} ]]; then
  echo "${scr_name}: ERROR: The basic grid description for IFS HRES ${fine_grid_base_dscr} is missing."
  return 1
fi

if [[ ! -f ${coarse_grid_dscr} ]]; then
  echo "${scr_name}: ERROR: The grid description for the coarse grid ${coarse_grid_dscr} is missing."
  return 1
fi

if [[ ! -f ${fine_grid_tar_dscr} ]]; then
  echo "${scr_name}: ERROR: The target grid description for IFS HRES ${fine_grid_tar_dscr} is missing."
  return 1
fi

#return 0

### Start the work ###

# shrink data to region of interest
filename_sd="${filename_base}_subdom.nc"

ncea -O -d latitude,${lat0},${lat1} -d longitude,${lon0},${lon1} $filename $filename_sd
#Change the variables names from latitude to lat and longitud to lon
ncrename -d latitude,lat -v latitude,lat -d longitude,lon -v longitude,lon ${filename_sd}
#change the precision, but why ?
ncap2 -O -s "lat=double(lat); lon=double(lon)" ${filename_sd} ${filename_sd}
#input solutio has 0.1 degree, and coarse the inputs to 0.8 degree grid. 
# reset coordinates for later slicingß
lat0="45.0"
lat1="54.5"
lon0="4.0"
lon1="16.71"


filename_dse="${filename_base}_dse.nc"
filename_dset="${filename_base}_dset.nc"


# add precipitation to file 
ncks -A -v lsp,cp,z $filename_sd $filename_dset

#calculate the total precipitation to a new file
cdo  expr,"tp=cp+lsp;z=z;"  $filename_dset $filename_dse

# remap the data (first-order conservative approach)
filename_crs="${filename_base}_coarse.nc"
cdo remapcon,${coarse_grid_dscr} -setgrid,${fine_grid_base_dscr} ${filename_dse} ${filename_crs}

# remap with extrapolation on the target high resoved grid with bilinear rempapping
filename_remapped="${filename_base}_remapped.nc"
cdo remapbil,${fine_grid_tar_dscr} -setgrid,${coarse_grid_dscr} ${filename_crs} ${filename_remapped}

# retransform dry static energy to t2m
#ncap2 -O -s "t2m_in=(s-z-${g}*2)/${cpd}" -o ${filename_remapped} ${filename_remapped}
# finally rename data to distinguish between input and target data (the later must be copied over from previous files)
ncrename -v tp,tp_in  -v z,z_in ${filename_remapped}
#ncks -O -x -v s ${filename_remapped} ${filename_remapped}
ncea -A -d lat,${lat0},${lat1} -d lon,${lon0},${lon1} -v tp,z ${filename_dse} ${filename_remapped}
ncrename -v tp,tp_tar -v z,z_tar ${filename_remapped}

### Return and clean-up in case of success ###
if [[ -f ${filename_remapped} ]]; then
  echo "${scr_name}: Processed data successfully from ${filename} to ${filename_remapped}. Cleaning-up..."
  rm $filename_sd $filename_dse $filename_crs 
  exit 0
else
  echo "${scr_name}: Something went wrong when processing ${filename}. Check intermediate files."
  exit 1
fi

