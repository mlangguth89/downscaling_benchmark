#!/bin/bash

##############################################################################
# Script to process the ERA5 and COSMO REA6 data that has also been used     #
# in the AtMoRep project. To be processable for the data loader in MAELSTROM,#
# the input and target data must be provided in monthly netCDF files.        #
# Furthermore, input and target variables must be denoted with _in and _tar, #
# respectively.                                                              #
##############################################################################

# author: Michael Langguth
# date: 2023-08-16
# update: 2023-08-16

# parameters

era5_basedir=/p/scratch/atmo-rep/data/era5/ml_levels/
crea6_basedir=/p/scratch/atmo-rep/data/cosmo_rea6/ml_levels/
output_dir=/p/scratch/atmo-rep/data/era5/ml_levels/
crea6_gdes=../grid_des/crea6_reg_grid

era5_vars=("t")
era5_vars_full=("temperature")
crea6_vars=("t2m")

ml_lvl_era5=( 96 105 114 123 135 )
ml_lvl_crea6=( 0 )

year_start=1995
year_end=2018

# main

echo "Loading required modules..."
ml purge
ml Stages/2022 GCC/11.2.0  OpenMPI/4.1.2 NCO/5.0.3 CDO/2.0.2

tmp_dir=${output_dir}/tmp

# create output directory
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

if [ ! -d $tmp_dir ]; then
    mkdir -p $tmp_dir
fi

# loop over years and months
for yr in $(eval echo "{$year_start..$year_end}"); do 
    for mm in {01..12}; do
        echo "Processing ERA5 data for ${yr}-${mm}..."
        for ivar in "${!era5_vars[@]}"; do 
            for ml_lvl in ${ml_lvl_era5}; do 
                # get file name
                era5_file=${era5_basedir}/${ml_lvl}/"${era5_vars_full[ivar]}"/reanalysis_"${era5_vars_full[ivar]}"_y${yr}_m${mm}_ml${mlvl}.grib
                tmp_file1=${tmp_dir}/era5_${era5_vars[ivar]}_y${yr}_m${mm}_ml${ml_lvl}_tmp1.nc 
                tmp_file2=${tmp_dir}/era5_${era5_vars[ivar]}_y${yr}_m${mm}_ml${ml_lvl}_tmp2.nc
                tmp_era5=${tmp_dir}/era5_${era5_vars[ivar]}_y${yr}_m${mm}_ml${ml_lvl}_tmp3.nc
                # convert to netCDF and slice to region of interest
                cdo -f nc copy -sellonlatbox,-1.5,25.75,42.25,56 ${era5_file} ${tmp_file1}
                cdo remapbil,./crea6_reg_grid ${tmp_file1} ${tmp_file2}
                # rename variable
                cdo --reduce_dim chname,${era5_vars[ivar]},${era5_vars[ivar]}_ml${ml_lvl}_in -selvar,${era5_vars[ivar]} ${tmp_file2} ${tmp_era5} 
                # clean-up
                rm ${tmp_file1} ${tmp_file2}
                break
            done
        done

        echo "Processing COSMO-REA6 data for ${yr}-${mm}..."
        for crea6_var in ${crea6_vars[@]}; do 
            for ml_lvl in ${ml_lvl_crea6}; do
                crea6_file=${crea6_basedir}/${ml_lvl}/${crea6_var}/cosmo_rea6_${crea6_var}_y${yr}_m${mm}_ml${mlvl}.nc
                tmp_crea6=${tmp_dir}/crea6_${crea6_var}_y${yr}_m${mm}_ml${ml_lvl}_tmp.nc
                cdo --reduce_dim -chname,${crea6_var},${crea6_var}_ml${ml_lvl}_tar -sellonlatbox,-1.25,25.6875,42.3125,55.75 -selvar,${crea6_var} ${crea6_file} ${tmp_crea6}
            done
        done

        cdo merge ${tmp_era5} ${tmp_crea6} ${output_dir}/downscaling_atmorep_train_${yr}-${mm}.nc;
    done
done






