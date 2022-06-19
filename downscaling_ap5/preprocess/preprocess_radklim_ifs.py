#!/usr/bin/env python
# coding: utf-8

# import modules
import os, sys 
import glob
sys.path.append("../utils")
import xarray as xr
import pandas as pd
import datetime as dt
from collections import OrderedDict
from typing import List
from tools_utils import CDO, NCRENAME
from other_utils import last_day_of_month


# auxiliary function
def add_varname_suffix(nc_file: str, varnames: List, suffix: str):
    """
    Rename variables in netCDF-file by adding a suffix
    :param nc_file: netCDF-file to process
    :param varnames: (old) variable names to modify
    :param suffix: suffix to add to variable names
    :return: status-flag
    """
    varnames_new = [varname + suffix for varname in varnames]
    varnames_pair = ["{0},{1}".format(varnames[i], varnames_new[i]) for i in range(len(varnames))]

    try:
        ncrename.run([nc_file], OrderedDict([("-v", varnames_pair)]))
        stat = True
    except RuntimeError as err:
        print("Could not rename all parsed variables: {0}".format(",".join(varnames)))
        raise err

    return stat

# main

# data-directories and parameters
ifs_dir = "/p/scratch/deepacf/maelstrom/maelstrom_data/ifs_hres/orig"
radklim_dir = "/p/scratch/deepacf/deeprain/radklim_process/netcdf/remapped/"

outdir_base = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_ifs_radklim"

sf_vars = ["cp", "lsp", "cape", "tclw", "tcwv", "sp","tisr"]
pl_vars = ["u", "v"]#{"u": {"pl": 70000.}, "v": {"pl": 70000.}}
radklim_var = "YW_hourly"

years = [2016]

lonlat_box = [8.0, 9.6, 50.001, 51.201]

# some parameter processing
ll_box_str = "{0:.3f},{1:.3f},{2:.3f},{3:.3f}".format(*lonlat_box)
rd_var_lower = radklim_var.lower()

# initialize tools
cdo, ncrename = CDO(), NCRENAME()

for yr in years:
    for mm in range(1, 13):
        print("Start processing data for {0:d}-{1:02d}...".format(yr, mm))
        # create output directory incl. tmp
        outdir = os.path.join(outdir_base, "{0:d}-{1:02d}".format(yr, mm))
        outdir_tmp = os.path.join(outdir, "tmp")
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(outdir_tmp, exist_ok=True)
        
        # get current IFS forecast directory
        dirr_curr_ifs = os.path.join(ifs_dir, "{0:d}/{0:d}-{1:02d}".format(yr, mm))
        
        # construct time-arrays for which data is required
        ld = last_day_of_month(dt.datetime.strptime("{0:d}{1:02d}01".format(yr, mm), "%Y%m%d"))
        init_runs = pd.date_range("{0:d}-{1:02d}-01 00:00".format(yr, mm),
                                  ld.replace(hour=12), freq="12H")
        radklim_dates = pd.date_range(init_runs[0] + dt.timedelta(hours=6), init_runs[-1] + dt.timedelta(hours=17),
                                      freq="1H")
        # loop over all IFS forecast-files
        print("Start processing IFS-data for {0:d}-{1:02d}...".format(yr, mm))
        for ir in init_runs:
            print(ir)
            ifs_sf_file = os.path.join(dirr_curr_ifs, "sfc_{0}.nc".format(ir.strftime("%Y%m%d_%H")))
            sf_file_out = os.path.join(outdir_tmp, "sfc_{0}.nc".format(ir.strftime("%Y%m%d%H")))
            
            ifs_pl_file, pl_file_out = ifs_sf_file.replace("sfc", "pl"), sf_file_out.replace("sfc", "pl")

            cdo.run([ifs_sf_file, sf_file_out], OrderedDict([("-selname", ",".join(sf_vars)),
                                                             ("-sellonlatbox", ll_box_str), ("-seltimestep", "6/17")]))
            cdo.run([ifs_pl_file, pl_file_out], OrderedDict([("--reduce_dim", ""), ("-selname", ",".join(pl_vars)),
                                                             ("-sellonlatbox", ll_box_str), ("-seltimestep", "6/17"),
                                                             ("-sellevel", "700")]))
        
        # merge IFS forecast-files 
        all_sf_files, all_pl_files = glob.glob(os.path.join(outdir_tmp, "sfc_*.nc")), glob.glob(os.path.join(outdir_tmp, "pl_*.nc"))
        sf_file = os.path.join(outdir, "sfc_{0}.nc".format(ir.strftime("%Y-%m")))
        pl_file = sf_file.replace("sfc_", "pl_")

        cdo.run(all_sf_files + [sf_file], OrderedDict([("mergetime", "")]))
        cdo.run(all_pl_files + [pl_file], OrderedDict([("mergetime", "")]))

        # rename variables
        add_varname_suffix(sf_file, sf_vars, "_in")
        add_varname_suffix(pl_file, pl_vars, "_in")
        ncrename.run([pl_file], OrderedDict([("-v", ["u_in,u700_in", "v_in,v700_in"])]))
        # loop over all corresponding RADKLIM timesteps
        print("Start processing RADKLIM-data for {0:d}-{1:02d}...".format(yr, mm))
        for rd in radklim_dates:
            print(rd)
            yr_now, mm_now, dd_now, hh_now = rd.strftime("%Y"), rd.strftime("%m"), rd.strftime("%d"), rd.strftime("%H")
            if hh_now == "00":
                rd_file = rd - dt.timedelta(days=1)
            else:
                rd_file = rd

            curr_file_lres = os.path.join(radklim_dir, "radklim_reg_lres", rd_var_lower, rd_file.strftime("%Y"),
                                          rd_file.strftime("%Y-%m"), "{0}_remapped_radklim_reg_lres_{1}.nc"
                                          .format(radklim_var, rd_file.strftime("%Y-%m-%d")))
            file_lres_out = os.path.join(outdir_tmp, "radklim_lres_{0}.nc".format(rd.strftime("%Y%m%d%H")))
            
            curr_file_hres = curr_file_lres.replace("radklim_reg_lres", "radklim_reg_hres")
            file_hres_out = file_lres_out.replace("_lres_", "_hres_")
            
            rd_dict_cdo = OrderedDict([("-L", ""), ("-seldate", rd.strftime("%Y-%m-%dT%H:00:00")), ("-selname", rd_var_lower),
                                       ("-sellonlatbox", ll_box_str)])
            cdo.run([curr_file_lres, file_lres_out], rd_dict_cdo)
            cdo.run([curr_file_hres, file_hres_out], rd_dict_cdo)
            
        # merge all RADKLIM files  
        all_rd_lres_files = glob.glob(os.path.join(outdir_tmp, "radklim_lres_*.nc"))
        all_rd_hres_files = glob.glob(os.path.join(outdir_tmp, "radklim_hres_*.nc"))
        
        rd_lres_file = os.path.join(outdir, "radklim_lres_{0}.nc".format(ir.strftime("%Y-%m")))
        rd_hres_file = rd_lres_file.replace("_lres_", "_hres_")
        
        cdo.run(all_rd_lres_files + [rd_lres_file], OrderedDict([("mergetime", "")]))
        cdo.run(all_rd_hres_files + [rd_hres_file], OrderedDict([("mergetime", "")]))

        # rename variables
        add_varname_suffix(rd_lres_file, [radklim_var], "_in")
        add_varname_suffix(rd_hres_file, [radklim_var], "_tar")
                                          
        
