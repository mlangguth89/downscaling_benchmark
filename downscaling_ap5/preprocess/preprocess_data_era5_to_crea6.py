__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-08-11"
__update__ = "2022-08-11"

# doc-string
"""
Main script to preprocess data for the downscaling task of mapping ERA5-data onto COSMO-REA6.
To harmonize the data, the ERA5 data will be remapped onto a grid which is defined in the same rotated pole coordinates
as the COSMO REA6-data. This has the advantage that the resulting grid is quasi-equidistant.
Since the downscaling factor between the two data sources is about 5 (ERA5 = 30 km, COSMO-REA6 = 6.5 km),
the (intermediate) grid spacing of the remapped ERA5-data is 0.275° (5*dx_CREA6 = 5*0.055°).
Note that the final ERA5-data is bilinearly interpolated on the target grid for the input of U-net based neural network/
generator. 
"""
# doc-string

import os, glob
from typing import Union, List
import subprocess as sp
import logging
import numbers
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
from collections import OrderedDict
#from tfrecords_utils import IFS2TFRecords
from preprocess_data_era5_to_ifs import PreprocessERA5toIFS
from preprocess_data_unet_tier1 import Preprocess_Unet_Tier1, CDOGridDes
from pystager_utils import PyStager
from tools_utils import CDO, NCRENAME, NCAP2, NCKS, NCEA
from other_utils import to_list, last_day_of_month, flatten, remove_files

number = Union[float, int]
num_or_List = Union[number, List[number]]
list_or_tuple = Union[List, tuple]
list_or_dict = Union[List, dict]


class PreprocessERA5toCREA6(PreprocessERA5toIFS):

    # re-instatiate CDO-object to disable extrapolation
    cdo = CDO(tool_envs={"REMAP_EXTRAPOLATE": "on"})
    # hard-coded constants [IFS-specific parameters (from Chapter 12 in http://dx.doi.org/10.21957/efyk72kl)]
    cpd, g = 1004.709, 9.80665
    # invariant variables expected in the invariant files
    const_vars = ["z", "lsm"]

    def __init__(self, in_datadir: str, tar_datadir: str, out_dir: str, constfile_in: str, constfile_tar: str,
                 grid_des_tar: str, predictors: dict, predictands: dict, downscaling_fac: int = 5):
        """
        Initialize class for ERA5-to-COSMO REA6 downscaling class.
        """
        # initialize from ERA5-to-IFS class (parent class)
        super().__init__(in_datadir, tar_datadir, out_dir, constfile_in, grid_des_tar, predictors, predictands,
                         downscaling_fac)

        self.name_preprocess = "preprocess_ERA5_to_CREA6"

        # sanity check on constant/invariant file of COSMO REA6
        if not os.path.isfile(constfile_tar):
            raise FileNotFoundError("Could not find file with invariant data '{0}'.".format(constfile_tar))

    def prepare_worker(self, years: List, season: str, **kwargs):
        """
        Prepare workers for preprocessing.
        :param years: List of years to be processed.
        :param season: Season-string to be processed.
        :param kwargs: Arguments such as jobname for logger-filename
        """
        preprocess_pystager, run_dict = super.prepare_worker(years, season, coarse_grid_name="crea6_", **kwargs)

        # correct (default) job name
        run_dict["kwargs"] = {"job_name": kwargs.get("jobname", "Preproc_ERA5_to_CREA6")}

        return preprocess_pystager, run_dict

    @staticmethod
    def preprocess_worker(year_months: List, dirin_era5: str, dirin_crea6: str, invar_file_era5: str,
                          invar_file_crea6: str, dirout: str, gdes_dict: dict, predictors: dict, predictands: dict,
                          logger: logging.Logger, max_warn: int = 3):
        """
        Function that preprocesses ERA5 (input) - and COSMO REA6 (output)-data on individual workers
        :param year_months: List of Datetime-objects indicating year and month for which data should be preprocessed
        :param dirin_era5: input directory of ERA5-dataset (top-level directory)
        :param dirin_crea6: input directory of COSMO REA6-data (top-level directory)
        :param invar_file_era5: data file providing invariant variables of ERA5 dataset
        :param invar_file_crea6: data file providing invariant variables of COSMO REA6-dataset
        :param dirout: output directoty to store preprocessed data
        :param predictors: nested dictionary of predictors, where the first-level key denotes the variable type,
                           and the second-level key-value pairs denote the variable as well as interpolation info
                           Example: { "sf": {"2t", "blh"}, "ml_fc": { "t", ["p85000", "p925000"]}}
        :param predictands: Similar to predictors, but with different convention for variable tyoe
                            Example: {"2D": {"t_2m"}, "const": {"hsurf"}}
        :param gdes_dict: dictionary containing grid description dictionaries for target and coarse grid
        :param logger: Logging instance for log process on worker
        :param max_warn: allowed maximum number of warnings/problems met during processing (default:3)
        :return: -
        """
        cdo = PreprocessERA5toCREA6.cdo
        # sanity checks
        assert isinstance(logger, logging.Logger), "logger-argument must be a logging.Logger instance"
        if not os.path.isfile(invar_file_era5):
            raise FileNotFoundError("File providing invariant data of ERA5-dataset'{0}' cannot be found."
                                    .format(invar_file_era5))

        if not os.path.isfile(invar_file_crea6):
            raise FileNotFoundError("File providing invariant data of COSMO REA6-dataset'{0}' cannot be found."
                                    .format(invar_file_crea6))

        sfvars_era5, mlvars_era5, fc_sfvars_era5, fc_mlvars_era5 = PreprocessERA5toIFS.organize_predictors(predictors)
        sfvars_crea6, const_vars_crea6 = PreprocessERA5toCREA6.organize_predictands(predictands)

        grid_des_tar, grid_des_coarse = gdes_dict["tar_grid_des"], gdes_dict["coa_grid_des"]

        for year_month in year_months:
            assert isinstance(year_month, dt.datetime),\
                "All year_months-argument must be a datetime-object. Current one is of type '{0}'"\
                .format(type(year_month))

            year, month = int(year_month.strftime("%Y")), int(year_month.strftime("%m"))
            year_str, month_str = str(year), "{0:02d}".format(int(month))
            last_day = last_day_of_month(year_month)

            subdir = year_month.strftime("%Y-%m")
            dir_curr_era5 = os.path.join(dirin_era5, year_str, month_str)
            _ = PreprocessERA5toCREA6.check_crea6_files(dirin_crea6, subdir, sfvars_era5, const_vars_crea6)
            dest_dir = os.path.join(dirout, "netcdf_data", year_str, subdir)
            final_file = os.path.join(dest_dir, "preproc_{0}.nc".format(subdir))
            os.makedirs(dest_dir, exist_ok=True)

            # further sanity checks
            if not os.path.isdir(dir_curr_era5):
                err_mess = "%{0}: Could not find directory for ERA5-data '{1}'".format(method, dir_curr_era5)
                logger.fatal(err_mess)
                raise NotADirectoryError(err_mess)

            if not os.path.isdir(dir_curr_ifs):
                err_mess = "%{0}: Could not find directory for IFS-data '{1}'".format(method, dir_curr_ifs)
                logger.fatal(err_mess)
                raise NotADirectoryError(err_mess)

            dates2op = pd.date_range(dt.datetime.strptime("{0}{1}0100".format(year_str, month_str), "%Y%m%d%H"),
                                     last_day, freq="H")

            # Perform logging, reset warning counter and loop over dates...
            logger.info("Start preprocessing data for month {0}...".format(subdir))
            nwarn = 0
            for date2op in dates2op:
                # !!!!!! ML: Preliminary fix to avoid processing data from 2015 !!!!!!
                if date2op <= dt.datetime.strptime("20160101 12", "%Y%m%d %H"): continue
                date_str = date2op.strftime("%Y%m%d%H")
                daily_file = os.path.join(dest_dir, "{}_preproc.nc".format(date_str))

                filelist, nwarn = PreprocessERA5toIFS.preprocess_era5_in(dir_curr_era5, invar_file_era5, dest_dir, date2op,
                                                                         grid_des_coarse, grid_des_tar, sfvars_era5,
                                                                         mlvars_era5, fc_sfvars_era5, fc_mlvars_era5,
                                                                         logger, nwarn, max_warn)

                # finally all temporary files for each time step and clean-up
                logger.info("Merge temporary files to daily netCDF-file '{0}'".format(daily_file))
                cdo.run(filelist + [daily_file], OrderedDict([("merge", "")]))
                remove_files(filelist, lbreak=False)

            # merge all time steps to monthly file and clean-up daily files
            logger.info("Merge all daily files to monthly datafile '{0}'".format(final_file))
            all_daily_files = glob.glob(os.path.join(dest_dir, "*_preproc.nc"))
            cdo.run(all_daily_files + [final_file], OrderedDict([("mergetime", "")]))
            remove_files(all_daily_files, lbreak=True)

            # process COSMO-REA6 doata which is already organized in monthly files
            PreprocessERA5toCREA6.preprocess_crea6_tar(dirin_crea6, invar_file_crea6, dest_dir, subdir,
                                                       sfvars_crea6, const_vars_crea6, logger, nwarn, max_warn)


        return nwarn


    @staticmethod
    def organize_predictands(predictands: dict) -> (List, List):
        """
        Organizes predictands from COSMO-REA6 dataset. Currently, only 2D variables and invariant data are supported.
        !!! To-Do !!!
        3D variables incl. interpolation (on pressure-levels) has to be integrated
        !!! To-Do !!
        :param predictands: dictionary for predictands with the form {"2D", {"t_2m"}}
        """

        known_vartypes = ["2D", "const"]

        pred_vartypes = list(predictands.keys())
        lpred_vartypes = [pred_vartype in known_vartypes for pred_vartype in pred_vartypes]
        if not all(lpred_vartypes):
            unknown_vartypes = [pred_vartypes[i] for i, flag in enumerate(lpred_vartypes) if not flag]
            raise ValueError("The following variables types in the predictands-dictionary are not supported: {0}"
                             .format(", ".join(unknown_vartypes)))

        vars_2d, vars_const = predictands.get("2D", None), predictands.get("const", None)
        vars_2d = [var_2d.capitalize() for var_2d in vars_2d]

        return vars_2d, vars_const

    @staticmethod
    def check_crea6_files(indir: str, const_file: str, yr_month: str, vars_2d: List, const_vars: List):
        """
        Checks if required predictands are available from the COSMO-REA6 dataset.
        Data is expected to live in monthly netCDF-files within subdirectories named as the variable of interest.
        For invariant data, the information is mandatory in the invariant file of the COSMO-REA6 dataset.
        :param indir: Directory under which COSMO-REA6 data is stored
        :param const_file: path to invariant datafile
        :param yr_month: Date-string for which data is required; format YYYY-MM
        :param vars_2d: list of 2D-variables serving as predictands
        :param const_vars: list of invariant variables serving as predictands (must be part of const_file)
        """
        for var_2d in vars_2d:
            var_2d = var_2d.capitalize()
            dfile_2d = os.path.join(indir, "2D", var_2d, f"{var_2d}.2D.{yr_month}")
            if not os.path.isfile(dfile_2d):
                FileNotFoundError(f"Could not find required file '{dfile_2d}' for predictand variable '{dfile_2d}'")

        if const_vars:
            fconsts = xr.open_dataset(const_file)
            varlist = list(fconsts.keys())
            stat = [cvar in varlist for cvar in const_vars]
            if not all(stat):
                miss_inds = np.where(not stat)
                raise ValueError("The following variables cannot be found in the constant file '{0}': {1}"
                                 .format(const_file, ",".join(list(np.array(const_vars)[miss_inds]))))

        return True


    @staticmethod
    def preprocess_crea6_tar(dirin: str, invar_file: str, fgdes_tar: str, dest_dir: str, date2op: dt.datetime,
                             vars_2d: List, const_vars: List, logger:logging.Logger, nwarn: int, max_warn):
        """
        Process IFS-file by slicing data to region of interest.
        :param dirin_ifs: top-level directory where IFS-data are placed (under <year>/<year>-<month>/-subdirectories)
        :param target_dir: Target directory to store the processed data in netCDF-files
        :param date2op: Date for which data should be processed
        :param fgdes_tar: grid description file for target (high-resolved) grid
        :param predictands: dictionary for predictand variables
        :return: path to processed netCDF-datafile
        """
        cdo, ncrename = PreprocessERA5toCREA6.cdo, PreprocessERA5toCREA6.ncrename

        date_str, date_str2 = date2op.strftime("%Y-%m"), date2op.strftime("%Y%m")
        tmp_dir = os.path.join(dest_dir, "tmp_{0}".format(date_str))

        gdes_tar = CDOGridDes(fgdes_tar)
        gdes_dict = gdes_tar.grid_des_dict

        lonlatbox = (*gdes_tar.get_slice_coords(gdes_dict["xfirst"], gdes_dict["xinc"], gdes_dict["xsize"]),
                     *gdes_tar.get_slice_coords(gdes_dict["yfirst"], gdes_dict["yinc"], gdes_dict["ysize"]))
        lonlatbox_str = ",".join("{:.3f}".format(coord) for coord in lonlatbox)

        # process 2D-files
        if vars_2d:
            for var in vars_2d:    # TBD: Put the following into a callable object to accumulate nwarn and filelist
                dfile_in = os.path.join(dirin, "2D", var.capitilize(), f"{var.capitilize()}.2D.{date_str2}.grb")
                dfile_out = os.path.join(tmp_dir, var.lowercase())
                if not os.path.isfile(dfile_in):
                    FileNotFoundError(f"Could not find required COSMO-REA6 file '{dfile_in}'.")

                cdo.run([dfile_in, dfile_out], OrderedDict([("-f nc", ""), ("copy", ""),
                                                            ("-sellonlatbox", lonlatbox_str)]))

                # rename varibale in resulting file (must be done in hacky manner)
                varname = sp.check_output(f"cdo showname {dfile_out}", shell=True)
                varname = varname.lstrip("'b").split("\\n")[0].strip()

                ncrename.run([dfile_out], OrderedDict([("-v", f"{varname},{var}")]))

        if const_vars:
            pass    # TBD


    @staticmethod
    def process_2d_file(file_2d: str, target_dir, date2op, fgdes_tar: str):




        cdo = PreprocessERA5toIFS.cdo

        # handle date and create tmp-directory and -files
        date_str = date2op.strftime("%Y%m%d%H")
        ifs_file,fh = PreprocessERA5toIFS.get_fc_file(dirin_ifs, date2op, model="ifs", suffix="sfc")
        tmp_dir = os.path.join(target_dir, "tmp_{0}".format(date_str))
        os.makedirs(tmp_dir, exist_ok=True)

        ftmp_hres = os.path.join(tmp_dir, "{0}_tar.nc".format(date_str))

        # get variables to retrieve from predictands-dictionary
        # ! TO-DO: Allow for variables given on pressure levels (in pl-files!) !
        if any(vartype != "sf" for vartype in predictands.keys()):
            raise ValueError("Only surface variables (i.e. vartype 'sf') are currently supported for IFS data.")
        ifsvars = list(predictands["sf"].keys())

        # get slicing coordinates from target grid description file
        gdes_tar = CDOGridDes(fgdes_tar)
        gdes_dict = gdes_tar.grid_des_dict

        lonlatbox = (*gdes_tar.get_slice_coords(gdes_dict["xfirst"], gdes_dict["xinc"], gdes_dict["xsize"]),
                     *gdes_tar.get_slice_coords(gdes_dict["yfirst"], gdes_dict["yinc"], gdes_dict["ysize"]))

        cdo.run([ifs_file, ftmp_hres],
                OrderedDict([("-seltimestep", "{0:d}".format(fh)), ("-selname", ",".join(ifsvars)),
                             ("-sellonlatbox", "{0:.2f},{1:.2f},{2:.2f},{3:.2f}".format(*lonlatbox))]))

        # rename variables
        PreprocessERA5toIFS.add_varname_suffix(ftmp_hres, ifsvars, "_tar")

        return ftmp_hres

