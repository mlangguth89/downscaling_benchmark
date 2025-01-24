# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2024-01-16"
__update__ = "2024-01-22"

# doc-string
"""
Main script to preprocess data for the downscaling task of mapping ERA5-data onto COSMO-REA6.
To harmonize the data, the ERA5 data will be remapped onto a grid which is defined in the same rotated pole coordinates
as the COSMO REA6-data. This has the advantage that the resulting grid is quasi-equidistant.
Since the downscaling factor between the two data sources is about 4 (ERA5 = 25 km, COSMO-REA6 = 6.5 km),
the (intermediate) grid spacing of the remapped ERA5-data is 0.225° (4*dx_CREA6 = 4*0.055°).
Note that the final ERA5-data is bilinearly interpolated on the target grid for the input of U-net based neural network/
generator. 
Furthermore, the script has been revised to process the data provided by main_download_era5.py which directly retireves the
data through the CDS API instead of using the data from the meteocloud. These files contain surface and model level data.
"""
# doc-string

import os
from typing import Union, List, Tuple
import subprocess as sp
import logging
import datetime as dt
import numpy as np
import xarray as xr
from collections import OrderedDict
# from tfrecords_utils import IFS2TFRecords
from abstract_preprocess import CDOGridDes
from preprocess_data_era5_to_ifs import PreprocessERA5toIFS
from tools_utils import CDO, NCRENAME, NCAP2, NCKS, NCEA
from other_utils import to_list, remove_files

number = Union[float, int]
num_or_List = Union[number, List[number]]
list_or_tuple = Union[List, tuple]
list_or_dict = Union[List, dict]


class PreprocessERA5toCREA6(PreprocessERA5toIFS):

    # re-instatiate CDO-object to disable extrapolation
    cdo, ncrename, ncap2, ncks, ncea = CDO(tool_envs={"REMAP_EXTRAPOLATE": "on"}), NCRENAME(), NCAP2(), NCKS(), NCEA()
    # hard-coded constants [IFS-specific parameters (from Chapter 12 in http://dx.doi.org/10.21957/efyk72kl)]
    cpd, g = 1004.709, 9.80665
    # invariant variables expected in the invariant files
    const_vars = ["z", "lsm"]

    def __init__(self, in_datadir: str, tar_datadir: str, out_dir: str, in_constfile: str, tar_constfile: str,
                 grid_des_tar: str, predictors: dict, predictands: dict, upscale_source: bool = True, downscaling_fac: int = 4):
        """
        Initialize class for ERA5-to-COSMO REA6 downscaling class.
        """
        # initialize from ERA5-to-IFS class (parent class)
        super().__init__(in_datadir, tar_datadir, out_dir, in_constfile, grid_des_tar, predictors, predictands, upscale_source,
                         downscaling_fac)

        self.name_preprocess = "preprocess_ERA5_to_CREA6"

        # sanity check on constant/invariant file of COSMO REA6
        if not os.path.isfile(tar_constfile):
            raise FileNotFoundError("Could not find file with invariant data '{0}'.".format(tar_constfile))
        self.constfile_tar = tar_constfile
        self.era5_sfc_vars, self.era5_ml_vars, self.era5_pl_vars = self.organize_predictors(predictors)
        self.crea6_sfc_vars, self.crea6_const_vars = self.organize_predictands(predictands)
        
        self.all_predictors = self.get_predictor_varnames(predictors)     # mlvars_era5 is a dictionary
        self.all_predictands = [e for e in self.crea6_sfc_vars + self.crea6_const_vars if e]


    def prepare_worker(self, years: List, season: str, **kwargs):
        """
        Prepare workers for preprocessing.
        :param years: List of years to be processed.
        :param season: Season-string to be processed.
        :param kwargs: Arguments such as jobname for logger-filename
        """
        preprocess_pystager, run_dict = super().prepare_worker(years, season, coarse_grid_name="crea6_", **kwargs)

        # correct (default) job name...
        run_dict["kwargs"] = {"job_name": kwargs.get("jobname", "Preproc_ERA5_to_CREA6")}
        # ... and update args-list to fit to call of preprocess_worker
        run_dict["args"].insert(3, self.constfile_tar)

        return preprocess_pystager, run_dict

    def preprocess_worker(self, year_months: List, dirin_era5: str, dirin_crea6: str, invar_file_era5: str,
                          invar_file_crea6: str, dirout: str, gdes_dict: dict, logger: logging.Logger, max_warn: int = 3):
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
                           Example: { "sf": {"2t", "blh"}, "ml: { "t", [137, 135]}}
        :param predictands: Similar to predictors, but with different convention for variable tyoe
                            Example: {"2D": {"t_2m"}, "const": {"hsurf"}}
        :param gdes_dict: dictionary containing grid description dictionaries for target and coarse grid
        :param logger: Logging instance for log process on worker
        :param max_warn: allowed maximum number of warnings/problems met during processing (default:3)
        :return: -
        """

        # sanity checks
        assert isinstance(logger, logging.Logger), "logger-argument must be a logging.Logger instance"
        if not os.path.isfile(invar_file_era5):
            raise FileNotFoundError("File providing invariant data of ERA5-dataset'{0}' cannot be found."
                                    .format(invar_file_era5))

        if not os.path.isfile(invar_file_crea6):
            raise FileNotFoundError("File providing invariant data of COSMO REA6-dataset'{0}' cannot be found."
                                    .format(invar_file_crea6))

        grid_des_tar, grid_des_coarse = gdes_dict["tar_grid_des"], gdes_dict["coa_grid_des"]

        # initialize number of warnings
        nwarn = 0
        for year_month in year_months:
            assert isinstance(year_month, dt.datetime),\
                "All year_months-argument must be a datetime-object. Current one is of type '{0}'"\
                .format(type(year_month))

            final_file = os.path.join(dirout, f"preproc_era5_crea6_{year_month.strftime('%Y-%m')}.nc")

            _ = self.check_crea6_files(dirin_crea6, year_month)
            # create temp-directory 
            outdir_tmp = os.path.join(dirout, f"{year_month.strftime('%Y-%m')}_tmp")
            # set-up temporary directory
            os.makedirs(outdir_tmp, exist_ok=True)

            final_file_era5, lfail, nwarn = self.preprocess_era5_in(dirin_era5, year_month, invar_file_era5, 
                                                                    outdir_tmp, logger, nwarn, max_warn)

            if lfail: continue       # skip month if preprocessing ERA5-data failed

            # process COSMO-REA6 doata which is already organized in monthly files
            final_file_crea6, nwarn = self.preprocess_crea6_tar(dirin_crea6, invar_file_crea6, grid_des_tar, outdir_tmp,
                                                                year_month, logger, nwarn, max_warn)

            # finally merge the ERA5- and the COSMO REA6-data
            self.remap_and_merge_data(final_file_era5, final_file_crea6, final_file, grid_des_coarse,
                                      grid_des_tar, nwarn, max_warn)

            # rename input-variables
            self.add_varname_suffix(final_file, self.all_predictors, "_in")

        return nwarn

    @staticmethod
    def organize_predictors(predictors: dict) -> Tuple[List, dict]:
        known_vartypes = ["sf", "ml", "pl"]

        pred_vartypes = list(predictors.keys())
        lpred_vartypes = [pred_vartype in known_vartypes for pred_vartype in pred_vartypes]
        if not all(lpred_vartypes):
            unknown_vartypes = [pred_vartypes[i] for i, flag in enumerate(lpred_vartypes) if not flag]
            raise ValueError("The following variables types in the predictor-dictionary are unknown: {0}"
                             .format(", ".join(unknown_vartypes)))

        sfvars, mlvars, plvars = predictors.get("sf", None), predictors.get("ml", None), predictors.get("pl", None)

        # some checks (level information redundant for surface-variables)
        if sfvars:
            try:
                sfvars = list(sfvars)
                assert all([isinstance(sfvar, str) for sfvar in sfvars]), f"List of surface variables must be strings."
            except Exception as err:
                raise TypeError(f"Surface-variables cannot be parsed into a list.")

        if mlvars:
            # mlvars must be a dictionary with variable names as keys and model-levels as values
            if not isinstance(mlvars, dict):
                raise TypeError("mlvars must be a dictionary with variable names as keys and model-levels as values.")
            # check if all keys are strings
            if not all([isinstance(k, str) for k in mlvars.keys()]):
                raise TypeError("All keys of mlvars must be strings.")
            # check if all values are lists
            if not all([isinstance(v, list) for v in mlvars.values()]):
                raise TypeError("All values of mlvars must be lists.")
            # check if all values are list of numbers
            if not all([all([isinstance(v, (int, float)) for v in mlvars[k]]) for k in mlvars.keys()]):
                raise TypeError("All values of mlvars must be lists of numbers.")

        if plvars:
            # plvars must be a dictionary with variable names as keys and model-levels as values
            if not isinstance(plvars, dict):
                raise TypeError("plvars must be a dictionary with variable names as keys and model-levels as values.")
            # check if all keys are strings
            if not all([isinstance(k, str) for k in plvars.keys()]):
                raise TypeError("All keys of plvars must be strings.")
            # check if all values are lists
            if not all([isinstance(v, list) for v in plvars.values()]):
                raise TypeError("All values of plvars must be lists.")
            # check if all values are list of numbers
            if not all([all([isinstance(v, (int, float)) for v in plvars[k]]) for k in plvars.keys()]):
                raise TypeError("All values of plvars must be lists of numbers.")            

        return sfvars, mlvars, plvars

    @staticmethod
    def organize_predictands(predictands: dict) -> Tuple[List, List]:
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
        vars_2d = [var_2d.upper() for var_2d in vars_2d]
        vars_const = [var_const.upper() for var_const in vars_const]

        return vars_2d, vars_const

    def check_crea6_files(self, indir: str, yr_month: str):
        """
        Checks if required predictands are available from the COSMO-REA6 dataset.
        Data is expected to live in monthly netCDF-files within subdirectories named as the variable of interest.
        For invariant data, the information is mandatory in the invariant file of the COSMO-REA6 dataset.
        :param indir: Directory under which COSMO-REA6 data is stored
        :param yr_month: Date-string for which data is required; format YYYY-MM
        """
        if self.crea6_sfc_vars:
            for var_2d in self.crea6_sfc_vars:
                var_2d = var_2d.capitalize()
                dfile_2d = os.path.join(indir, "2D", var_2d, f"{var_2d}.2D.{yr_month}")
                if not os.path.isfile(dfile_2d):
                    FileNotFoundError(f"Could not find required file '{dfile_2d}' for predictand variable '{dfile_2d}'")

        if self.crea6_const_vars:
            fconsts = xr.open_dataset(self.constfile_tar)
            varlist = list(fconsts.keys())
            stat = [cvar in varlist for cvar in self.crea6_const_vars]
            if not all(stat):
                miss_inds = np.where(not stat)
                raise ValueError("The following variables cannot be found in the constant file '{0}': {1}"
                                 .format(self.constfile_tar, ",".join(list(np.array(self.crea6_const_vars)[miss_inds]))))

        return True
    
    def preprocess_era5_in(self, era5_dir: str, year_month, invar_file: str, dest_dir: str, logger: logging.Logger, nwarn: int, max_warn: int):
        """
        Retrieve the predictor data from the monthly ERA5-datafiles.
        """
        year_month_str = year_month.strftime("%Y-%m")   

        lfail = False
        filelist = []
        logger.info(f"Start preprocessing ERA5-data for {year_month_str}.")

        # process surface variables of ERA5 (predictors)
        if self.era5_sfc_vars:
            logger.info(f"Process surface variables for {year_month_str} of ERA5.")
            nwarn, file2merge = self.run_preproc_func(self.process_era5_sf, [era5_dir, year_month, invar_file, dest_dir],
                                                      {}, logger, nwarn, max_warn)
            if file2merge:
                filelist.append(file2merge)
            else:
                lfail = True   # skip month if data is missing

        # process multi-level variables of ERA5 (predictors)
        if self.era5_ml_vars and not lfail:
            logger.info(f"Process model level variables for {year_month_str} of ERA5.")
            nwarn, file2merge = self.run_preproc_func(self.process_era5_vl, [era5_dir, year_month, dest_dir, "ml"],
                                                      {}, logger, nwarn, max_warn)
            
            if file2merge:
                filelist.append(file2merge)
            else:
                lfail = True   # skip month if some data is missing

        # process pressure-level variables of ERA5 (predictors)
        if self.era5_pl_vars and not lfail:
            logger.info(f"Process pressure level variables for {year_month_str} of ERA5.")
            nwarn, file2merge = self.run_preproc_func(self.process_era5_vl, [era5_dir, year_month, dest_dir, "pl"],
                                                      {}, logger, nwarn, max_warn)
            if file2merge:
                filelist.append(file2merge)
            else:
                lfail = True   # skip month if some data is missing

        monthly_file = os.path.join(dest_dir, f"preproc_era5_{year_month_str}.nc")

        if filelist and not lfail:
            if len(filelist) == 1:
                # just rename file
                logger.info("Rename temporary ERA5-file to monthly netCDF-file '{0}'".format(monthly_file))
                os.rename(filelist[0], monthly_file)
                filelist.pop(0)
            else:
                logger.info("Merge temporary ERA5-files to hourly netCDF-file '{0}'".format(monthly_file))
                _ = self.merge_multiple_netcdf(filelist, monthly_file)
        
        if os.path.isfile(monthly_file):
            remove_files(filelist, lbreak=True)
        else:
            lfail = True

        return monthly_file, lfail, nwarn

    def preprocess_crea6_tar(self, dirin: str, invar_file: str, fgdes_tar: str, dest_dir: str, date2op: dt.datetime,
                             logger: logging.Logger, nwarn: int, max_warn):
        """
        Process COSMO REA6-files based on requested 2D- and invariant variables.
        :param dirin: top-level directory where COSMO REA6-data are placed (under <year>/<year>-<month>/-subdirectories)
        :param invar_file: datafile providing invariant COSMO REA6-data, e.g. HSURF
        :param fgdes_tar: file to CDO grid description file of target data
        :param dest_dir: Target directory to store the processed data in netCDF-files
        :param date2op: Date for which data should be processed
        :param logger: logging-instance
        :param nwarn: number of faced warnings in processing chain (will be updated here)
        :param max_warn: maximum number of allowd warnings
        :return: path to processed netCDF-datafile and updated number of warnings
        """
        cdo  = PreprocessERA5toCREA6.cdo

        date_str, date_str2 = date2op.strftime("%Y-%m"), date2op.strftime("%Y%m")
        final_file = os.path.join(dest_dir, f"preproc_crea6_{date_str}.nc")
        if os.path.isfile(final_file):
            logger.info("Monthly COSMO REA6-file '{0}' already exists. Ensure that data is as expected.".format(final_file))
            return final_file, nwarn

        gdes_tar = CDOGridDes(fgdes_tar)

        filelist = []

        lfail = False

        # process 2D-files
        if self.crea6_sfc_vars:
            for var in self.crea6_sfc_vars:    # TBD: Put the following into a callable object to accumulate nwarn and filelist
                dfile_in = os.path.join(dirin, "2D", var.upper(), f"{var.upper()}.2D.{date_str2}.grb")
                ## check if grib- or netcdf-file is available
                #if not os.path.isfile(dfile_in):
                #    dfile_in = dfile_in.replace(".grb", ".nc")

                nwarn, file2merge = self.run_preproc_func(self.process_crea6_2d, [dfile_in, dest_dir, date_str, gdes_tar],
                                                          {}, logger, nwarn, max_warn)

                if not file2merge:
                    lfail = True
                else:
                    filelist = self.manage_filemerge(filelist, file2merge, dest_dir)

        if self.crea6_const_vars and not lfail:
            nwarn, file2merge = self.run_preproc_func(self.process_crea6_const, [dest_dir, gdes_tar],
                                                      {}, logger, nwarn, max_warn)
            if not file2merge:
                lfail = True
            else:
                filelist = self.manage_filemerge(filelist, file2merge, dest_dir)

        if lfail:
            nwarn = max_warn + 1
        else:
            # merge the data
            cdo.run(filelist + [final_file], OrderedDict([("merge", "")]))

            # rename variables
            self.add_varname_suffix(final_file, self.all_predictands, "_tar")

            # rename dimension of COSMO REA6-data for clarity if source data is not upscaled
            if not self.upscale_source: 
                self.rename_variables(final_file, {"rlat": "rlat_tar", "rlon": "rlon_tar"})

        return final_file, nwarn

    def process_era5_sf(self, dirin_era5: str, year_month, invar_file: str, tmp_dir: str) -> str:
        """
        Process surface-data from ERA5 files and combine with invariant surface data.
        :param dirin_era5: input directory of ERA5-dataset
        :param year_month: Date for which data should be processed
        :param invar_file: data file providing invariant variables of ERA5 dataset
        :param tmp_dir: temporary directory to store intermediate files
        :return: path to processed netCDF-datafile
        """
        cdo = self.cdo

        year_month_str = year_month.strftime("%Y-%m")   

        # check if ERA5 data file is available
        sf_file = os.path.join(dirin_era5, f"era5_{year_month_str}.nc")
        if not os.path.isfile(sf_file):
            raise FileNotFoundError(f"Could not find ERA5 data file {sf_file}.")
        
        if not os.path.isfile(invar_file):
            raise FileNotFoundError("Could not find required invariant-file '{1}'".format(invar_file))

        ftmp_era5 = os.path.join(tmp_dir, f"{year_month_str}_sf_tmp.nc")
        fera5_now = ftmp_era5.replace("_sf_tmp.nc", "_sf.nc")

        # handle dynamical and invariant variables
        sfvars_stat, sfvars_dyn = self.split_dyn_static(self.era5_sfc_vars)

        # choose variables of interest
        cdo.run([sf_file, ftmp_era5], OrderedDict([("-selname", ",".join(sfvars_dyn))]))
        print(sfvars_dyn)

        if sfvars_stat:
            ftmp_era5_2 = os.path.join(tmp_dir, "era5_invar.nc")
            if not os.path.isfile(ftmp_era5_2):   # has only to be done once
                cdo.run([invar_file, ftmp_era5_2], OrderedDict([("-selname", ",".join(sfvars_stat))]))
            # NOTE: ftmp_hres must be at first position to overwrite time-dimension of ftmp_hres2
            #       which would not fit since it is retrieved from an invariant datafile with arbitrary timestamp
            #       This works at least for CDO 2.0.2!!!
            cdo.run([ftmp_era5, ftmp_era5_2, fera5_now], OrderedDict([("-O", ""), ("merge", "")]))
            # clean-up temporary files
            remove_files([ftmp_era5], lbreak=False)
        else:
            fera5_now = ftmp_era5

        return fera5_now
    
    def process_era5_vl(self, dirin_era5: str, year_month, tmp_dir: str, vl_type: str) -> str:
        """
        Process multi-level data from ERA5 files by renaming variables
        :param dirin_era5: input directory of ERA5-dataset 
        :param year_month: Date for which data should be processed
        :param tmp_dir: temporary directory to store intermediate files
        :param vl_type: type of vertical level for which data should be processed ("ml" or "pl")
        :return: path to processed netCDF-datafile
        """
        assert vl_type in ["ml", "pl"], "vlvl_type must be either 'ml' or 'pl'."

        vl_vars_dict = self.era5_ml_vars if vl_type == "ml" else self.era5_pl_vars

        cdo = self.cdo

        year_month_str = year_month.strftime("%Y-%m")   

        # check if ERA5 data file is available
        vl_file = os.path.join(dirin_era5, f"era5_{year_month_str}.nc")
        if not os.path.isfile(vl_file):
            raise FileNotFoundError(f"Could not find ERA5 data file {vl_file}.")
        
        tmp_patt = os.path.join(tmp_dir, f"era5_{year_month_str}_{vl_type}")

        # get list of all unique model levels
        vl_list = list(set([vl for vl_list in vl_vars_dict.values() for vl in vl_list]))
        vlvars_str = ",".join(vl_vars_dict.keys())

        # split levels into files
        cdo.run([vl_file, tmp_patt], OrderedDict([("-L", ""), ("--reduce_dim", ""), ("-splitlevel", ""), ("-selname", vlvars_str)]))

        # rename variables in each file
        ftmp_list = []
        for vl in vl_list:
            ftmp_era5 = os.path.join(tmp_dir, f"era5_{year_month_str}_{vl_type}{vl:06d}.nc")
            if not os.path.isfile(ftmp_era5):
                raise FileNotFoundError(f"Could not find required file '{ftmp_era5}'.")

            # choose variables of interest
            vars_now = [var for var, vl_list in vl_vars_dict.items() if vl in vl_list]

            self.add_varname_suffix(ftmp_era5, vars_now, f"_{vl_type}{vl:d}")
            ftmp_list.append(ftmp_era5)

        # merge all files
        ftmp_era5 = os.path.join(tmp_dir, f"era5_{year_month_str}_{vl_type}.nc")
        _ = self.merge_multiple_netcdf(ftmp_list, ftmp_era5)

        # remove temporary files
        remove_files(ftmp_list, lbreak=False)
        
        return ftmp_era5

    def process_crea6_2d(self, file_2d: str, target_dir: str, date_str: str, gdes_tar):
        """
        Process 2D-variables of the COSMO-REA6 dataset, i.e. convert from grib to netCDF-format
        and slice the data to the domain of interest.
        :param file_2d: input grib-file containing a 2D-variable of COSMO-REA6
        :param target_dir: output-directory where processed file will be saved
        :param date_str: date-string with format YYYY-MM indicating month of interest
        :param gdes_tar: CDOGridDes-instance for the target domain
        :return file_out: path to resulting ouput file
        """
        cdo, ncrename = self.cdo, self.ncrename

        # retrieve grid information from CDOGridDes-instance
        gdes_dict = gdes_tar.grid_des_dict

        lonlatbox = (*gdes_tar.get_slice_coords(gdes_dict["xfirst"], gdes_dict["xinc"], gdes_dict["xsize"]),
                     *gdes_tar.get_slice_coords(gdes_dict["yfirst"], gdes_dict["yinc"], gdes_dict["ysize"]))
        lonlatbox_str = ",".join("{:.3f}".format(coord) for coord in lonlatbox)

        # sanity check
        if not os.path.isfile(file_2d):
            # check if netCDF-file is alternatively available
            file_2d = file_2d.replace(".grb", ".nc")
            if not os.path.isfile(file_2d):
                FileNotFoundError(f"Could not find required COSMO-REA6 file '{file_2d}'.")
        
        # retrieve variable name back from path to file
        var = os.path.basename(os.path.dirname(file_2d))
        dfile_out = os.path.join(target_dir, f"{var}_{date_str}.nc")

        # slice data and convert to netCDF
        # NOTE:
        # A remapping step is applied to avoid spurious differences in the underlying coordinates compared to the
        # remapped ERA5-data due to precision issues. Clearly, the user has to ensure that the grid description file
        # actually fits to the data at hand to avoid undesired remapping effects.
        cdo.run([file_2d, dfile_out], OrderedDict([("--reduce_dim", ""), ("-f nc", ""), ("copy", ""),
                                                   ("-sellonlatbox", lonlatbox_str), ("-remapcon", gdes_tar.file)]))

        # rename varibale in resulting file for grib-files where they may appear as var11 or similar (must be done in hacky manner)
        if file_2d.endswith(".grb"):
            varname = str(sp.check_output(f"cdo showname {dfile_out}", shell=True))
            varname = varname.lstrip("'b").split("\\n")[0].strip()

            ncrename.run([dfile_out], OrderedDict([("-v", f"{varname},{var}")]))

        return dfile_out

    def process_crea6_const(self, target_dir: str, gdes_tar):
        """
        Process invariant variables of the COSMO-REA6 dataset, i.e. convert from grib to netCDF-format
        and slice the data to the domain of interest.
        :param target_dir: output-directory where processed file will be saved
        :param gdes_tar: CDOGridDes-instance for the target domain
        :return file_out: path to resulting ouput file
        """
        cdo = self.cdo

        const_vars = to_list(self.crea6_const_vars)

        # retrieve grid information from CDOGridDes-instance
        gdes_dict = gdes_tar.grid_des_dict

        lonlatbox = (*gdes_tar.get_slice_coords(gdes_dict["xfirst"], gdes_dict["xinc"], gdes_dict["xsize"]),
                     *gdes_tar.get_slice_coords(gdes_dict["yfirst"], gdes_dict["yinc"], gdes_dict["ysize"]))
        lonlatbox_str = ",".join("{:.3f}".format(coord) for coord in lonlatbox)

        # sanity check
        if not os.path.isfile(self.constfile_tar):
            FileNotFoundError(f"Could not find required COSMO-REA6 file '{self.constfile_tar}'.")
        # retrieve variable name back from path to file
        dfile_out = os.path.join(target_dir, f"const_crea6.nc")

        cdo.run([self.constfile_tar, dfile_out], OrderedDict([("selname", ",".join(const_vars)),
                                                              ("-sellonlatbox", lonlatbox_str), ("-remapcon", gdes_tar.file)]))

        return dfile_out
    
    def remap_and_merge_data(self, file_in: str, file_tar: str, final_file: str, fgdes_coarse: str, fgdes_tar: str,
                             nwarn: int, max_warn: int) -> int:
        """
        Perform the remapping step on the predictor data and finally merge it with the predictand data
        :param file_in: netCDF-file with predictor data
        :param file_tar: netCDF-file with predictand data
        :param final_file: name of the resulting merged netCDF-file
        :param fgdes_coarse: CDO grid description file corresponding to the coarse-grained predictor data
        :param fgdes_tar: CDO grid description file corresponding to the high-resolved predictand data
        :param nwarn: current number of issued warnings
        :param max_warn: maximum allowed number of warnings
        :return: updated nwarn and resulting merged netCDF-file
        """
        cdo = self.cdo

        if not file_in.endswith(".nc"):
            raise ValueError(f"Input data-file '{file_in}' must be a netCDF-file.")
        file_in_coa = file_in.replace(".nc", "_coa.nc")
        file_in_merge = file_in.replace(".nc", "_merge.nc")

        # remap coarse ERA5-data
        cdo.run([file_in, file_in_coa], OrderedDict([("-remapcon", fgdes_coarse)]))
    
        if self.upscale_source: 
            # bi-linear interpolation onto target grid
            cdo.run([file_in_coa, file_in_merge], OrderedDict([("-remapbil", fgdes_tar)]))
        else: 
            # keep coarse-grained grid, but slice data and rename dimensions
            # To-Do: Dependency on lextrapolate-flag of create_coarsened_grid_des-method
            #        Slicing is only required if lextrapolate is True when creating the grid description of the coarse-grained input data
            #        So far, this is hard-coded in the parent class (cf. l.91 in preprocess_data_era5_to_ifs.py)
            
            # Obtain number of grid points for slicing
            gdes_coa = CDOGridDes(fgdes_coarse)
            nx, ny = int(gdes_coa.grid_des_dict["xsize"]), int(gdes_coa.grid_des_dict["ysize"])

            cdo.run([file_in_coa, file_in_merge], OrderedDict([("selindexbox", f"2,{nx-1:d},2,{ny-1:d}")]))
            # rename dimension of ERA5-data
            self.rename_variables(file_in_merge, {"rlat": "rlat_in", "rlon": "rlon_in"})

        # merge input and target data
        stat = self.merge_multiple_netcdf([file_in_merge, file_tar], final_file)

        if not (stat and os.path.isfile(final_file)):
            nwarn = max_warn + 1
        else:
            remove_files([file_in_coa, file_in_merge, file_tar], lbreak=True)

        return nwarn
    
    @staticmethod
    def get_predictor_varnames(var_dict):
        all_varnames = list(var_dict.get("sf", []))

        lvl_types = ["ml", "pl"]
        for vl in lvl_types:
            vlvars = var_dict.get(vl, {})

            for vlvar in vlvars:
                levels = var_dict[vl].get(vlvar)
                all_varnames += [f"{vlvar}_{vl}{lvl}" for lvl in levels]

        return all_varnames
