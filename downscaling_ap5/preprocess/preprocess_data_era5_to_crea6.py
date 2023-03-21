__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-08-11"
__update__ = "2022-08-22"

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

from collections import OrderedDict
import datetime as dt
import glob
import logging
import os
import subprocess as sp
from typing import Union, List

import pandas as pd

# from tfrecords_utils import IFS2TFRecords
from abstract_preprocess import CDOGridDes
from preprocess_data_era5_to_ifs import PreprocessERA5toIFS
from tools_utils import CDO
from other_utils import to_list, last_day_of_month, remove_files

from aux_funcs import add_varname_suffix, check_crea6_files, get_varnames_from_mlvars, manage_filemerge, organize_predictands, organize_predictors, preprocess_crea6_tar, preprocess_era5_in, remap_and_merge_data, run_preproc_func
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

    def __init__(self, in_datadir: str, tar_datadir: str, out_dir: str, in_constfile: str, tar_constfile: str,
                 grid_des_tar: str, predictors: dict, predictands: dict, downscaling_fac: int = 5):
        """
        Initialize class for ERA5-to-COSMO REA6 downscaling class.
        """
        # initialize from ERA5-to-IFS class (parent class)
        super().__init__(in_datadir, tar_datadir, out_dir, in_constfile, grid_des_tar, predictors, predictands,
                         downscaling_fac)

        self.name_preprocess = "preprocess_ERA5_to_CREA6"

        # sanity check on constant/invariant file of COSMO REA6
        if not os.path.isfile(tar_constfile):
            raise FileNotFoundError("Could not find file with invariant data '{0}'.".format(tar_constfile))
        self.constfile_tar = tar_constfile

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

        # get lists of predictor and predictand variables
        sfvars_era5, mlvars_era5, fc_sfvars_era5, fc_mlvars_era5 = organize_predictors(predictors)
        
        all_predictors = to_list(sfvars_era5) + get_varnames_from_mlvars(mlvars_era5) + \
                         to_list(fc_sfvars_era5) + get_varnames_from_mlvars(fc_mlvars_era5)
        all_predictors = [e for e in all_predictors if e]

        sfvars_crea6, const_vars_crea6 = organize_predictands(predictands)
        all_predictands = [e for e in sfvars_crea6 + const_vars_crea6 if e]

        # append list of surface variables in case that 2m temperature (2t) is involved for special remapping approach
        if "2t" in to_list(sfvars_era5):
            sfvars_era5.append("z")
        if "2t" in to_list(fc_sfvars_era5):
            fc_sfvars_era5.append("2t")

        grid_des_tar, grid_des_coarse = gdes_dict["tar_grid_des"], gdes_dict["coa_grid_des"]

        # initialize number of warnings
        nwarn = 0
        for year_month in year_months:
            assert isinstance(year_month, dt.datetime),\
                "All year_months-argument must be a datetime-object. Current one is of type '{0}'"\
                .format(type(year_month))

            year, month = int(year_month.strftime("%Y")), int(year_month.strftime("%m"))
            year_str, month_str = str(year), "{0:02d}".format(int(month))
            last_day = last_day_of_month(year_month)

            subdir = year_month.strftime("%Y-%m")
            dir_curr_era5 = os.path.join(dirin_era5, year_str, month_str)
            _ = check_crea6_files(dirin_crea6, invar_file_crea6, subdir, sfvars_crea6, const_vars_crea6)
            dest_dir = os.path.join(dirout, "netcdf_data", year_str, subdir)
            final_file_era5 = os.path.join(dest_dir, "preproc_era5_{0}.nc".format(subdir))
            final_file = final_file_era5.replace("preproc_era5_", "preproc_")
            os.makedirs(dest_dir, exist_ok=True)

            # sanity check on ERA5-directory
            if not os.path.isdir(dir_curr_era5):
                err_mess = "Could not find directory for ERA5-data '{0}'".format(dir_curr_era5)
                logger.fatal(err_mess)
                raise NotADirectoryError(err_mess)

            if not os.path.isfile(final_file_era5):
                dates2op = pd.date_range(dt.datetime.strptime("{0}{1}0100".format(year_str, month_str), "%Y%m%d%H"),
                                         last_day.replace(hour=23), freq="H")

                # Perform logging, reset warning counter and loop over dates...
                logger.info("Start preprocessing data for month {0}...".format(subdir))

                for date2op in dates2op:
                    # !!!!!! ML: Preliminary fix to avoid processing data from 2015 !!!!!!
                    if date2op <= dt.datetime.strptime("20060101 12", "%Y%m%d %H"): continue
                    date_str, date_pr = date2op.strftime("%Y%m%d%H"), date2op.strftime("%Y-%m-%d %H:00 UTC")
                    hourly_file_era5 = os.path.join(dest_dir, "{}_preproc_era5.nc".format(date_str))
                    # Skip time step if file already exists
                    if os.path.isfile(hourly_file_era5): continue

                    lfail, nwarn = preprocess_era5_in(dirin_era5, invar_file_era5, hourly_file_era5,
                                                                          date2op, sfvars_era5, mlvars_era5, fc_sfvars_era5,
                                                                          fc_mlvars_era5, logger, nwarn, max_warn)

                    if not lfail: continue       # skip day if preprocessing ERA5-data failed

                    # finally all temporary files for each time step and clean-up
                    logger.info(f"Data for day {date_pr} successfully preprocessed.")

                # merge all time steps of the ERA5-data to monthly file and clean-up hourly files
                logger.info("Merge all hourly files to monthly datafile '{0}'".format(final_file_era5))
                all_hourly_files_era5 = glob.glob(os.path.join(dest_dir, "*_preproc_era5.nc"))
                cdo.run(all_hourly_files_era5 + [final_file_era5], OrderedDict([("mergetime", "")]))
                remove_files(all_hourly_files_era5, lbreak=True)
            else:
                logger.info("Monthly ERA5-file '{0}' already exists. Ensure that data is as expected."
                            .format(final_file_era5))

            # process COSMO-REA6 doata which is already organized in monthly files
            final_file_crea6, nwarn = \
                preprocess_crea6_tar(dirin_crea6, invar_file_crea6, grid_des_tar, dest_dir,
                                                           year_month, sfvars_crea6, const_vars_crea6, logger, nwarn,
                                                           max_warn)

            # finally merge the ERA5- and the COSMO REA6-data
            remap_and_merge_data(final_file_era5, final_file_crea6, final_file, grid_des_coarse,
                                                     grid_des_tar, all_predictors, all_predictands, nwarn, max_warn)

            # rename input-variables
            add_varname_suffix(final_file, all_predictors, "_in")

        return nwarn
    
    @classmethod
    def preprocess_crea6_tar(cls, dirin: str, invar_file: str, fgdes_tar: str, dest_dir: str, date2op: dt.datetime, vars_2d: List, const_vars: List, logger: logging.Logger, nwarn: int, max_warn):
        """
        Process COSMO REA6-files based on requested 2D- and invariant variables.
        :param dirin: top-level directory where COSMO REA6-data are placed (under <year>/<year>-<month>/-subdirectories)
        :param invar_file: datafile providing invariant COSMO REA6-data, e.g. HSURF
        :param fgdes_tar: file to CDO grid description file of target data
        :param dest_dir: Target directory to store the processed data in netCDF-files
        :param date2op: Date for which data should be processed
        :param vars_2d: List of requested 2D-variables
        :param const_vars: List of requested invariant variables
        :param logger: logging-instance
        :param nwarn: number of faced warnings in processing chain (will be updated here)
        :param max_warn: maximum number of allowd warnings
        :return: path to processed netCDF-datafile and updated number of warnings
        """
        date_str, date_str2 = date2op.strftime("%Y-%m"), date2op.strftime("%Y%m")
        tmp_dir = os.path.join(dest_dir, "tmp_{0}".format(date_str))
        final_file = os.path.join(dest_dir, f"preproc_crea6_{date_str}.nc")
        if os.path.isfile(final_file):
            logger.info("Monthly COSMO REA6-file '{0}' already exists. Ensure that data is as expected.".format(final_file))
            return final_file, nwarn

        gdes_tar = CDOGridDes(fgdes_tar)

        filelist = []

        lfail = False

        # process 2D-files
        if vars_2d:
            for var in vars_2d:    # TBD: Put the following into a callable object to accumulate nwarn and filelist
                dfile_in = os.path.join(dirin, "2D", var.upper(), f"{var.upper()}.2D.{date_str2}.grb")
                nwarn, file2merge = run_preproc_func(cls.process_2d_file, [dfile_in, dest_dir, date_str, gdes_tar], {}, logger, nwarn, max_warn)

                if not file2merge:
                    lfail = True
                else:
                    filelist = manage_filemerge(filelist, file2merge, tmp_dir)

        if const_vars and not lfail:
            nwarn, file2merge = run_preproc_func(cls.process_const_file, [invar_file, dest_dir, const_vars, date_str, gdes_tar], {}, logger, nwarn, max_warn)
            if not file2merge:
                lfail = True
            else:
                filelist = manage_filemerge(filelist, file2merge, tmp_dir)

        if lfail:
            nwarn = max_warn + 1
        else:
            # merge the data
            cls.cdo.run(filelist + [final_file], OrderedDict([("merge", "")]))
            # replicate constant data over all timesteps
            for const_var in const_vars:
                cls.ncap2.run([final_file, final_file], OrderedDict([("-A", ""),
                                                                    ("-s", f"{const_var}z[time,rlat,rlon]={const_var}")]))
                cls.ncks.run([final_file, final_file], OrderedDict([("-O", ""), ("-x", ""), ("-v", const_var)]))
                cls.ncrename.run([final_file], OrderedDict([("-v", f"{const_var}z,{const_var}")]))

            # rename variables
            add_varname_suffix(final_file, vars_2d + const_vars, "_tar")

        return final_file, nwarn
    
    @classmethod
    def process_const_file(cls, const_file: str, target_dir: str, const_vars: List, date_str: str, gdes_tar):
        """
        Process invariant variables of the COSMO-REA6 dataset, i.e. convert from grib to netCDF-format
        and slice the data to the domain of interest.
        :param const_file: input grib-file containing a 2D-variable of COSMO-REA6
        :param target_dir: output-directory where processed file will be saved
        :param const_vars: inariant variables of interest
        :param date_str: date-string with format YYYY-MM indicating month of interest
        :param gdes_tar: CDOGridDes-instance for the target domain
        :return file_out: path to resulting ouput file
        """

        const_vars = to_list(const_vars)

        # retrieve grid information from CDOGridDes-instance
        gdes_dict = gdes_tar.grid_des_dict

        lonlatbox = (*gdes_tar.get_slice_coords(gdes_dict["xfirst"], gdes_dict["xinc"], gdes_dict["xsize"]), *gdes_tar.get_slice_coords(gdes_dict["yfirst"], gdes_dict["yinc"], gdes_dict["ysize"]))
        lonlatbox_str = ",".join("{:.3f}".format(coord) for coord in lonlatbox)

        # sanity check
        if not os.path.isfile(const_file):
            FileNotFoundError(f"Could not find required COSMO-REA6 file '{const_file}'.")
        # retrieve variable name back from path to file
        dfile_out = os.path.join(target_dir, f"const_{date_str}.nc")

        cls.cdo.run([const_file, dfile_out], OrderedDict([("selname", ",".join(const_vars)), ("-sellonlatbox", lonlatbox_str), ("-remapcon", gdes_tar.file)]))

        return dfile_out
    
    @classmethod
    def process_2d_file(cls, file_2d: str, target_dir: str, date_str: str, gdes_tar):
        """
        Process 2D-variables of the COSMO-REA6 dataset, i.e. convert from grib to netCDF-format
        and slice the data to the domain of interest.
        :param file_2d: input grib-file containing a 2D-variable of COSMO-REA6
        :param target_dir: output-directory where processed file will be saved
        :param date_str: date-string with format YYYY-MM indicating month of interest
        :param gdes_tar: CDOGridDes-instance for the target domain
        :return file_out: path to resulting ouput file
        """
        # retrieve grid information from CDOGridDes-instance
        gdes_dict = gdes_tar.grid_des_dict

        lonlatbox = (*gdes_tar.get_slice_coords(gdes_dict["xfirst"], gdes_dict["xinc"], gdes_dict["xsize"]), *gdes_tar.get_slice_coords(gdes_dict["yfirst"], gdes_dict["yinc"], gdes_dict["ysize"]))
        lonlatbox_str = ",".join("{:.3f}".format(coord) for coord in lonlatbox)

        # sanity check
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
        cls.cdo.run([file_2d, dfile_out], OrderedDict([("--reduce_dim", ""), ("-f nc", ""), ("copy", ""), ("-sellonlatbox", lonlatbox_str), ("-remapcon", gdes_tar.file)]))

        # rename varibale in resulting file (must be done in hacky manner)
        varname = str(sp.check_output(f"cdo showname {dfile_out}", shell=True))
        varname = varname.lstrip("'b").split("\\n")[0].strip()

        cls.ncrename.run([dfile_out], OrderedDict([("-v", f"{varname},{var}")]))

        return dfile_out