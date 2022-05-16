__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-04-22"
__update__ = "2022-04-29"

# doc-string
"""
Main script to preprocess ERA5 data (provided on a 0.3°-grid) for first real downscaling application. 
The target of the downscaling will be IFS HRES data on a 0.1°-grid as in preprocess_data_unet_tier1.py.
Contrarily to the previous, simplified approach, no slicing regarding daytime and season will be performed. 
Thus, hourly input data are produced from the ERA5-dataset. For the output, hourly target data, 
IFS forecasts with lead time 6 to 17 hours is used. 
"""
# doc-string

import os, glob
from typing import Union, List
import shutil
import logging
import numbers
import datetime as dt
import numpy as np
import pandas as pd
from collections import OrderedDict
#from tfrecords_utils import IFS2TFRecords
from abstract_preprocess import AbstractPreprocessing
from preprocess_data_unet_tier1 import Preprocess_Unet_Tier1, CDOGridDes
from pystager_utils import PyStager
from tools_utils import CDO, NCRENAME, NCAP2, NCKS, NCEA
from other_utils import to_list, last_day_of_month, flatten, remove_files

number = Union[float, int]
num_or_List = Union[number, List[number]]
list_or_tuple = Union[List, tuple]
list_or_dict = Union[List, dict]


class PreprocessERA5toIFS(AbstractPreprocessing):

    # get required tool-instances (cdo with activated extrapolation)
    cdo, ncrename, ncap2, ncks, ncea = CDO(tool_envs={"REMAP_EXTRAPOLATE": "on"}), NCRENAME(), NCAP2(), NCKS(), NCEA()
    # hard-coded constants [IFS-specific parameters (from Chapter 12 in http://dx.doi.org/10.21957/efyk72kl)]
    cpd, g = 1004.709, 9.80665
    # invariant variables expected in the invarinat files
    const_vars = ["z", "lsm"]

    def __init__(self, in_datadir: str, tar_datadir: str, out_dir: str, in_constfile: str, grid_des_tar: str,
                 predictors: dict, predictands: dict, downscaling_fac: int = 8):
        """
        Initialize class for ERA5-to-IFS downscaling class.
        """
        super().__init__("preprocess_ERA5_to_IFS", in_datadir, tar_datadir, predictors, predictands, out_dir)

        # sanity checks
        if not os.path.isfile(grid_des_tar):
            raise FileNotFoundError("Preprocess_Unet_Tier1: Could not find target grid description file '{0}'"
                                    .format(grid_des_tar))
        if not os.path.isfile(in_constfile): 
            raise FileNotFoundError("Could not find file with invariant data '{0}'.".format(in_constfile))

        self.grid_des_tar = grid_des_tar
        self.invar_file = in_constfile
        self.downscaling_fac = downscaling_fac

        self.my_rank = None                     # to be set in __call__

    def prepare_worker(self, years: List, season: str, **kwargs):
        """
        Prepare workers for preprocessing.
        :param years: List of years to be processed.
        :param season: Season-string to be processed.
        :param kwargs: Arguments such as jobname for logger-filename
        """
        method = Preprocess_Unet_Tier1.__call__.__name__

        years = to_list(years)
        # sanity checks on years and season arguments
        assert all([isinstance(year, numbers.Number) for year in years]), \
            "%{0}: All elements of years must be numbers".format(method)

        years = [int(year) for year in years]
        months = PreprocessERA5toIFS.check_season(season)

        # initialize and set-up Pystager
        preprocess_pystager = PyStager(self.preprocess_worker, "year_month_list", nmax_warn=3)
        preprocess_pystager.setup(years, months)

        # Create grid description files needed for preprocessing (requires rank-information)
        self.my_rank = preprocess_pystager.my_rank

        ifs_grid_des = CDOGridDes(self.grid_des_tar)
        coa_gdes_d = ifs_grid_des.create_coarsened_grid_des(self.target_dir, self.downscaling_fac, self.my_rank,
                                                            name_base="era5_", lextrapolate=True)

        gdes_dict = {"tar_grid_des": ifs_grid_des.grid_des_dict["file"], "coa_grid_des": coa_gdes_d}
        # define arguments and keyword arguments for running PyStager later
        run_dict = {"args": [self.source_dir_in, self.source_dir_out, self.invar_file, self.target_dir, gdes_dict,
                             self.predictors, self.predictands],
                    "kwargs": {"job_name": kwargs.get("jobname", "Preproce_ERA5_to_IFS")}}

        return preprocess_pystager, run_dict

    @staticmethod
    def preprocess_worker(year_months: List, dirin_era5: str, dirin_ifs: str, invar_file: str, dirout: str,
                          gdes_dict: dict, predictors: dict, predictands: dict, logger: logging.Logger,
                          max_warn: int = 3):
        """
        Function that preprocesses ERA5 (input) - and IFS (output)-data on individual workers
        :param year_months: List of Datetime-objects indicating year and month for which data should be preprocessed
        :param dirin_era5: input directory of ERA5-dataset (top-level directory)
        :param dirin_ifs: input directory of IFS-forecasts
        :param invar_file: data file providing invariant variables
        :param dirout: output directoty to store preprocessed data
        :param predictors: nested dictionary of predictors, where the first-level key denotes the variable type,
                           and the second-level key-value pairs denote the variable as well as interpolation info
                           Example: { "sf": {"2t", "blh"}, "ml_fc": { "t", ["p85000", "p925000"]}}
        :param predictands: Same as predictors, but for predictands
        :param gdes_dict: dictionary containing grid description dictionaries for target, base and coarse grid
        :param logger: Logging instance for log process on worker
        :param max_warn: allowed maximum number of warnings/problems met during processing (default:3)
        :return: -
        """
        method = PreprocessERA5toIFS.preprocess_worker.__name__

        cdo = PreprocessERA5toIFS.cdo
        # sanity checks
        assert isinstance(logger, logging.Logger), "%{0}: logger-argument must be a logging.Logger instance" \
                                                   .format(method)
        if not os.path.isfile(invar_file):
            raise FileNotFoundError("File providing invariant data '{0}' cannot be found.".format(invar_file))

        sfvars, mlvars, fc_sfvars, fc_mlvars = PreprocessERA5toIFS.organize_predictors(predictors)

        grid_des_tar, grid_des_coarse = gdes_dict["tar_grid_des"], gdes_dict["coa_grid_des"]

        for year_month in year_months:
            assert isinstance(year_month, dt.datetime),\
                "%{0}: All year_months-argument must be a datetime-object. Current one is of type '{1}'"\
                .format(method, type(year_month))

            year, month = int(year_month.strftime("%Y")), int(year_month.strftime("%m"))
            year_str, month_str = str(year), "{0:02d}".format(int(month))
            last_day = last_day_of_month(year_month)

            subdir = year_month.strftime("%Y-%m")
            dir_curr_era5 = os.path.join(dirin_era5, year_str, month_str)
            dir_curr_ifs = os.path.join(dirin_ifs, year_str, subdir)
            dest_dir = os.path.join(dirout, "netcdf_data", year_str, subdir)
            final_file = os.path.join(dest_dir, "preproc_{0}".format(subdir))
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
                filelist = []
                date_str = date2op.strftime("%Y%m%d%H")
                tmp_dir = os.path.join(dest_dir, "tmp_{0}".format(date_str))
                daily_file = os.path.join(dest_dir, "{}_preproc.nc".format(date_str))
                logger.info("Start preprocessing data for {0}".format(date2op.strftime("%Y-%m-%d %H:00")))
                # process surface variables of ERA5 (predictors)
                if sfvars:
                    sf_file = os.path.join(dir_curr_era5, "{0}_sf.grb".format(date_str))
                    logger.info("Preprocess predictor from surface file '{0}' of ERA5-dataset".format(sf_file))
                    nwarn, file2merge = PreprocessERA5toIFS.run_preproc_func(PreprocessERA5toIFS.process_sf_file,
                                                                             [sf_file, invar_file, dest_dir, date2op,
                                                                              grid_des_coarse, grid_des_tar, sfvars],
                                                                             {}, logger, nwarn, max_warn)
                    filelist = PreprocessERA5toIFS.manage_filemerge(filelist, file2merge, tmp_dir)
                    if not file2merge: continue                           # skip day if some data is missing
                # process multi-level variables of ERA5 (predictors)
                if mlvars:
                    ml_file = os.path.join(dir_curr_era5, "{0}_ml.grb".format(date_str))
                    logger.info("Preprocess predictor from multi-level file '{0}' of ERA5-dataset".format(ml_file))
                    nwarn, file2merge = PreprocessERA5toIFS.run_preproc_func(PreprocessERA5toIFS.process_ml_file,
                                                                             [ml_file, dest_dir, date2op,
                                                                              grid_des_coarse, grid_des_tar, sfvars],
                                                                             {"interp": True}, logger, nwarn, max_warn)
                    filelist = PreprocessERA5toIFS.manage_filemerge(filelist, file2merge, tmp_dir)
                    if not file2merge: continue                           # skip day if some data is missing
                # process forecasted surface variables of ERA5 (predictors)
                if fc_sfvars:
                    fc_file = PreprocessERA5toIFS.get_fc_file(dirin_era5, date2op, model="era5", prefix="sf_fc")
                    logger.info("Preprocess predictor from surface forecast file '{0}' of ERA5-dataset"
                                .format(fc_file))
                    nwarn, file2merge = PreprocessERA5toIFS.run_preproc_func(PreprocessERA5toIFS.process_sf_file,
                                                                             [fc_file, invar_file, dest_dir, date2op,
                                                                              grid_des_coarse, grid_des_tar, fc_sfvars],
                                                                             {}, logger, nwarn, max_warn)
                    filelist = PreprocessERA5toIFS.manage_filemerge(filelist, file2merge, tmp_dir)
                    if not file2merge: continue                           # skip day if some data is missing
                # process forecasted multi-level variables of ERA5 (predictors)
                if fc_mlvars:
                    fc_file = PreprocessERA5toIFS.get_fc_file(dirin_era5, date2op, model="era5", prefix="ml_fc")
                    logger.info("Preprocess predictor from surface forecast file '{0}' of ERA5-dataset"
                                .format(fc_file))
                    nwarn, file2merge = PreprocessERA5toIFS.run_preproc_func(PreprocessERA5toIFS.process_ml_file,
                                                                             [fc_file, dest_dir, date2op,
                                                                              grid_des_coarse, grid_des_tar, fc_mlvars],
                                                                             {"interp": False}, logger, nwarn, max_warn)
                    filelist = PreprocessERA5toIFS.manage_filemerge(filelist, file2merge, tmp_dir)
                    if not file2merge: continue                           # skip day if some data is missing
                # process predictand variables of IFS
                logger.info("Preprocess predictands from IFS forecast files under '{0}'".format(dir_curr_ifs))
                nwarn, file2merge = PreprocessERA5toIFS.run_preproc_func(PreprocessERA5toIFS.process_ifs_file,
                                                                         [dirin_ifs, dest_dir, date2op,
                                                                          grid_des_tar, predictands], {}, logger,
                                                                         nwarn, max_warn)
                filelist = PreprocessERA5toIFS.manage_filemerge(filelist, file2merge, tmp_dir)
                if not file2merge: continue  # skip day if some data is missing
                # finally all temporary files for each time step
                logger.info("Merge temporary files to daily netCDF-file '{0}'".format(daily_file))
                cdo.run(filelist + [daily_file], OrderedDict([("merge", "")]))

            # merge all time steps to monthly file and clean-up daily files
            logger.info("Merge all daily files to monthly datafile '{0}'".format(final_file))
            all_daily_files = glob.glob(os.path.join(dest_dir, "*_preproc.nc"))
            cdo.run(all_daily_files + [final_file], OrderedDict([("mergetime", "")]))
            remove_files(all_daily_files, lbreak=True)

    @staticmethod
    def organize_predictors(predictors: dict) -> (List, dict, List):
        """
        Checks predictors for variables to process and returns condensed information for further processing
        :param predictors: dictionary for predictors
        :return: list of surface and forecast variables and dictionary of multi-level variables to interpolate
        """
        method = PreprocessERA5toIFS.organize_predictors.__name__

        known_vartypes = ["sf", "ml", "fc_sf", "fc_pl"]

        pred_vartypes = list(predictors.keys())
        lpred_vartypes = [pred_vartype in known_vartypes for pred_vartype in pred_vartypes]
        if not all(lpred_vartypes):
            unknown_vartypes = [pred_vartypes[i] for i, flag in enumerate(lpred_vartypes) if not flag]
            raise ValueError("%{0}: The following variables types in the predictor-dictionary are unknown: {1}"
                             .format(method, ", ".join(unknown_vartypes)))

        sfvars, mlvars, fc_sfvars, fc_plvars = predictors.get("sf", None), predictors.get("ml", None),\
                                               predictors.get("fc_sf", None), predictors.get("fc_pl", None)

        # some checks (level information redundant for surface-variables)
        if sfvars:
            if any([i is not None for i in sfvars.values()]):
                print("%{0}: Some values of sf-variables are not None, but do not have any effect.".format(method))
            sfvars = list(sfvars)

        if fc_sfvars:
            if any([i is not None for i in fc_sfvars.values()]):
                print("%{0}: Some values of fc_sf-variables are not None, but do not have any effect.".format(method))
            fc_sfvars = list(fc_sfvars)

        if mlvars:
            mlvars["plvls"] = PreprocessERA5toIFS.retrieve_plvls(mlvars)

        if fc_plvars:
            fc_plvars["plvls"] = PreprocessERA5toIFS.retrieve_plvls(fc_plvars)

        return sfvars, mlvars, fc_sfvars, fc_plvars

    @staticmethod
    def run_preproc_func(preproc_func: callable, args: List, kwargs: dict, logger: logging.Logger, nwarns: int,
                         max_warns: int) -> (int, str):
        """
        Run a function where arguments are parsed from list. Counts failures as warnings unless max_warns is exceeded
        or the error is not a Runtime-Error
        :param preproc_func: the callable preprocessing-function
        :param args: list of arguments to be parsed to preproc_func
        :param kwargs: dictionary of keyword arguments to be parsed to preproc_func
        :param logger: logger instance
        :param nwarns: current number of issued warnings
        :param max_warns: maximum allowed number of warnings
        :return: updated nwarns and outfile
        """
        assert callable(preproc_func), "func is not a callable, but of type '{0}'".format(type(preproc_func))

        try:
            outfile = preproc_func(*args, **kwargs)
        except (RuntimeError, FileNotFoundError) as err:
            mess = "Pre-Processing data from '{0}' failed! ".format(args[0])
            nwarns += 1
            if nwarns > max_warns:
                logger.fatal(mess + "Maximum number of warnings exceeded.")
                raise err
            else:
                logger.error(mess), logger.error(str(err))
                outfile = None
        except BaseException as err:
            logger.fatal("Something unexpected happened when handling data from '{0}'. See error-message"
                         .format(args[0]))
            raise err

        return nwarns, outfile

    @staticmethod
    def manage_filemerge(filelist: List, file2merge: str, tmp_dir: str, search_patt: str = "*.nc"):
        """
        Add file2merge to list of files or clean-up temp-dirctory if file2merge is None
        :param filelist: list of files to be updated
        :param file2merge: file to merge
        :param tmp_dir: directory for temporary data
        :param search_patt: search pattern for files to remove
        :return: updated filelist
        """
        if file2merge:
            filelist.append(file2merge)
        else:
            remove_list = glob.iglob(os.path.join(tmp_dir, search_patt))
            remove_files(remove_list, lbreak=True)
        return filelist

    @staticmethod
    def process_sf_file(sf_file: str, invar_file: str, target_dir: str, date2op: dt.datetime, fgdes_coarse: str,
                        fgdes_tar: str, sfvars: List) -> str:
        """
        Process surface ERA5-file, i.e. remap conservatively on coarsened grid followed by bilinear remapping
        onto the target (high-resolved) grid.
        :param sf_file: ERA5-file with surface variables to process
        :param invar_file: ERA5-file with invariant variables
        :param target_dir: Target directory to store the processed data in netCDF-files
        :param date2op: Date for which data should be processed
        :param fgdes_coarse: grid description file for coarse grid
        :param fgdes_tar: grid description file for target (high-resolved) grid
        :param sfvars: list of surface predictor variables
        :return: path to processed netCDF-datafile
        """
        method = PreprocessERA5toIFS.process_sf_file.__name__

        cdo = PreprocessERA5toIFS.cdo

        # handle date and create tmp-directory and -files
        date_str = date2op.strftime("%Y%m%d%H")
        tmp_dir = os.path.join(target_dir, "tmp_{0}".format(date_str))
        os.makedirs(tmp_dir, exist_ok=True)

        if not os.path.isfile(sf_file):
            raise FileNotFoundError("%{0}: Could not find required surface-file '{1}'".format(method, sf_file))

        ftmp_coarse = os.path.join(tmp_dir, "{0}_sf_coarse.nc".format(date_str))
        ftmp_hres = ftmp_coarse.replace("sf_coarse", "sf_hres")

        # handle dynamical and invariant variables
        sfvars_stat, sfvars_dyn = PreprocessERA5toIFS.split_dyn_static(sfvars)

        l2t = False
        if "2t" in sfvars_dyn:
            # remove 2t from dynamical variables list
            sfvars_dyn.remove("2t")
            l2t = True

        # run remapping
        cdo.run([sf_file, ftmp_coarse], OrderedDict([("--eccodes", ""), ("-f nc", ""), ("copy", ""),
                                                     ("-remapcon", fgdes_coarse), ("-selname", ",".join(sfvars_dyn))]))
        if sfvars_stat:
            ftmp_coarse2 = ftmp_coarse.replace("sf", "sf_stat")
            if not os.path.isfile(ftmp_coarse2):   # has only to be done once
                cdo.run([invar_file, ftmp_coarse2], OrderedDict([("--eccodes", ""), ("-f nc", ""), ("copy", ""),
                                                                 ("-remapcon", fgdes_coarse),
                                                                 ("-selname", ",".join(sfvars_stat))]))
            cdo.run([ftmp_coarse2, ftmp_coarse, ftmp_coarse], OrderedDict([("-O", ""), ("merge", "")]))

        cdo.run([ftmp_coarse, ftmp_hres], OrderedDict([("remapbil", fgdes_tar)]))

        # special handling of 2m temperature
        if l2t:
            PreprocessERA5toIFS.remap2t_and_cat(sf_file, invar_file, ftmp_hres, fgdes_coarse, fgdes_tar)
            sfvars_dyn.append("2t")

        # clean-up temporary files and rename variables
        remove_files([ftmp_coarse], lbreak=False)
        PreprocessERA5toIFS.add_varname_suffix(ftmp_hres, sfvars, "_in")

        return ftmp_hres

    @staticmethod
    def process_ml_file(ml_file: str, target_dir: str, date2op: dt.datetime, fgdes_coarse: str,
                        fgdes_tar: str, mlvars: dict, interp: bool = True) -> str:
        """
        Process multi-level ERA5-file, i.e. interpolate on desired pressure levels, remap conservatively on coarsened
        grid and finally perform a bilinear remapping onto the target (high-resolved) grid.
        :param ml_file: ERA5-file with variables on multi-levels or pressure-levels to process
        :param target_dir: Target directory to store the processed data in netCDF-files
        :param date2op: Date for which data should be processed
        :param fgdes_coarse: grid description file for coarse grid
        :param fgdes_tar: grid description file for target (high-resolved) grid
        :param mlvars: dictionary of predictor variables to be interpolated onto pressure levels,
                       e.g. {"t": {"p85000", "p70000"}}
        :param interp: True if pressure interpolation is required or False if data is available on pressure levels
        :return: path to processed netCDF-datafile
        """
        method = PreprocessERA5toIFS.process_ml_file.__name__

        cdo = PreprocessERA5toIFS.cdo
        ncrename = PreprocessERA5toIFS.ncrename

        # handle date and create tmp-directory and -files
        date_str = date2op.strftime("%Y%m%d%H")
        tmp_dir = os.path.join(target_dir, "tmp_{0}".format(date_str))
        os.makedirs(tmp_dir, exist_ok=True)

        if not os.path.isfile(ml_file):
            raise FileNotFoundError("%{0}: Could not find required multi level-file '{1}'".format(method, ml_file))

        # construct filenames for all temporary files
        ftmp_plvl1 = os.path.join(tmp_dir, "{0}_plvl.nc".format(date_str))
        ftmp_plvl2 = ftmp_plvl1.replace("plvl.nc", "plvl_all.nc")
        ftmp_coarse = os.path.join(tmp_dir, "{0}_ml_coarse.nc".format(date_str))
        ftmp_hres = ftmp_coarse.replace("ml_coarse", "ml_hres")

        # Create lists of variables as well as pressure strings required for pressure interpolation
        mlvars_list = list(mlvars.keys())
        mlvars_list.remove("plvls")
        mlvars_list_interp = mlvars_list + ["t", "lnsp", "z"]
        plvl_strs = ",".join(["{0:d}".format(int(plvl)) for plvl in mlvars["plvls"]])
        var_new_req = ["{0}{1}".format(var, int(int(plvl.lstrip("p"))/100))
                       for var in mlvars_list for plvl in mlvars[var]]

        # interpolate variables of interest onto pressure levels
        if interp:
            cdo.run([ml_file, ftmp_plvl1], OrderedDict([("--eccodes", ""), ("-f nc", ""), ("copy", ""),
                                                        ("-selname", ",".join(mlvars_list)),
                                                        ("-ml2plx,{0}".format(plvl_strs)),
                                                        ("-selname", ",".join(mlvars_list_interp))]))
        else:
            cdo.run([ml_file, ftmp_plvl1], OrderedDict([("--eccodes", ""), ("-f nc", ""), ("copy", ""),
                                                        ("-selname", ",".join(mlvars_list)),
                                                        ("-sellevel", plvl_strs)]))

        # Split pressure-levels into seperate files and ...
        cdo.run([ftmp_plvl1, ftmp_plvl1.rstrip(".nc")], OrderedDict([("--reduce_dim", ""), ("splitlevel", "")]))
        # ... rename variables accordingly in each resulting file
        for plvl in mlvars["plvls"]:
            for var in mlvars_list:
                var_new = "{0}{1:d}".format(var, int(plvl/100.))
                ncrename.run([ftmp_plvl1.replace(".nc", "{0:06d}.nc".format(int(plvl)))],
                             OrderedDict([("-v", "{0},{1}".format(var, var_new))]))

        # concatenate pressure-level files, reduce to final variables of interest and do the remapping steps
        cdo.run([ftmp_plvl1.replace(".nc", "??????.nc"), ftmp_plvl2], OrderedDict([("-O", ""), ("merge", "")]))
        cdo.run([ftmp_plvl2, ftmp_coarse], OrderedDict([("-remapcon", fgdes_coarse),
                                                        ("-selname", ",".join(var_new_req))]))
        cdo.run([ftmp_coarse, ftmp_hres], OrderedDict([("remapbil", fgdes_tar)]))

        # clean-up temporary files and rename variables
        remove_files([ftmp_plvl1, ftmp_plvl1.replace(".nc", "??????.nc"), ftmp_plvl2, ftmp_coarse], lbreak=False)
        PreprocessERA5toIFS.add_varname_suffix(ftmp_hres, var_new_req, "_in")

        return ftmp_hres

    @staticmethod
    def process_ifs_file(dirin_ifs: str, target_dir: str, date2op: dt.datetime, fgdes_tar: str,
                         predictands: dict) -> str:
        """
        Process IFS-file by slicing data to region of interest.
        :param dirin_ifs: top-level directory where IFS-data are placed (under <year>/<year>-<month>/-subdirectories)
        :param target_dir: Target directory to store the processed data in netCDF-files
        :param date2op: Date for which data should be processed
        :param fgdes_tar: grid description file for target (high-resolved) grid
        :param predictands: dictionary for predictand variables
        :return: path to processed netCDF-datafile
        """
        cdo = PreprocessERA5toIFS.cdo

        # handle date and create tmp-directory and -files
        date_str = date2op.strftime("%Y%m%d%H")
        ifs_file = PreprocessERA5toIFS.get_fc_file(dirin_ifs, date2op, model="ifs", suffix="sfc")
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
                OrderedDict([("-seltimestep", "7/12"), ("-selname", ",".join(ifsvars)),
                             ("-sellonlatbox", "{0:.2f},{1:.2f},{2:.2f},{3:.2f}".format(*lonlatbox))]))

        # clean-up temporary files and rename variables
        PreprocessERA5toIFS.add_varname_suffix(ftmp_hres, ifsvars, "_tar")

        return ftmp_hres

    @staticmethod
    def remap2t_and_cat(infile: str, invar_file: str, outfile: str, grid_des_coarse: str, grid_des_tar: str) -> None:
        """
        Remap 2m temperature by transforming to dry static energy and concatenate outfile with the result.
        First, data is conservative remapping onto the coarse grid is performed, followed by bilinear remapping onto the
        target grid.
        :param infile: input data file with 2m temperature (2t)
        :param invar_file: invariant data file providing geopotential z
        :param outfile: output-file which will be concatenated
        :param grid_des_coarse: grid description file for coarse grid
        :param grid_des_tar: grid description file for target (high-resolved) grid.
        :return:
        """
        cdo = PreprocessERA5toIFS.cdo

        cpd, g = PreprocessERA5toIFS.cpd, PreprocessERA5toIFS.g

        # temporary files (since CDO does not support modifying the input-file in place)
        ftmp_invar = outfile.replace(".nc", "_invar.nc")
        ftmp_in = outfile.replace(".nc", "in.nc")
        ftmp_coarse = outfile.replace(".nc", "_s_coarse.nc")
        ftmp_hres = outfile.replace(".nc", "_2t_tmp.nc")

        # run CDO-command chain
        if not os.path.isfile(ftmp_invar):   # extract geopotential from invariant datafile
            cdo.run([invar_file, ftmp_invar], OrderedDict([("--eccodes", ""), ("-f nc", ""), ("copy", ""),
                                                           ("-selname", "z")]))

        # extract 2m temperature from datafile of interest, merge with invariant geopotential file and...
        cdo.run([infile, ftmp_in], OrderedDict([("--eccodes", ""), ("-f nc", ""), ("copy", ""), ("-selname", "2t")]))
        cdo.run([ftmp_invar, ftmp_in, ftmp_in], OrderedDict([("-O", ""), ("merge", "")]))
        # ... finally do the remapping
        cdo.run([ftmp_in, ftmp_coarse], OrderedDict([("-remapcon", grid_des_coarse), ("-selname", "s,z"),
                                                     ("-aexpr", "'s={0}*2t+z+{1}*2'".format(cpd, g))]))
        cdo.run([ftmp_coarse, ftmp_hres], OrderedDict([("-remapbil", grid_des_tar), ("-selname", "2t"),
                                                       ("-aexpr", "'2t=(s-z-{0}*2)/{1}'".format(g, cpd))]))
        cdo.run([ftmp_hres, outfile, outfile], OrderedDict([("-O", ""), ("merge", "")]))

        # clean-up temporary files
        remove_files([ftmp_in, ftmp_coarse, ftmp_hres], lbreak=False)

    @staticmethod
    def split_dyn_static(sfvars: List):
        """
        Split list of surface variables into lists of static and dynamical variables (see const_vars-variable of class).
        :param sfvars: input list of surface variables
        :return: two lists where the first holds the static and the second holds the dynamical variables
        """
        sfvars_stat = [sfvar for sfvar in sfvars if sfvar in PreprocessERA5toIFS.const_vars]
        sfvars_dyn = [sfvar for sfvar in sfvars if sfvar not in sfvars_stat]

        return sfvars_stat, sfvars_dyn

    @staticmethod
    def get_fc_file(dirin_base: str, date: dt.datetime, offset: int = 6,  model: str = "era5", suffix="",
                    prefix="") -> str:
        """
        Construct path to forecast file corresponding to specific date from ECMWF forecasts (e.g. IFS or ERA5).
        :param dirin_base: top-level directory where ECMWF forecasts are placed (in <year>/<year>-<month>/-subdirs)
        :param date: The date for which forecast data is requested
        :param model: The ECMWF model for which forecast file is requested (either 'ERA5' or 'IFS')
        :param offset: Offset in hours for forecasts to use (e.g. 6 means that lead times smaller 6 hours are not used)
        :param suffix: Suffix to forecast filename (for IFS-forecasts only, e.g. 'sfc' or 'pl')
        :param prefix: Prefix to forecast filename (for ERA5-forecasts only, e.g. 'sf_fc' or 'pl_fc')
        :return: path to corresponding forecast file
        """
        # sanity checks and setting of model initialization time
        assert offset < 12, "Offset must be smaller than 12, but is {0:d}".format(offset)

        model = model.lower()
        if model == "era5":
            init_model = [6, 18]
        elif model == "ifs":
            init_model = [0, 12]
        else:
            raise "Model {0} is not supported. Only IFS and ERA5 are valid models.".format(model)
        # get daytime hour
        hour = int(date.strftime("%H"))

        # construct initialization time of model run and corresponding forecast hour
        if hour < offset + init_model[0]:
            fh = 24 - init_model[1] + hour
            run_init = date.replace(hour=init_model[1]) - dt.timedelta(days=1)
        elif offset + init_model[0] <= hour < offset + init_model[1]:
            fh = hour - init_model[0]
            run_init = date.replace(hour=init_model[0])
        elif hour > init_model[1] + offset and init_model[1] + offset < 24:
            fh = hour - init_model[1]
            run_init = date.replace(hour=init_model[1])
        else:
            raise ValueError("Combination of init hours ({0:d}, {1:d}) and offset {2} not implemented."
                             .format(init_model[0], init_model[1], offset))
        # construct resulting filenames
        if model == "era5":
            nc_file = os.path.join(dirin_base, run_init.strftime("%Y"), run_init.strftime("%m"),
                                   "fc_{0}".format(run_init.strftime("%H")),
                                   "{0}_{1:d}00_{2:d}_{3}.grb".format(run_init.strftime("%Y%m%d"),
                                                                      int(run_init.strftime("%H")), fh, prefix))
        elif model == "ifs":
            nc_file = os.path.join(dirin_base, run_init.strftime("%Y"), run_init.strftime("%Y-%m"),
                                   "{0}_{1}_{2}.nc".format(suffix, run_init.strftime("%Y%m%d"),
                                                           run_init.strftime("%H")))

        if not os.path.isfile(nc_file):
            raise FileNotFoundError("Could not find requested forecast file '{0}'".format(nc_file))

        return nc_file

    @staticmethod
    def add_varname_suffix(nc_file: str, varnames: List, suffix: str):
        """
        Rename variables in netCDF-file by adding a suffix
        :param nc_file: netCDF-file to process
        :param varnames: (old) variable names to modify
        :param suffix: suffix to add to variable names
        :return: status-flag
        """
        ncrename = PreprocessERA5toIFS.ncrename

        varnames_new = [varname + suffix for varname in varnames]
        varnames_pair = ["{0},{1}".format(varnames[i], varnames_new[i]) for i in range(len(varnames))]

        try:
            ncrename.run([nc_file], OrderedDict([("-v", varnames_pair)]))
            stat = True
        except RuntimeError as err:
            print("Could not rename all parsed variables: {0}".format(",".join(varnames)))
            raise err

        return stat

    @staticmethod
    def retrieve_plvls(mlvars_dict):
        """
        Returns list of unique pressure levels from nested variable dictionary of form
        :param mlvars_dict: nested variable dictionary, e.g. {<var1>: ["p85000", "p92500"], <var2>: ["p85000"]}
        :return: list of uniues pressure levels, e.g [85000, 925000] in this example
        """
        lvls = set(list(flatten(mlvars_dict.values())))
        plvls = [int(float(lvl.lstrip("p"))) for lvl in lvls if lvl.startswith("p")]
        # Currently only pressure-level interpolation is supported. Thus, we stop here if other level identifier is used
        if len(lvls) != len(plvls):
            raise ValueError("Could not retrieve all parsed level imformation. Check the folllowing: {0}"
                             .format(", ".join(lvls)))

        return plvls

    @staticmethod
    def check_season(season: str) -> List:
        """
        Check if season-string is known.
        :param season: the seson string identifier
        :return: corresponding months as list of integers
        """
        method = PreprocessERA5toIFS.check_season.__name__

        known_seasons = ["DJF", "MMA", "JJA", "SON", "summer", "winter", "all"]

        if season == "DJF":
            months = [12, 1, 2]
        elif season == "MMA":
            months = [3, 4, 5]
        elif season == "JJA":
            months = [6, 7, 8]
        elif season == "SON":
            months = [9, 10, 11]
        elif season == "summer":
            months = list(np.arange(4, 10))
        elif season == "winter":
            months = list(np.arange(1, 4)) + list(np.arange(10, 13))
        elif season == "all":
            months = list(np.arange(1, 13))
        else:
            raise ValueError("%{0}: Parsed season-string '{1}' is unknown. Handle one of the following known ones: {1}"
                             .format(method, ", ".join(known_seasons)))

        return months
