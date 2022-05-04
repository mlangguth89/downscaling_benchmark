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

    # expected key of grid description files
    expected_keys_gdes = ["gridtype", "xsize", "ysize", "xfirst", "xinc", "yfirst", "yinc"]
    # get required tool-instances (cdo with activated extrapolation)
    cdo, ncrename, ncap2, ncks, ncea = CDO(tool_envs={"REMAP_EXTRAPOLATE", "on"}), NCRENAME(), NCAP2(), NCKS(), NCEA()
    # hard-coded constants [IFS-specific parameters (from Chapter 12 in http://dx.doi.org/10.21957/efyk72kl)]
    cpd, g = 1004.709, 9.80665

    def __init__(self, source_dir_era5: str, source_dir_ifs, output_dir: str, grid_des_tar: str, predictors: dict,
                 predictands: dict, downscaling_fac: int = 8):
        """
        Initialize class for tier-1 downscaling dataset.
        """
        super().__init__("preprocess_ERA5_to_IFS", source_dir_era5, source_dir_ifs, predictors, predictands, output_dir)

        if not os.path.isfile(grid_des_tar):
            raise FileNotFoundError("Preprocess_Unet_Tier1: Could not find target grid description file '{0}'"
                                    .format(grid_des_tar))
        self.grid_des_tar = grid_des_tar
        self.my_rank = None                     # to be set in __call__
        self.downscaling_fac = downscaling_fac

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
                                                            name_base="era5_", lextrapolate=False)

        gdes_dict = {"tar_grid_des": ifs_grid_des.grid_des_dict, "coa_grid_des": coa_gdes_d}
        # define arguments and keyword arguments for running PyStager later
        run_dict = {"args": [self.source_dir_in, self.source_dir_out, self.target_dir, gdes_dict],
                    "kwargs": {"job_name": kwargs.get("jobname", "Preproce_ERA5_to_IFS")}}

        return preprocess_pystager, run_dict

    @staticmethod
    def preprocess_worker(year_months: List, dirin_era5: str, dirin_ifs: str, dirout: str, gdes_dict: dict,
                          predictors: dict, predictands: dict, logger: logging.Logger, nmax_warn: int = 3):
        """
        Function that preprocesses ERA5 (input) - and IFS (output)-data on individual workers
        :param year_months: List of Datetime-objects indicating year and month for which data should be preprocessed
        :param dirin_era5: input directory of ERA5-dataset (top-level directory)
        :param dirin_ifs: input directory of IFS-forecasts
        :param dirout: output directoty to store preprocessed data
        :param gdes_dict: dictionary containing grid description dictionaries for target, base and coarse grid
        :param logger: Logging instance for log process on worker
        :param nmax_warn: allowed maximum number of warnings/problems met during processing (default:3)
        :return: -
        """
        method = PreprocessERA5toIFS.preprocess_worker.__name__

        cdo = PreprocessERA5toIFS.cdo
        # sanity checks
        assert isinstance(logger, logging.Logger), "%{0}: logger-argument must be a logging.Logger instance" \
                                                   .format(method)

        sfvars, mlvars, fcvars = PreprocessERA5toIFS.get_vars(predictors)

        grid_des_tar, grid_des_coarse = gdes_dict["tar_grid_des"], gdes_dict["coa_grid_des"]

        for year_month in year_months:
            assert isinstance(year_month, dt.datetime),\
                "%{0}: All year_months-argument must be a datetime-object. Current one is of type '{1}'"\
                .format(method, type(year_month))

            year, month = int(year_month.strftime("%Y")), int(year_month.strftime("%m"))
            year_str, month_str = str(year), "{0:02d}".format(int(month))
            last_day = last_day_of_month(year_month)

            subdir = year_month.strftime("%Y-%m")
            dirr_curr_era5 = os.path.join(dirin_era5, str(year), subdir)
            dirr_curr_ifs = dirr_curr_era5.replace(dirin_era5, dirin_ifs)
            dest_nc_dir = os.path.join(dirout, "netcdf_data", year_str, subdir)
            os.makedirs(dest_nc_dir, exist_ok=True)

            # further sanity checks
            if not os.path.isdir(dirr_curr_era5):
                err_mess = "%{0}: Could not find directory for ERA5-data '{1}'".format(method, dirr_curr_era5)
                logger.critical(err_mess)
                raise NotADirectoryError(err_mess)

            if not os.path.isdir(dirr_curr_ifs):
                err_mess = "%{0}: Could not find directory for IFS-data '{1}'".format(method, dirr_curr_ifs)
                logger.critical(err_mess)
                raise NotADirectoryError(err_mess)

            dates2op = pd.date_range(dt.datetime.strptime("{0}{1}0100".format(year_str, month_str), "%Y%M%D%H"),
                                     last_day, freq="H")

            for date2op in dates2op:
                files2merge = []
                target_file = os.path.join(dest_nc_dir, "{}_preproc.nc".format(date2op.strftime("%Y%m%d%H")))
                # process surface variables
                if sfvars is not None:
                    files2merge.append(PreprocessERA5toIFS.process_sf_file(dirr_curr_era5, dest_nc_dir, date2op,
                                                                           grid_des_coarse, grid_des_tar, sfvars))
                # process multi-level files
                if mlvars is not None:
                    files2merge.append(PreprocessERA5toIFS.process_ml_file(dirr_curr_era5, dest_nc_dir, date2op,
                                                                           grid_des_coarse, grid_des_tar, mlvars))
                # process forecats files
                if fcvars is not None:
                    files2merge.append(PreprocessERA5toIFS.process_fc_file(dirr_curr_era5, dest_nc_dir, date2op,
                                                                           grid_des_coarse, grid_des_tar, fcvars))

                # finally all files
                cdo.run([target_file], ("merge", " ".join(files2merge)))

    @staticmethod
    def organize_predictors(predictors: dict) -> (List, dict, List):
        """
        Checks predictors for variables to process and returns condensed information for further processing
        :param predictors: dictionary for predictors
        :return: list of surface and forecast variables and dictionary of multi-level variables to interpolate
        """
        method = PreprocessERA5toIFS.organize_predictors.__name__

        known_vartypes =["sf", "ml", "sf_fc"]

        pred_vartypes = list(predictors.keys())
        lpred_vartypes = [pred_vartype in known_vartypes for pred_vartype in pred_vartypes]
        if not all(lpred_vartypes):
            unknown_vartypes = [pred_vartypes[i] for i, flag in enumerate(lpred_vartypes) if not flag]
            raise ValueError("%{0}: The following variables types in the predictor-dictionary are unknown: {1}"
                             .format(method, ", ".join(unknown_vartypes)))

        sfvars, mlvars, fcvars = predictors.get("sf", None), predictors.get("ml", None), predictors.get("fc_sf", None)

        # some checks (level information redundant for surface-variables)
        if any(i is not None for i in sfvars.values()):
            print("%{0}: Some values of sf-variables are not None, but do not have any effect.".format(method))

        if any(i is not None for i in fcvars.values()):
            print("%{0}: Some values of fc_sf-variables are not None, but do not have any effect.".format(method))

        sfvars, fcvars = list(sfvars), list(fcvars)
        # Process get list of unique target levels for interpolation
        lvls = set(list(flatten(mlvars.values())))
        plvls = [int(float(lvl.lstrip("p"))) for lvl in lvls if lvl.startswith("p")]
        # Currently only pressure-level interpolation is supported. Thus, we stop here if sth. unknown was parsed
        if len(lvls) != len(plvls):
            raise ValueError("%{0}: Could not retrieve all parsed level imformation. Check the folllowing: {1}"
                             .format(method, ", ".join(lvls)))
        mlvars["plvls"] = plvls

        return sfvars, mlvars, fcvars

    @staticmethod
    def process_sf_file(dir_curr_era5: str, target_dir: str, date2op: dt.datetime, fgdes_coarse: str,
                        fgdes_tar: str, sfvars: List) -> str:
        """
        Process surface ERA5-file, i.e. remap conservatively on coarsened grid followed by bilinear remapping
        onto the target (high-resolved) grid.
        :param dir_curr_era5: Input directory where the currently request ERA5-grib file is located
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
        date_str = date2op.strftime("%Y%M%D%H")
        sf_file = os.path.join(dir_curr_era5, "{0}_sf.grb".format(date_str))
        tmp_dir = os.path.join(target_dir, "tmp_{0}".format(date_str))
        os.makedirs(tmp_dir, exist_ok=True)

        if not os.path.isfile(sf_file):
            raise FileNotFoundError("%{0}: Could not find required surface-file '{1}'".format(method, sf_file))

        ftmp_coarse = os.path.join(tmp_dir, "{0}_sf_coarse.nc".format(date_str))
        ftmp_hres = ftmp_coarse.replace("sf_coarse", "sf_hres")

        l2t = False
        if "2t" in sfvars:
            sfvars.remove("2t")
            l2t = True

        # run remapping
        cdo.run([sf_file, ftmp_coarse], OrderedDict([("--eccodes", ""), ("-f", "nc"), ("copy", ""),
                                                     ("remapcon", fgdes_coarse), ("-selname", ",".join(sfvars))]))
        cdo.run([ftmp_coarse, ftmp_hres], OrderedDict([("remapbil", fgdes_tar)]))

        # special handling of 2m temperature
        if l2t:
            PreprocessERA5toIFS.remap_and_cat_2t(sf_file, ftmp_hres, fgdes_coarse, fgdes_tar)

        # clean-up temporary files
        remove_files([ftmp_coarse], lbreak=False)

        return ftmp_hres

    @staticmethod
    def process_ml_file(dir_curr_era5: str, target_dir: str, date2op: dt.datetime, fgdes_coarse: str,
                        fgdes_tar: str, mlvars: dict) -> str:
        """
        Process multi-level ERA5-file, i.e. interpolate on desired pressure levels, remap conservatively on coarsened
        grid and finally perform a bilinear remapping onto the target (high-resolved) grid.
        :param dir_curr_era5: Input directory where the currently request ERA5-grib file is located
        :param target_dir: Target directory to store the processed data in netCDF-files
        :param date2op: Date for which data should be processed
        :param fgdes_coarse: grid description file for coarse grid
        :param fgdes_tar: grid description file for target (high-resolved) grid
        :param mlvars: dictionary of predictor variables to be interpolated onto pressure levels,
                       e.g. {"t": {"p85000", "p70000"}}
        :return: path to processed netCDF-datafile
        """
        method = PreprocessERA5toIFS.process_ml_file.__name__

        cdo = PreprocessERA5toIFS.cdo
        ncrename = PreprocessERA5toIFS.ncrename

        # handle date and create tmp-directory and -files
        date_str = date2op.strftime("%Y%M%D%H")
        ml_file = os.path.join(dir_curr_era5, "{0}_ml.grb".format(date_str))
        tmp_dir = os.path.join(target_dir, "tmp_{0}".format(date_str))
        os.makedirs(tmp_dir, exist_ok=True)

        if not os.path.isfile(ml_file):
            raise FileNotFoundError("%{0}: Could not find required multi level-file '{1}'".format(method, ml_file))

        # construct filenames for all temporary files
        ftmp_plvl1 = os.path.join(tmp_dir, "{0}_plvl.nc".format(date_str))
        ftmp_plvl2 = ftmp_plvl1.replace("plvl.nc", "plvl_all")
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
        cdo.run([ml_file, ftmp_plvl1], OrderedDict([("--eccodes", ""), ("-f", "nc"), ("copy", ""),
                                                    ("-selname", ",".join(mlvars_list)),
                                                    ("-ml2plx,{0}".format(plvl_strs)),
                                                    ("-selname", ",".join(mlvars_list_interp))]))

        # Split pressure-levels into seperate files and ...
        cdo.run([ftmp_plvl1, ftmp_plvl1.rstrip(".nc")], OrderedDict([("--reduce_dim", ""), ("splitlevel", "")]))
        # ... rename variables accordingly in each resulting file
        for plvl in mlvars["plvls"]:
            for var in mlvars:
                var_new = "{0}{1:d}".format(var, int(plvl/100.))
                ncrename.run([ftmp_plvl1.replace(".nc", "{0:0d}.nc".format(int(plvl)))],
                             OrderedDict([("-v", "{0},{1}".format(var, var_new))]))

        # concatenate pressure-level files, reduce to final variables of interest and do the remapping steps
        cdo.run([ftmp_plvl2], ("merge", ftmp_plvl1.replace(".nc", "??????.nc")))
        cdo.run([ftmp_plvl2, ftmp_coarse], OrderedDict([("remapcon,{0}".format(fgdes_coarse)),
                                                        ("-selname", ",".join(var_new_req))]))
        cdo.run([ftmp_coarse, ftmp_hres], OrderedDict([("remapbil", fgdes_tar)]))

        # clean-up temporary files
        remove_files([ftmp_plvl1, ftmp_plvl1.replace(".nc", "??????.nc"),ftmp_plvl2, ftmp_coarse], lbreak=False)

        return ftmp_hres

    @staticmethod
    def process_fc_file(dirin_era5: str, target_dir: str, date2op: dt.datetime, fgdes_coarse: str,
                        fgdes_tar: str, fcvars: List) -> str:
        """
        Process short-range forecast ERA5-file, i.e. remap conservatively on coarsened grid followed by bilinear
        remapping onto the target (high-resolved) grid.
        :param dirin_era5: top-level directory where ERA5-data are placed (under <year>/<year>-<month>/-subdirectories)
        :param target_dir: Target directory to store the processed data in netCDF-files
        :param date2op: Date for which data should be processed
        :param fgdes_coarse: grid description file for coarse grid
        :param fgdes_tar: grid description file for target (high-resolved) grid
        :param fcvars: list of short-range forecast predictor variables
        :return: path to processed netCDF-datafile
        """
        cdo = PreprocessERA5toIFS.cdo

        # handle date and create tmp-directory and -files
        date_str = date2op.strftime("%Y%M%D%H")
        fc_file = PreprocessERA5toIFS.get_fc_file(dirin_era5, date2op)
        tmp_dir = os.path.join(target_dir, "tmp_{0}".format(date_str))
        os.makedirs(tmp_dir, exist_ok=True)

        ftmp_coarse = os.path.join(tmp_dir, "{0}_fc_coarse.nc".format(date_str))
        ftmp_hres = ftmp_coarse.replace("fc_coarse", "fc_hres")

        # flag for special handling of 2m temperature-data (variabe 2t in ERA5)
        l2t = False
        if "2t" in fcvars:
            fcvars.remove("2t")
            l2t = True

        # run remapping
        cdo.run([fc_file, ftmp_coarse], OrderedDict([("--eccodes", ""), ("-f", "nc"), ("copy", ""),
                                                     ("remapcon", fgdes_coarse), ("-selname", ",".join(fcvars))]))
        cdo.run([ftmp_coarse, ftmp_hres], OrderedDict([("remapbil", fgdes_tar)]))

        # special handling of 2m temperature
        if l2t:
            PreprocessERA5toIFS.remap_and_cat_2t(fc_file, ftmp_hres, fgdes_coarse, fgdes_tar)

        # clean-up temporary files
        remove_files([ftmp_coarse], lbreak=False)

        return ftmp_hres

    @staticmethod
    def remap_and_cat_2t(infile, outfile, grid_des_coarse, grid_des_tar) -> None:
        """
        Remap 2m temperature by transforming to dry static energy and concatenate outfile with the result.
        First, data is conservative remapping onto the coarse grid is performed, followed by bilinear remapping onto the
        target grid.
        :param infile: input data file with 2m temperature (2t) and surface geopotential (z)
        :param outfile: output-file which will be concatenated
        :param grid_des_coarse: grid description for coarse grid
        :param grid_des_tar: grid description for target (high-resolved) grid.
        :return:
        """
        cdo = PreprocessERA5toIFS.cdo

        cpd, g = PreprocessERA5toIFS.cpd, PreprocessERA5toIFS.g

        # temporary files (since CDO does not support modifying the input-file in place)
        ftmp_coarse = outfile.replace(".nc", "_s_coarse.nc")
        ftmp_coarse2 = ftmp_coarse.replace("_coarse", "_coarse2")
        ftmp_hres = outfile.replace(".nc", "_2t_tmp.nc")

        # run CDO-command chain
        cdo.run([infile, ftmp_coarse], OrderedDict([("-f", "nc"), ("copy", ""), ("-selname", "s,z"),
                                                     ("-expr", "'s={0}*2t+z+{1}*2'".format(cpd, g))]))
        cdo.run([ftmp_coarse, ftmp_coarse2], OrderedDict([("remapcon", grid_des_coarse)]))
        cdo.run([ftmp_coarse2, ftmp_hres], OrderedDict([("remapbil", grid_des_tar),
                                                        ("-expr", "'2t=(s-z-{0}*2)/{1}'".format(g, cpd))]))
        cdo.run([ftmp_hres, infile, infile], OrderedDict([("-O", ""), ("merge", "")]))

        # clean-up temporary files
        remove_files([ftmp_coarse, ftmp_coarse2, ftmp_hres], lbreak=False)

    @staticmethod
    def get_fc_file(dirin_era5: str, date: dt.datetime) -> str:
        """
        Get initialization of forecast run and forecast hour for retrieving properly data for given date
        :param dirin_era5: top-level directory where ERA5-data are placed (under <year>/<year>-<month>/-subdirectories)
        :param date: The date for which forecast data is requested
        :return: path to corresponding forecast file
        """
        hour = int(date.strftime("%H"))

        # only use forecasts between 6 and 18 hours lead time
        if hour <= 6:
            fh = 6 + hour
            run_init = date.replace(hour=18) - dt.timedelta(days=1)
        elif 6 < hour <= 18:
            fh = hour - 6
            run_init = date.replace(hour=6)
        else:
            fh = hour - 18
            run_init = date.replace(hour=18)

        # construct filename
        fc_file = os.path.join(dirin_era5, run_init.strftime("%Y"), run_init.strftime("%Y-%m"),
                               "fc_{0}".format(run_init.strftime("%H")),
                               "{0}_{1:d}00_{2:d}_sf_fc.grb".format(run_init.strftime("%Y%m%d%"),
                                                                    int(run_init.strftime("%H")), fh))

        if not os.path.isfile(fc_file):
            raise FileNotFoundError("Could not find requested forecast file '{0}'".format(fc_file))

        return fc_file

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
