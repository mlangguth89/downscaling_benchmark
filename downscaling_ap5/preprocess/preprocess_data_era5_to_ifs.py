__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-04-22"
__update__ = "2022-08-22"

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
import logging
import numbers
import datetime as dt
from collections import OrderedDict

import numpy as np
import pandas as pd

# from tfrecords_utils import IFS2TFRecords

from abstract_preprocess import AbstractPreprocessing, CDOGridDes
from other_utils import to_list, last_day_of_month, remove_files
from pystager_utils import PyStager
from tools_utils import CDO, NCRENAME, NCAP2, NCKS, NCEA, NCWA
from dataset_utils import CDOGrid, Variable, Files

from aux_funcs import (
    check_season,
    get_fc_file,
    get_varnames_from_mlvars,
    manage_filemerge,
    merge_two_netcdf,
    preprocess_ifs_tar,
    organize_predictors,
    remap_and_merge_data,
    run_preproc_func,
    split_dyn_static,
)

number = Union[float, int]
num_or_List = Union[number, List[number]]
list_or_tuple = Union[List, tuple]
list_or_dict = Union[List, dict]


class PreprocessERA5toIFS(AbstractPreprocessing):
    # get required tool-instances (cdo with activated extrapolation)
    cdo, ncrename, ncap2, ncks, ncea = (
        CDO(tool_envs={"REMAP_EXTRAPOLATE": "on"}),
        NCRENAME(),
        NCAP2(),
        NCKS(),
        NCEA(),
    )
    ncwa = NCWA()
    # hard-coded constants [IFS-specific parameters (from Chapter 12 in http://dx.doi.org/10.21957/efyk72kl)]
    cpd, g = 1004.709, 9.80665
    # invariant variables expected in the invarinat files
    const_vars = ["z", "lsm"]

    def __init__(
        self,
        in_datadir: str,
        tar_datadir: str,
        out_dir: str,
        in_constfile: str,
        grid_des_tar: str,
        predictors: dict,
        predictands: dict,
        downscaling_fac: int = 8,
    ):
        """
        Initialize class for ERA5-to-IFS downscaling class.
        """
        super().__init__(
            "preprocess_ERA5_to_IFS",
            in_datadir,
            tar_datadir,
            predictors,
            predictands,
            out_dir,
        )

        # sanity checks
        if not os.path.isfile(grid_des_tar):
            raise FileNotFoundError(
                "Could not find target grid description file '{0}'".format(grid_des_tar)
            )
        if not os.path.isfile(in_constfile):
            raise FileNotFoundError(
                "Could not find file with invariant data '{0}'.".format(in_constfile)
            )

        self.grid_des_tar = grid_des_tar
        self.invar_file = in_constfile
        self.downscaling_fac = downscaling_fac

        self.my_rank = None  # to be set in __call__

    def prepare_worker(self, years: List, season: str, **kwargs):
        """
        Prepare workers for preprocessing.
        :param years: List of years to be processed.
        :param season: Season-string to be processed.
        :param kwargs: Arguments such as jobname for logger-filename
        """
        years = to_list(years)
        # sanity checks on years and season arguments
        assert all(
            [isinstance(year, numbers.Number) for year in years]
        ), "All elements of years must be numbers"

        years = [int(year) for year in years]
        months = check_season(season)

        # initialize and set-up Pystager
        preprocess_pystager = PyStager(
            self.preprocess_worker, "year_month_list", nmax_warn=3
        )
        preprocess_pystager.setup(years, months)

        # Create grid description files needed for preprocessing (requires rank-information)
        self.my_rank = preprocess_pystager.my_rank

        ifs_grid_des = CDOGridDes(self.grid_des_tar)
        coa_gdes_d = ifs_grid_des.create_coarsened_grid_des(
            self.target_dir,
            self.downscaling_fac,
            self.my_rank,
            name_base=kwargs.get("coarse_grid_name", "era5_"),
            lextrapolate=True,
        )

        gdes_dict = {
            "tar_grid_des": ifs_grid_des.grid_des_dict["file"],
            "coa_grid_des": coa_gdes_d,
        }
        # define arguments and keyword arguments for running PyStager later
        run_dict = {
            "args": [
                self.source_dir_in,
                self.source_dir_out,
                self.invar_file,
                self.target_dir,
                gdes_dict,
                self.predictors,
                self.predictands,
            ],
            "kwargs": {"job_name": kwargs.get("jobname", "Preproce_ERA5_to_IFS")},
        }

        return preprocess_pystager, run_dict
    
    def worker(
        self,
        time: List[dt.datetime],
        grid: CDOGrid,
        pathes: Files,
        predictors: List[Variable],
        predictands: List[Variable],
        max_warn = 3
    ):
        "implement new interface, but imitate old one."
        
        kwargs = {
            "year_months": time, # List[datetime]
            "source_dir_input": pathes.input_dir_source,
            "source_dir_target": pathes.input_dir_target, # only for era5 to ifs/crea6
            "invar_file_input": pathes.invars_source, # only for era5 to ifs/crea6
            "dirout": pathes.output_dir, # all
            "gdes_dict": {"tar_grid_des","coa_grid_des"}, # all
            "predictors": predictors.to_old_format,
            "predictands": predictands.to_old_format,
            "logger": logging.Logger, # all => implement differently
            "max_warn": 3 # all
        }
        
        return self.__class__.preprocess_worker(kwargs)

    @staticmethod
    def preprocess_worker(
        year_months: List,
        dirin_era5: str,
        dirin_ifs: str,
        invar_file: str,
        dirout: str,
        gdes_dict: dict,
        predictors: dict,
        predictands: dict,
        logger: logging.Logger,
        max_warn: int = 3,
    ):
        """
        Function that preprocesses ERA5 (input) - and IFS (output)-data on individual workers.
        For each month in year_months, the following steps are performed:
            1.) Retrieve the predictors from the hourly ERA5-data and perform vertical interpolation if required
            2.) Retrieve the predictands from the hourly IFS-data
            3.) Merge all predictor and predictand data to separate monthly files.
            4.) Remap predictor data to match the grid of the predictands after bilinear interpolation.
            5.) Merge predictor and predictand files into final file.

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
        assert isinstance(
            logger, logging.Logger
        ), "%{0}: logger-argument must be a logging.Logger instance".format(method)
        if not os.path.isfile(invar_file):
            raise FileNotFoundError(
                "File providing invariant data '{0}' cannot be found.".format(
                    invar_file
                )
            )

        # get lists of predictor and predictand variables
        sfvars, mlvars, fc_sfvars, fc_mlvars = organize_predictors(predictors)
        all_predictors = (
            to_list(sfvars)
            + PreprocessERA5toIFS.get_varnames_from_mlvars(mlvars)
            + to_list(fc_sfvars)
            + PreprocessERA5toIFS.get_varnames_from_mlvars(fc_mlvars)
        )
        all_predictors = [e for e in all_predictors if e]

        if any(vartype != "sf" for vartype in predictands.keys()):
            raise ValueError(
                "Only surface variables (i.e. vartype 'sf') are currently supported for IFS data."
            )
        all_predictands = list(predictands["sf"].keys())

        # append list of surface variables in case that 2m temperature (2t) is involved for special remapping approach
        if "2t" in sfvars:
            sfvars.append("z")
        if "2t" in fc_sfvars:
            fc_sfvars.append("2t")

        grid_des_tar, grid_des_coarse = (
            gdes_dict["tar_grid_des"],
            gdes_dict["coa_grid_des"],
        )

        # initailize counter for warnings
        nwarn = 0
        for year_month in year_months:
            assert isinstance(
                year_month, dt.datetime
            ), "%{0}: All year_months-argument must be a datetime-object. Current one is of type '{1}'".format(
                method, type(year_month)
            )

            year, month = int(year_month.strftime("%Y")), int(year_month.strftime("%m"))
            year_str, month_str = str(year), "{0:02d}".format(int(month))
            last_day = last_day_of_month(year_month)
            subdir_name = year_month.strftime("%Y-%m")

            # construct directory and file names
            dir_curr_era5 = os.path.join(dirin_era5, year_str, month_str)
            dir_curr_ifs = os.path.join(dirin_ifs, year_str, subdir_name)
            dest_dir = os.path.join(dirout, "netcdf_data", year_str, subdir_name)
            # names of monthly data files (final_file_era5 and final_file_ifs will be merged to final_file)
            final_file = os.path.join(dest_dir, "preproc_{0}.nc".format(subdir_name))
            final_file_era5, final_file_ifs = final_file.replace(
                ".nc", "_era5.nc"
            ), final_file.replace(".nc", "_ifs.nc")
            os.makedirs(dest_dir, exist_ok=True)

            # further sanity checks
            if not os.path.isdir(dir_curr_era5):
                err_mess = "%{0}: Could not find directory for ERA5-data '{1}'".format(
                    method, dir_curr_era5
                )
                logger.fatal(err_mess)
                raise NotADirectoryError(err_mess)

            if not os.path.isdir(dir_curr_ifs):
                err_mess = "%{0}: Could not find directory for IFS-data '{1}'".format(
                    method, dir_curr_ifs
                )
                logger.fatal(err_mess)
                raise NotADirectoryError(err_mess)

            dates2op = pd.date_range(
                dt.datetime.strptime(
                    "{0}{1}0100".format(year_str, month_str), "%Y%m%d%H"
                ),
                last_day.replace(hour=23),
                freq="H",
            )

            # Perform logging, reset warning counter and loop over dates...
            logger.info("Start preprocessing data for month {0}...".format(subdir_name))

            for date2op in dates2op:
                # !!!!!! ML: Preliminary fix to avoid processing data from 2015 !!!!!!
                # if date2op <= dt.datetime.strptime("20160101 12", "%Y%m%d %H"): continue
                date_str, date_pr = date2op.strftime("%Y%m%d%H"), date2op.strftime(
                    "%Y-%m-%d %H:00 UTC"
                )
                hourly_file_era5 = os.path.join(
                    dest_dir, "{}_preproc_era5.nc".format(date_str)
                )
                hourly_file_ifs = hourly_file_era5.replace("era5", "ifs")
                # Skip time step if file already exists
                if os.path.isfile(hourly_file_era5):
                    continue

                lfail, nwarn = PreprocessERA5toIFS.preprocess_era5_in(
                    dirin_era5,
                    invar_file,
                    hourly_file_era5,
                    date2op,
                    sfvars,
                    mlvars,
                    fc_sfvars,
                    fc_mlvars,
                    logger,
                    nwarn,
                    max_warn,
                )

                if not lfail:
                    continue  # skip hour if preprocessing ERA5-data failed
                # Skip time step if file already exists
                if os.path.isfile(hourly_file_ifs):
                    continue

                lfail, nwarn = preprocess_ifs_tar(
                    dirin_ifs,
                    hourly_file_ifs,
                    date2op,
                    grid_des_tar,
                    predictands,
                    logger,
                    nwarn,
                    max_warn,
                )

                if not lfail:
                    continue  # skip hour if preprocessing IFS-data failed

                # finally all temporary files for each time step and clean-up
                logger.info(f"Data for date {date_pr} successfully preprocessed.")

            # merge all time steps to monthly file and clean-up hourly files
            logger.info(
                "Merge all hourly files to monthly datafile '{0}'".format(final_file)
            )

            if not os.path.isfile(final_file_era5):
                all_hourly_files_era5 = glob.glob(
                    os.path.join(dest_dir, "*_preproc_era5.nc")
                )
                cdo.run(
                    all_hourly_files_era5 + [final_file_era5],
                    OrderedDict([("mergetime", "")]),
                )
                remove_files(all_hourly_files_era5, lbreak=True)

            if not os.path.isfile(final_file_ifs):
                all_hourly_files_ifs = glob.glob(
                    os.path.join(dest_dir, "*_preproc_ifs.nc")
                )
                cdo.run(
                    all_hourly_files_ifs + [final_file_ifs],
                    OrderedDict([("mergetime", "")]),
                )
                remove_files(all_hourly_files_ifs, lbreak=True)

            # remap input data and merge
            remap_and_merge_data(
                final_file_era5,
                final_file_ifs,
                final_file,
                grid_des_coarse,
                grid_des_tar,
                all_predictors,
                all_predictands,
                nwarn,
                max_warn,
            )

            # rename data variables
            _ = PreprocessERA5toIFS.add_varname_suffix(
                final_file, all_predictors, "_in"
            )
            _ = PreprocessERA5toIFS.add_varname_suffix(
                final_file, all_predictands, "_tar"
            )

        return nwarn

    @classmethod
    def add_varname_suffix(cls, nc_file: str, varnames: List, suffix: str):
        """
        Rename variables in netCDF-file by adding a suffix. Also adheres to the convention to use lower-case names!
        :param nc_file: netCDF-file to process
        :param varnames: (old) variable names to modify
        :param suffix: suffix to add to variable names
        :return: status-flag
        """
        varnames_new = [varname + suffix for varname in varnames]
        varnames_pair = [
            "{0},{1}".format(varnames[i], varnames_new[i].lower())
            for i in range(len(varnames))
        ]

        try:
            cls.ncrename.run([nc_file], OrderedDict([("-v", varnames_pair)]))
            stat = True
        except RuntimeError as err:
            print(
                "Could not rename all parsed variables: {0}".format(",".join(varnames))
            )
            raise err

        return stat

    @classmethod
    def preprocess_ifs_tar(
        cls,
        dirin_ifs: str,
        hourly_file: str,
        date: dt.datetime,
        grid_des_tar: str,
        predictands: dict,
        logger,
        nwarn,
        max_warn,
    ):
        """
        Retrieve the predictand data from the hourly IFS-dataset.
        """
        dest_dir = os.path.dirname(hourly_file)
        date_pr = date.strftime("%Y-%m-%d %H:00 UTC")
        lfail = True

        logger.info(f"Preprocess predictands from IFS forecast for {date_pr}")
        nwarn, file2merge = run_preproc_func(
            cls.process_ifs_file,
            [dirin_ifs, dest_dir, date, grid_des_tar, predictands],
            {},
            logger,
            nwarn,
            max_warn,
        )
        if os.path.isfile(file2merge):
            lfail = False

        return lfail, nwarn

    @classmethod
    def process_ifs_file(
        cls,
        dirin_ifs: str,
        target_dir: str,
        date2op: dt.datetime,
        fgdes_tar: str,
        predictands: dict,
    ) -> str:
        """
        Process IFS-file by slicing data to region of interest.
        :param dirin_ifs: top-level directory where IFS-data are placed (under <year>/<year>-<month>/-subdirectories)
        :param target_dir: Target directory to store the processed data in netCDF-files
        :param date2op: Date for which data should be processed
        :param fgdes_tar: grid description file for target (high-resolved) grid
        :param predictands: dictionary for predictand variables
        :return: path to processed netCDF-datafile
        """
        # handle date and create tmp-directory and -files
        date_str = date2op.strftime("%Y%m%d%H")
        ifs_file, fh = get_fc_file(dirin_ifs, date2op, model="ifs", suffix="sfc")
        tmp_dir = os.path.join(target_dir, "tmp_{0}".format(date_str))
        os.makedirs(tmp_dir, exist_ok=True)

        ftmp_hres = os.path.join(tmp_dir, "{0}_tar.nc".format(date_str))

        # get variables to retrieve from predictands-dictionary
        # ! TO-DO: Allow for variables given on pressure levels (in pl-files!) !
        if any(vartype != "sf" for vartype in predictands.keys()):
            raise ValueError(
                "Only surface variables (i.e. vartype 'sf') are currently supported for IFS data."
            )
        ifsvars = list(predictands["sf"].keys())

        # get slicing coordinates from target grid description file
        gdes_tar = CDOGridDes(fgdes_tar)
        gdes_dict = gdes_tar.grid_des_dict

        lonlatbox = (
            *gdes_tar.get_slice_coords(
                gdes_dict["xfirst"], gdes_dict["xinc"], gdes_dict["xsize"]
            ),
            *gdes_tar.get_slice_coords(
                gdes_dict["yfirst"], gdes_dict["yinc"], gdes_dict["ysize"]
            ),
        )

        cls.cdo.run(
            [ifs_file, ftmp_hres],
            OrderedDict(
                [
                    ("-seltimestep", "{0:d}".format(fh)),
                    ("-selname", ",".join(ifsvars)),
                    (
                        "-sellonlatbox",
                        "{0:.2f},{1:.2f},{2:.2f},{3:.2f}".format(*lonlatbox),
                    ),
                ]
            ),
        )

        return ftmp_hres

    @classmethod
    def preprocess_era5_in(
        cls,
        era5_dir: str,
        invar_file: str,
        hourly_file: str,
        date: dt.datetime,
        sfvars: List,
        mlvars: dict,
        fc_sfvars: List,
        fc_mlvars: dict,
        logger: logging.Logger,
        nwarn: int,
        max_warn: int,
    ):
        """
        Retrieve the predictor data from the hourly ERA5-dataset.
        """
        # construct date-strings, path to temp-directory and initialize filelist for later merging
        date_str, date_pr = date.strftime("%Y%m%d%H"), date.strftime(
            "%Y-%m-%d %H:00 UTC"
        )
        dest_dir = os.path.dirname(hourly_file)
        tmp_dir = os.path.join(dest_dir, "tmp_{0}".format(date_str))
        filelist = []
        lfail = False

        logger.info(f"Start preprocessing ERA5-data for {date_pr}")

        # process surface variables of ERA5 (predictors)
        if sfvars:
            sf_file = os.path.join(
                era5_dir,
                date.strftime("%Y"),
                date.strftime("%m"),
                "{0}_sf.grb".format(date_str),
            )
            logger.info(
                "Preprocess predictor from surface file '{0}' of ERA5-dataset for time step {1}".format(
                    sf_file, date_pr
                )
            )

            nwarn, file2merge = run_preproc_func(
                cls.process_sf_file,
                [sf_file, invar_file, dest_dir, date, sfvars],
                {},
                logger,
                nwarn,
                max_warn,
            )
            filelist = manage_filemerge(filelist, file2merge, tmp_dir)
            if not file2merge:
                lfail = True  # skip day if some data is missing

        # process multi-level variables of ERA5 (predictors)
        if mlvars and not lfail:
            ml_file = os.path.join(
                era5_dir,
                date.strftime("%Y"),
                date.strftime("%m"),
                "{0}_ml.grb".format(date_str),
            )
            logger.info(
                "Preprocess predictor from multi-level file '{0}' of ERA5-dataset for time step {1}".format(
                    ml_file, date_pr
                )
            )
            nwarn, file2merge = run_preproc_func(
                cls.process_ml_file,
                [ml_file, dest_dir, date, mlvars],
                {"interp": True},
                logger,
                nwarn,
                max_warn,
            )
            filelist = manage_filemerge(filelist, file2merge, tmp_dir)
            if not file2merge:
                lfail = True  # skip day if some data is missing

        # process forecasted surface variables of ERA5 (predictors)
        if fc_sfvars and not lfail:
            fc_file, _ = get_fc_file(era5_dir, date, model="era5", prefix="sf_fc")
            logger.info(
                "Preprocess predictor from surface fcst. file '{0}' of ERA5-dataset for time step {1}".format(
                    fc_file, date_pr
                )
            )
            nwarn, file2merge = run_preproc_func(
                cls.process_sf_file,
                [fc_file, invar_file, dest_dir, date, fc_sfvars],
                {},
                logger,
                nwarn,
                max_warn,
            )
            filelist = manage_filemerge(filelist, file2merge, tmp_dir)
            if not file2merge:
                lfail = True  # skip day if some data is missing

        # process forecasted multi-level variables of ERA5 (predictors)
        if fc_mlvars and not lfail:
            fc_file, _ = get_fc_file(era5_dir, date, model="era5", prefix="ml_fc")
            logger.info(
                "Preprocess predictor from surface fcst. file '{0}' of ERA5-dataset for time step {1}".format(
                    fc_file, date_pr
                )
            )
            nwarn, file2merge = run_preproc_func(
                cls.process_ml_file,
                [fc_file, dest_dir, date, fc_mlvars],
                {"interp": False},
                logger,
                nwarn,
                max_warn,
            )
            filelist = manage_filemerge(filelist, file2merge, tmp_dir)

        if filelist:
            logger.info(
                "Merge temporary ERA5-files to hourly netCDF-file '{0}'".format(
                    hourly_file
                )
            )
            cls.cdo.run(filelist + [hourly_file], OrderedDict([("merge", "")]))

        if os.path.isfile(hourly_file):
            lfail = False
            remove_files(filelist, lbreak=True)

        return lfail, nwarn

    @classmethod
    def process_ml_file(
        cls,
        ml_file: str,
        target_dir: str,
        date2op: dt.datetime,
        mlvars: dict,
        interp: bool = True,
    ) -> str:
        """
        Process multi-level ERA5-file, i.e. interpolate on desired pressure levels, remap conservatively on coarsened
        grid and finally perform a bilinear remapping onto the target (high-resolved) grid.
        :param ml_file: ERA5-file with variables on multi-levels or pressure-levels to process
        :param target_dir: Target directory to store the processed data in netCDF-files
        :param date2op: Date for which data should be processed
        :param mlvars: dictionary of predictor variables to be interpolated onto pressure levels,
                        e.g. {"t": {"p85000", "p70000"}}
        :param interp: True if pressure interpolation is required or False if data is available on pressure levels
        :return: path to processed netCDF-datafile
        """
        # handle date and create tmp-directory and -files
        date_str = date2op.strftime("%Y%m%d%H")
        tmp_dir = os.path.join(target_dir, "tmp_{0}".format(date_str))
        os.makedirs(tmp_dir, exist_ok=True)

        if not os.path.isfile(ml_file):
            raise FileNotFoundError(
                "%Could not find required multi level-file '{ml_file}'"
            )

        # construct filenames for all temporary files
        ftmp_plvl1 = os.path.join(tmp_dir, "{0}_plvl.nc".format(date_str))
        ftmp_plvl2 = ftmp_plvl1.replace("plvl.nc", "plvl_all.nc")
        ftmp_hres = os.path.join(tmp_dir, f"{date_str}_ml_hres.nc")

        # Create lists of variables as well as pressure strings required for pressure interpolation
        mlvars_list = list(mlvars.keys())
        mlvars_list.remove("plvls")
        mlvars_list_interp = mlvars_list + ["t", "lnsp", "z"]
        plvl_strs = ",".join(["{0:d}".format(int(plvl)) for plvl in mlvars["plvls"]])
        var_new_req = get_varnames_from_mlvars(mlvars)

        # interpolate variables of interest onto pressure levels
        if interp:
            cls.cdo.run(
                [ml_file, ftmp_plvl1],
                OrderedDict(
                    [
                        ("--eccodes", ""),
                        ("-f nc", ""),
                        ("copy", ""),
                        ("-selname", ",".join(mlvars_list)),
                        ("-ml2plx", plvl_strs),
                        ("-selname", ",".join(mlvars_list_interp)),
                        ("-sellonlatbox", "0.,30.,30.,60."),
                    ]
                ),
            )
        else:
            cls.cdo.run(
                [ml_file, ftmp_plvl1],
                OrderedDict(
                    [
                        ("--eccodes", ""),
                        ("-f nc", ""),
                        ("copy", ""),
                        ("-selname", ",".join(mlvars_list)),
                        ("-sellevel", plvl_strs),
                        ("-sellonlatbox", "0.,30.,30.,60."),
                    ]
                ),
            )

        # Split pressure-levels into seperate files and ...
        cls.cdo.run(
            [ftmp_plvl1, ftmp_plvl1.rstrip(".nc")], OrderedDict([("splitlevel", "")])
        )
        # ... rename variables accordingly in each resulting file
        for plvl in mlvars["plvls"]:
            curr_file = ftmp_plvl1.replace(".nc", "{0:06d}.nc".format(int(plvl)))
            # trick to remove singleton plev- while keeping time-dimension
            cls.ncwa.run(
                [curr_file, curr_file], OrderedDict([("-O", ""), ("-a", "plev")])
            )
            cls.ncks.run(
                [curr_file, curr_file],
                OrderedDict([("-O", ""), ("-x", ""), ("-v", "plev")]),
            )

            for var in mlvars_list:
                var_new = "{0}{1:d}".format(var, int(plvl / 100.0))
                cls.ncrename.run(
                    [ftmp_plvl1.replace(".nc", "{0:06d}.nc".format(int(plvl)))],
                    OrderedDict([("-v", "{0},{1}".format(var, var_new))]),
                )

        # concatenate pressure-level files, reduce to final variables of interest and do the remapping steps
        cls.cdo.run(
            [ftmp_plvl1.replace(".nc", "??????.nc"), ftmp_plvl2],
            OrderedDict([("-O", ""), ("merge", "")]),
        )
        cls.cdo.run(
            [ftmp_plvl2, ftmp_hres], OrderedDict([("-selname", ",".join(var_new_req))])
        )

        # clean-up temporary files and rename variables
        plvl_files = list(glob.glob(ftmp_plvl1.replace(".nc", "??????.nc")))
        remove_files(plvl_files + [ftmp_plvl1, ftmp_plvl2], lbreak=False)

        return ftmp_hres

    @classmethod
    def process_sf_file(
        cls,
        sf_file: str,
        invar_file: str,
        target_dir: str,
        date2op: dt.datetime,
        sfvars: List,
    ) -> str:
        """
        Process surface ERA5-file, i.e. remap conservatively on coarsened grid followed by bilinear remapping
        onto the target (high-resolved) grid.
        :param sf_file: ERA5-file with surface variables to process
        :param invar_file: ERA5-file with invariant variables
        :param target_dir: Target directory to store the processed data in netCDF-files
        :param date2op: Date for which data should be processed
        :param sfvars: list of surface predictor variables
        :return: path to processed netCDF-datafile
        """
        # handle date and create tmp-directory and -files
        date_str = date2op.strftime("%Y%m%d%H")
        tmp_dir = os.path.join(target_dir, "tmp_{0}".format(date_str))
        os.makedirs(tmp_dir, exist_ok=True)

        if not os.path.isfile(sf_file):
            raise FileNotFoundError(
                f"%Could not find required surface-file '{sf_file}'"
            )

        ftmp_hres = os.path.join(tmp_dir, f"{date_str}_sf_hres.nc")

        # handle dynamical and invariant variables
        sfvars_stat, sfvars_dyn = split_dyn_static(sfvars)

        # run remapping
        cls.cdo.run(
            [sf_file, ftmp_hres],
            OrderedDict(
                [
                    ("--eccodes", ""),
                    ("-f nc", ""),
                    ("copy", ""),
                    ("-selname", ",".join(sfvars_dyn)),
                    ("-sellonlatbox", "0.,30.,30.,60."),
                ]
            ),
        )
        if sfvars_stat:
            ftmp_hres2 = ftmp_hres.replace("sf", "sf_stat")
            if not os.path.isfile(ftmp_hres2):  # has only to be done once
                cls.cdo.run(
                    [invar_file, ftmp_hres2],
                    OrderedDict(
                        [
                            ("--eccodes", ""),
                            ("-f nc", ""),
                            ("copy", ""),
                            ("-selname", ",".join(sfvars_stat)),
                            ("-sellonlatbox", "0.,30.,30.,60."),
                        ]
                    ),
                )
            # NOTE: ftmp_hres must be at first position to overwrite time-dimension of ftmp_hres2
            #       which would not fit since it is retrieved from an invariant datafile with arbitrary timestamp
            #       This works at least for CDO 2.0.2!!!
            cls.cdo.run(
                [ftmp_hres, ftmp_hres2, ftmp_hres],
                OrderedDict([("-O", ""), ("merge", "")]),
            )
            # clean-up temporary files
            remove_files([ftmp_hres2], lbreak=False)

        return ftmp_hres

    @classmethod
    def remap_and_merge_data(
        cls,
        file_in: str,
        file_tar: str,
        final_file: str,
        gdes_coarse: str,
        gdes_tar: str,
        predictors: List,
        predictands: List,
        nwarn: int,
        max_warn: int,
    ) -> int:
        """
        Perform the remapping step on the predictor data and finally merge it with the predictand data
        :param file_in: netCDF-file with predictor data
        :param file_tar: netCDF-file with predictand data
        :param final_file: name of the resulting merged netCDF-file
        :param gdes_coarse: CDO grid description file corresponding to the coarse-grained predictor data
        :param gdes_tar: CDO grid description file corresponding to the high-resolved predictand data
        :param predictors: list of (actual) predictor variables (Note: file_in may comprise more variables)
        :param predictands: list of predictand variables
        :param nwarn: current number of issued warnings
        :param max_warn: maximum allowed number of warnings
        :return: updated nwarn and resulting merged netCDF-file
        """
        cdo = PreprocessERA5toIFS.cdo

        if not file_in.endswith(".nc"):
            raise ValueError(f"Input data-file '{file_in}' must be a netCDF-file.")
        file_in_coa = file_in.replace(".nc", "_coa.nc")
        file_in_hres = file_in.replace(".nc", "_hres.nc")

        # preprocess input data
        if "2t" in predictors:
            predictors.remove("2t")
            l2t = True
        else:
            l2t = False

        cdo.run(
            [file_in, file_in_coa],
            OrderedDict(
                [("-remapcon", gdes_coarse), ("-selname", ",".join(predictors))]
            ),
        )
        cdo.run([file_in_coa, file_in_hres], OrderedDict([("-remapbil", gdes_tar)]))

        if l2t:
            cls.remap2t_and_cat(file_in, file_in_hres, gdes_coarse, gdes_tar)
            predictors.append("2t")  # to ensure subsequent renaming

        # merge input and target data
        stat = merge_two_netcdf(file_in_hres, file_tar, final_file)
        # cdo.run([file_in_hres, file_tar, final_file], OrderedDict([("-merge", "")]))
        if not (stat and os.path.isfile(final_file)):
            nwarn = max_warn + 1
        else:
            remove_files([file_in_coa, file_in_hres, file_tar], lbreak=True)

        return nwarn

    @classmethod
    def remap2t_and_cat(
        cls, infile: str, outfile: str, grid_des_coarse: str, grid_des_tar: str
    ) -> None:
        """
        Remap 2m temperature by transforming to dry static energy and concatenate outfile with the result.
        First, data is conservative remapping onto the coarse grid is performed, followed by bilinear remapping onto the
        target grid.
        :param infile: input data file with 2m temperature (2t)
        :param outfile: output-file which will be concatenated
        :param grid_des_coarse: grid description file for coarse grid
        :param grid_des_tar: grid description file for target (high-resolved) grid.
        :return:
        """
        # temporary files (since CDO does not support modifying the input-file in place)
        ftmp_coarse = outfile.replace(".nc", "_s_coarse.nc")
        ftmp_hres = outfile.replace(".nc", "_2t_tmp.nc")

        cls.cdo.run(
            [infile, ftmp_coarse],
            OrderedDict(
                [
                    ("-remapcon", grid_des_coarse),
                    ("-selname", "s,z"),
                    ("-aexpr", f"'s={cls.cpd}*2t+z+{cls.g}*2'"),
                ]
            ),
        )
        cls.cdo.run(
            [ftmp_coarse, ftmp_hres],
            OrderedDict(
                [
                    ("-remapbil", grid_des_tar),
                    ("-selname", "2t"),
                    ("-aexpr", f"'2t=(s-z-{cls.cpd}*2)/{cls.g}'"),
                ]
            ),
        )
        cls.cdo.run(
            [ftmp_hres, outfile, outfile], OrderedDict([("-O", ""), ("merge", "")])
        )

        # clean-up temporary files
        remove_files([ftmp_coarse, ftmp_hres], lbreak=False)
