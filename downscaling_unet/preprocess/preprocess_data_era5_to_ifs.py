__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-03-16"
__update__ = "2022-03-18"

# doc-string
"""
Main script to preprocess ERA5 data (provided on a 0.3°-grid) for first real downscaling application. 
The target of the downscaling will be IFS HRES data on a 0.1°-grid as in preprocess_data_unet_tier1.py.
Contrarily to the previous, simplified approach, no slicing regarding daytime and season will be performed. 
Thus, hourly input data are produced from the ERA5-dataset. For the output, hourly target data, 
IFS forecasts with lead time 3 to 14 hours is used. 
"""
# doc-string

import os, glob
from typing import Union, List
import shutil
import logging
import numbers
import datetime as dt
import numpy as np
from collections import OrderedDict
#from tfrecords_utils import IFS2TFRecords
from other_utils import to_list
from pystager_utils import PyStager
from abstract_preprocess import Abstract_Preprocessing
from preprocess_data_unet_tier1 import Preprocess_Unet_Tier1
from tools_utils import CDO, NCRENAME, NCAP2, NCKS, NCEA

number = Union[float, int]
num_or_List = Union[number, List[number]]
list_or_tuple = Union[List, tuple]

class Preprocess_ERA5_to_IFS(Abstract_Preprocessing):

    # expected key of grid description files
    expected_keys_gdes = ["gridtype", "xsize", "ysize", "xfirst", "xinc", "yfirst", "yinc"]

    def __init__(self, source_dir: str, output_dir: str, grid_des_tar: str, downscaling_fac: int = None):
        """
        Initialize class for tier-1 downscaling dataset.
        """
        super().__init__("preprocess_ERA5_to_IFS", source_dir, output_dir)

        if not os.path.isfile(grid_des_tar):
            raise FileNotFoundError("Preprocess_Unet_Tier1: Could not find target grid description file '{0}'"
                                    .format(grid_des_tar))
        self.grid_des_tar = grid_des_tar
        self.my_rank = None                     # to be set in __call__

    def __call__(self, years: List, season: str, jobname: str = "dummy", **kwargs):
        """
        Run preprocessing.
        :param years: List of years to be processed.
        :param months: List of months to be processed.
        :param jobname: jobname-identifier for PyStager
        """
        method = Preprocess_ERA5_to_IFS.__call__.__name__

        years = to_list(years)
        # sanity checks on years and season arguments
        assert all([isinstance(year, numbers.Number) for year in years]), \
            "%{0}: All elements of years must be numbers".format(method)

        years = [int(year) for year in years]
        months = Preprocess_ERA5_to_IFS.check_season(season)

        # instantiate PyStager
        preprocess_pystager = PyStager(self.preprocess_worker, "year_month_list", nmax_warn=3)
        self.my_rank = preprocess_pystager.my_rank

        tar_grid_des_dict = Preprocess_ERA5_to_IFS.read_grid_des(self.grid_des_tar)
        # use process_tar_grid_des-method from Preprocess_Unet_Tier1
        base_gdes_d, coa_gdes_d = Preprocess_Unet_Tier1.process_tar_grid_des(self, tar_grid_des_dict, **kwargs)
        gdes_dict = {"tar_grid_des": tar_grid_des_dict, "base_grid_des": base_gdes_d, "coa_grid_des": coa_gdes_d}

        preprocess_pystager.setup(years, months)
        preprocess_pystager.run(self.source_dir, self.target_dir, gdes_dict, job_name=jobname)

    @staticmethod
    def preprocess_worker(year_months: List, dirin_era5: str, dirin_ifs: str, dirout: str, gdes_dict: dict,
                          logger: logging.Logger, nmax_warn: int = 3):
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
        method = Preprocess_ERA5_to_IFS.preprocess_worker.__name__

        # sanity checks
        assert isinstance(logger, logging.Logger), "%{0}: logger-argument must be a logging.Logger instance" \
                                                   .format(method)

        grid_des_tar, grid_des_base, grid_des_coarse = gdes_dict["tar_grid_des"], gdes_dict["base_grid_des"], \
                                                       gdes_dict["coa_grid_des"]
        for year_month in year_months:
            assert isinstance(year_month, dt.datetime),\
                "%{0}: All year_months-argument must be a datetime-object. Current one is of type '{1}'"\
                .format(method, type(year_month))

            year, month = int(year_month.strftime("%Y")), int(year_month.strftime("%m"))
            year_str, month_str = str(year), "{0:02d}".format(int(month))

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

    @staticmethod
    def check_season(season: str):
        """
        Check if season-string is known.
        :param season: the seson string identifier
        :return: corresponding months as list of integers
        """
        method = Preprocess_ERA5_to_IFS.check_season.__name__

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