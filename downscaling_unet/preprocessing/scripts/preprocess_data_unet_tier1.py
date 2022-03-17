__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-03-16"
__update__ = "2022-03-16"

# doc-string
"""
Main script to preprocess IFS HRES data for downscaling with UNet-architecture to create tier-1 dataset.
This inlcudes only 2m temperature ad surface elevation extracted from forecast hour 0 of IFS runs (i.e. 00 and 12 UTC).
Further details are provided in deliverable 1.1. of the MAELSTROM project (AP 5 therein): 
https://www.maelstrom-eurohpc.eu/content/docs/upload/doc6.pdf
"""
# doc-string

import os, glob
import shutil
import argparse
import logging
import numbers
import subprocess as sp
import datetime as dt
from typing import Union, List
from tfrecords_utils import IFS2TFRecords
from tools_utils import to_list
from pystager_utils import PyStager
from abstract_preprocess import Abstract_Preprocessing

number = Union[float, int]
num_or_List = Union[number, List[number]]
list_or_tuple = Union[List, tuple]


class Preprocess_Unet_Tier1(Abstract_Preprocessing):

    def __init__(self, source_dir: str, output_dir: str, grid_des_tar: str):
        """
        Initialize class for tier-1 downscaling dataset.
        """
        super().__init__("preprocess_unet_tier1", source_dir, output_dir)

        if not os.path.isfile(grid_des_tar):
            raise FileNotFoundError("Preprocess_Unet_Tier1: Could not find target grid description file '{0}'"
                                    .format(grid_des_tar))
        self.grid_des_tar = grid_des_tar

    def __call__(self, years: List, months: List, logger: logging.Logger = None):
        """
        Run preprocessing.
        :param years: List of years to be processed.
        :param months: List of months to be processed.
        :param logger: logger-object
        """
        method = Preprocess_Unet_Tier1.__call__.__name__

        years = to_list(years)
        months = to_list(months)
        # sanity checks
        assert all([isinstance(year, numbers.Number) for year in years]),\
            "%{0}: All elements of years must be numbers".format(method)

        assert all([(isinstance(month, numbers.Number) and (1 <= int(month) <= 12)) for month in months]), \
            "%{0}: All elements of months must be numbers between 1 and 12.".format(method)

        years = [int(year) for year in years]
        months = [int(month) for month in months]

        tar_grid_des_dict = Preprocess_Unet_Tier1.read_grid_des(self.grid_des_tar)
        base_grid_des, coarse_grid_des, sw_xy, nxy, dx_c = Preprocess_Unet_Tier1.process_tar_grid_des(tar_grid_des_dict)

        preprocess_pystager = PyStager(self.preprocess_worker, "year_month_list", nmax_warn=3)
        preprocess_pystager.setup(years, months)
        preprocess_pystager.run(self.source_dir, self.target_dir)

    @staticmethod
    def preprocess_worker(year_months: list, dir_in: str, dir_out: str, logger: logging.Logger,
                          nmax_warn: int = 3, hour: int = None):
        """
        Function that runs job of an individual worker.
        :param year_months: Datetime-objdect indicating year and month for which data should be preprocessed
        :param dir_in: Top-level input directory for original IFS HRED netCDF-files
        :param dir_out: Top-level output directory wheer netCDF-files and TFRecords of remapped data will be stored
        :param logger: Logging instance for log process on worker
        :param nmax_warn: allowed maximum number of warnings/problems met during processing (default:3)
        :param hour: hour of the dy for which data should be preprocessed (default: None)
        :return: number of warnings/problems met during processing (if they do not trigger an error)
        """
        method = Preprocess_Unet_Tier1.preprocess_worker.__name__

        this_dir = os.path.dirname(os.path.realpath(__file__))

        for year_month in year_months:
            assert isinstance(year_month, dt.datetime),\
                "%{0}: All year_months-argument must be a datetime-object. Current one is of type '{1}'"\
                .format(method, type(year_month))

            year, month = int(year_month.strftime("%Y")), int(year_month.strftime("%m"))
            year_str, month_str = str(year), "{0:02d}".format(int(month))
            hh_str = "*[0-2][0-9]" if hour is None else "{0:02d}".format(int(hour))

            subdir = year_month.strftime("%Y-%m")
            dirr_curr = os.path.join(dir_in, str(year), subdir)
            dest_nc_dir = os.path.join(dir_out, "netcdf_data", year_str, subdir)
            os.makedirs(dest_nc_dir, exist_ok=True)


            assert isinstance(logger, logging.Logger), "%{0}: logger-argument must be a logging.Logger instance"\
                .format(method)

            if not os.path.isdir(dirr_curr):
                err_mess = "%{0}: Could not find directory '{1}'".format(method, dirr_curr)
                logger.critical(err_mess)
                raise NotADirectoryError(err_mess)

            search_patt = os.path.join(dirr_curr, "sfc_{0}{1}*_{2}.nc".format(year_str, month_str, hh_str))
            logger.info("%{0}: Serach for netCDF-files under '{1}' for year {2}, month {3} and hour {4}"
                        .format(method, dirr_curr, year_str, month_str, hh_str))
            nc_files = glob.glob(search_patt)

            if not nc_files:
                err_mess = "%{0}: Could not find any netCDF-file in '{1}' with search pattern '{2}'"\
                    .format(method, dirr_curr, search_patt)
                logger.critical(err_mess)
                raise FileNotFoundError(err_mess)

            nfiles = len(nc_files)
            logger.info("%{0}: Found {1:d} files under '{2}' for preprocessing.".format(method, nfiles, dirr_curr))
            nwarns = 0
            # Perform remapping
            for i, nc_file in enumerate(nc_files):
                logger.info("%{0}: Start remapping of data from file '{1}' ({2:d}/{3:d})"
                            .format(method, nc_file, i+1, nfiles))
                cmd = "{0} {1}".format(os.path.join(this_dir, "coarsen_ifs_hres.sh"), nc_file)
                try:
                    _ = sp.check_output(cmd, shell=True)
                    shutil.move(nc_file.replace(".nc", "_remapped.nc"), os.path.join(dest_nc_dir, nc_file_new))
                    logger.info("%{0} Data has been remapped successfully and moved to '{1}'-directory."
                                .format(method, dest_nc_dir))
                except Exception as err:
                    nwarns += 1
                    logger.debug("%{0}: A problem was faced when handling file '{1}'.".format(method, nc_file) +
                                 "Remapping of this file presumably failed.")
                    if nwarns > nmax_warn:
                        logger.critical(
                            "%{0}: More warnings triggered than allowed ({1:d}). ".format(method, nmax_warn) +
                            "Job will be trerminated and see error below.")
                        raise err
                    else:
                        pass
            # Conversion to TFRecords does not work yet (2021-11-19)
            # move remapped data to own directory
            # tfr_data_dir = os.path.join(dir_out, "tfr_data")
            # ifs_tfr = IFS2TFRecords(tfr_data_dir, os.path.join(dest_nc_dir, os.path.basename(nc_files[0])
            #                                                    .replace(".nc", "_remapped.nc")))
            # ifs_tfr.get_and_write_metadata()
            # logger.info("%{0}: IFS2TFRecords-class instance has been initalized successully.".format(method))
            # try:
            #     ifs_tfr.write_monthly_data_to_tfr(dest_nc_dir, patt="*remapped.nc")
            # except Exception as err:
            #     logger.critical("%{0}: Error when writing TFRecord-file. Investigate error-message below.".format(method))
            #     raise err
            #
            # logger.info("%{0}: TFRecord-files have been created succesfully under '{1}'".format(method, tfr_data_dir))
            # logger.info("%{0}: During processing {1:d} warnings have been faced.".format(method, nwarns))
            #
            # logger.info("%{0}: TFRecord-files have been created succesfully under '{1}'".format(method, tfr_data_dir))
            # logger.info("%{0}: During processing {1:d} warnings have been faced.".format(method, nwarns))

        return nwarns

    @staticmethod
    def process_tar_grid_des(tar_grid_des: dict):

        expected_keys = ["lonlat", "xsize", "ysize", "xfirst", "xinc", "yfirst"]





