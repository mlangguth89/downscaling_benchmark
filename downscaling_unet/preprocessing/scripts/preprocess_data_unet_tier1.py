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


class Preprocess_Unet_Tier1(Abstract_Preprocessing):

    def __init__(self, source_dir, output_dir):
        """
        Initialize class for tier-1 downscaling dataset.
        """
        super().__init__("preprocess_unet_tier1", source_dir, output_dir)

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

