# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Script to download ERA5 data from the CDS API.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2023-11-22"
__update__ = "2023-08-22"

# import modules
import os, sys
import json as js
import logging
import argparse
from download_era5_data import ERA5_Data_Loader

# get logger
logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

def main(parser_args):

    data_dir = parser_args.data_dir
    
    # read configuration files for model and dataset
    with parser_args.data_req_file as fdreq:
        req_dict = js.load(fdreq)


    # create output directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # create logger handlers
    logfile = os.path.join(data_dir, f"download_era5_{parser_args.exp_name}.log")
    if os.path.isfile(logfile): os.remove(logfile)
    fh = logging.FileHandler(logfile)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    fh.setLevel(logging.INFO)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh), logger.addHandler(ch)

    # create data loader instance
    data_loader = ERA5_Data_Loader(parser_args.nworkers)

    # download data
    _ = data_loader(req_dict, data_dir, parser_args.start, parser_args.end, parser_args.format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", "-data_dir", dest="data_dir", type=str, required=True,
                        help="Directory where test dataset (netCDF-file) is stored.")
    parser.add_argument("--data_request_file", "-data_req_file", dest="data_req_file", type=argparse.FileType("r"), required=True,
                        help="File containing data request information for the CDS API.")
    parser.add_argument("--year_start", "-start", dest="start", type=int, default=1995, 
                        help="Start year of ERA5-data request.")
    parser.add_argument("--year_end", "-end", dest="end", type=int, default=2019,
                        help="End year of ERA5-data request.")
    parser.add_argument("--nworkers", "-nw", dest="nowrkers", type=int, default=4,
                        help="Number of workers to download ERA5 data.")
    parser.add_argument("--format", "-format", dest="format", type=str, default="netcdf",
                        help="Format of downloaded data.")

    args = parser.parse_args()
    main(args)