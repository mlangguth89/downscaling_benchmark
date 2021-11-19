# ********** Info **********
# @Creation: 2021-08-01
# @Update: 2021-08-01
# @Author: Michael Langguth
# @Site: Juelich supercomputing Centre (JSC) @ FZJ
# @File: preproces_downscaling_data.py
# ********** Info **********

# doc-string
"""
Main script to preprocess IFS HRES data for downscaling with UNet-architecture.
"""
# doc-string

import os, glob
import shutil
import argparse
import logging
import subprocess as sp
import datetime as dt
from tfrecords_utils import IFS2TFRecords
from pystager_utils import PyStager

scr_name = "preprocess_downsclaing_data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_parent_dir", "-src_dir", dest="src_dir", type=str,
                        default="/p/scratch/deepacf/maelstrom/maelstrom_data/ifs_hres",
                        help="Top-level directory under which IFS HRES data are stored with subdirectories " +
                             "<year>/<month>.")
    parser.add_argument("--out_parent_dir", "-out_dir", dest="out_dir", type=str, required=True,
                        help="Top-level directory under which remapped data will be stored.")
    parser.add_argument("--years", "-y", dest="years", type=int, nargs="+", default=[2016, 2017, 2018, 2019, 2020],
                        help="Years of data to be preprocessed.")
    parser.add_argument("--months", "-m", dest="months", type=int, nargs="+", default=range(3, 10),
                        help="Months of data to be preprocessed.")

    args = parser.parse_args()
    dir_in = args.src_dir
    dir_out = args.out_dir
    years = args.years
    months = args.months

    if not os.path.isdir(dir_in):
        raise NotADirectoryError("%{0}: Parsed source directory does not exist.".format(scr_name))

    if not os.path.isdir(dir_out):
        os.makedirs(dir_out, exist_ok=True)
        print("%{0}: Created output-directory for remapped data '{1}'".format(scr_name, dir_out))

    ifs_hres_pystager = PyStager(preprocess_worker, "year_month_list", nmax_warn=3)
    ifs_hres_pystager.setup(years, months)
    ifs_hres_pystager.run(dir_in, dir_out)


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
    method = preprocess_worker.__name__

    nwarns = 0
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
                #logger.info("%{0}: Processing of netCDF-files already done.".format(method))
                _ = sp.check_output(cmd, shell=True)
                nc_file_new = os.path.basename(nc_file).replace(".nc", "_remapped.nc")
                shutil.move(nc_file.replace(".nc", "_remapped.nc"), os.path.join(dest_nc_dir, nc_file_new))
                logger.info("%{0} Data has been remapped successfully and moved to '{1}'-directory."
                            .format(method, dest_nc_dir))
            except Exception as err:
                nwarns += 1
                logger.debug("%{0}: A problem was faced when handling file '{1}'.".format(method, nc_file) +
                             "Remapping of this file presumably failed.")
                if nwarns > nmax_warn:
                    logger.critical("%{0}: More warnings triggered than allowed ({1:d}). ".format(method, nmax_warn) +
                                    "Job will be trerminated and see error below.")
                    raise err
                else:
                    pass

        # move remapped data to own directory
        tfr_data_dir = os.path.join(dir_out, "tfr_data")
        ifs_tfr = IFS2TFRecords(tfr_data_dir, os.path.join(dest_nc_dir, os.path.basename(nc_files[0])
                                                           .replace(".nc", "_remapped.nc")))
        ifs_tfr.get_and_write_metadata()
        logger.info("%{0}: IFS2TFRecords-class instance has been initalized successully.".format(method))
        try:
            ifs_tfr.write_monthly_data_to_tfr(dest_nc_dir, patt="*remapped.nc")
        except Exception as err:
            logger.critical("%{0}: Error when writing TFRecord-file. Investigate error-message below.".format(method))
            raise err

        logger.info("%{0}: TFRecord-files have been created succesfully under '{1}'".format(method, tfr_data_dir))
        logger.info("%{0}: During processing {1:d} warnings have been faced.".format(method, nwarns))

    return nwarns


if __name__ == "__main__":
    main()
