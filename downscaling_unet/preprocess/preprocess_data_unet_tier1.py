__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-03-16"
__update__ = "2022-04-29"

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
import logging
import numbers
import datetime as dt
import numpy as np
from collections import OrderedDict
from typing import Union, List
#from tfrecords_utils import IFS2TFRecords
from other_utils import to_list
from pystager_utils import PyStager
from abstract_preprocess import AbstractPreprocessing, CDOGridDes
from tools_utils import CDO, NCRENAME, NCAP2, NCKS, NCEA

number = Union[float, int]
num_or_List = Union[number, List[number]]
list_or_tuple = Union[List, tuple]


class Preprocess_Unet_Tier1(AbstractPreprocessing):

    # expected key of grid description files
    expected_keys_gdes = ["gridtype", "xsize", "ysize", "xfirst", "xinc", "yfirst", "yinc"]

    def __init__(self, source_dir: str, output_dir: str, grid_des_tar: str, downscaling_fac: int = 8):
        """
        Initialize class for tier-1 downscaling dataset.
        Pure downscaling task. Thus, pass None for source_dir_out to initializer.
        Following Sha et al., 2020, 2m temperature and surface elevation act as predictors and predictands.
        """
        super().__init__("preprocess_unet_tier1", source_dir, None, {"sf": {"2t": None, "z": None}},
                         {"sf": {"2t": None, "z": None}}, output_dir)

        if not os.path.isfile(grid_des_tar):
            raise FileNotFoundError("Preprocess_Unet_Tier1: Could not find target grid description file '{0}'"
                                    .format(grid_des_tar))
        self.source_dir = self.source_dir_in        # set source_dir for backwards compatability
        self.grid_des_tar = grid_des_tar
        self.my_rank = None                     # to be set in __call__
        self.downscaling_fac = downscaling_fac

    def prepare_worker(self, years: List, months: List, **kwargs):
        """
        Prepare workers for preprocessing.
        :param years: List of years to be processed.
        :param months: List of months to be processed.
        :param kwargs:
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

        # initialize and set-up Pystager
        preprocess_pystager = PyStager(self.preprocess_worker, "year_month_list", nmax_warn=3)
        preprocess_pystager.setup(years, months)

        # Create grid description files needed for preprocessing (requires rank-information)
        self.my_rank = preprocess_pystager.my_rank

        ifs_grid_des = CDOGridDes(self.grid_des_tar)
        base_gdes_d, coa_gdes_d = ifs_grid_des.create_coarsened_grid_des(self.target_dir, self.downscaling_fac,
                                                                         self.my_rank, name_base="ifs_hres_")
        gdes_dict = {"tar_grid_des": ifs_grid_des.grid_des_dict, "base_grid_des": base_gdes_d, "coa_grid_des": coa_gdes_d}
        # define arguments and keyword arguments for running PyStager later
        run_dict = {"args": [self.source_dir, self.target_dir, gdes_dict],
                    "kwargs": {"job_name": kwargs.get("jobname", "Preproc_Unet_tier1")}}

        return preprocess_pystager, run_dict

    @staticmethod
    def preprocess_worker(year_months: list, dir_in: str, dir_out: str, gdes_dict: dict, logger: logging.Logger,
                          nmax_warn: int = 3, hour: int = None):
        """
        Function that runs job of an individual worker.
        :param year_months: Datetime-objdect indicating year and month for which data should be preprocessed
        :param dir_in: Top-level input directory for original IFS HRED netCDF-files
        :param dir_out: Top-level output directory wheer netCDF-files and TFRecords of remapped data will be stored
        :param gdes_dict: dictionary containing grid description dictionaries for target, base and coarse grid
        :param logger: Logging instance for log process on worker
        :param nmax_warn: allowed maximum number of warnings/problems met during processing (default:3)
        :param hour: hour of the dy for which data should be preprocessed (default: None)
        :return: number of warnings/problems met during processing (if they do not trigger an error)
        """
        method = Preprocess_Unet_Tier1.preprocess_worker.__name__

        grid_des_tar, grid_des_base, grid_des_coarse = gdes_dict["tar_grid_des"], gdes_dict["base_grid_des"], \
                                                       gdes_dict["coa_grid_des"]
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
                try:
                    _ = Preprocess_Unet_Tier1.process_one_file(nc_file, grid_des_tar, grid_des_coarse, grid_des_base)
                    nc_file_new = os.path.basename(nc_file).replace(".nc", "_remapped.nc")
                    shutil.move(nc_file.replace(".nc", "_remapped.nc"), os.path.join(dest_nc_dir, nc_file_new))
                    logger.info("%{0} Data has been remapped successfully and moved to '{1}'-directory."
                                .format(method, dest_nc_dir))
                except Exception as err:
                    nwarns += 1
                    logger.debug("%{0}: A problem was faced when handling file '{1}'.".format(method, nc_file) +
                                 " Remapping of this file presumably failed.")
                    if nwarns > nmax_warn:
                        logger.debug("%{0}: More warnings triggered than allowed ({1:d}).".format(method, nmax_warn) +
                                     " Job will be trerminated and see error below.")
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
            #     logger.critical("%{0}: Error when writing TFRecord-file. Investigate error-message below."
            #     .format(method))
            #     raise err
            #
            # logger.info("%{0}: TFRecord-files have been created succesfully under '{1}'".format(method, tfr_data_dir))
            # logger.info("%{0}: During processing {1:d} warnings have been faced.".format(method, nwarns))
            #
            # logger.info("%{0}: TFRecord-files have been created succesfully under '{1}'".format(method, tfr_data_dir))
            # logger.info("%{0}: During processing {1:d} warnings have been faced.".format(method, nwarns))

        return nwarns

    @staticmethod
    def process_one_file(nc_file_in: str, grid_des_tar: dict, grid_des_coarse: dict, grid_des_base: dict):
        """
        Preprocess one netCDF-datafile.
        :param nc_file_in: input netCDF-file to be preprocessed.
        :param grid_des_tar: dictionary for grid description of target data
        :param grid_des_coarse: dictionary for grid description of coarse data
        :param grid_des_base: dictionary for grid description of auxiliary data
        """
        method = Preprocess_Unet_Tier1.process_one_file.__name__

        # sanity check
        if not os.path.isfile(nc_file_in): raise FileNotFoundError("%{0}: Could not find netCDF-file '{1}'."
                                                                   .format(method, nc_file_in))
        # hard-coded constants [IFS-specific parameters (from Chapter 12 in http://dx.doi.org/10.21957/efyk72kl)]
        cpd, g = 1004.709, 9.80665
        # get path to grid description files
        kf = "file"
        fgrid_des_base, fgrid_des_coarse, fgrid_des_tar = grid_des_base[kf], grid_des_coarse[kf], grid_des_tar[kf]
        # get parameters
        lon0_b, lon1_b = Preprocess_Unet_Tier1.get_slice_coords(grid_des_base["xfirst"], grid_des_base["xinc"],
                                                                grid_des_base["xsize"])
        lat0_b, lat1_b = Preprocess_Unet_Tier1.get_slice_coords(grid_des_base["yfirst"], grid_des_base["yinc"],
                                                                grid_des_base["ysize"])
        lon0_tar, lon1_tar = Preprocess_Unet_Tier1.get_slice_coords(grid_des_tar["xfirst"], grid_des_tar["xinc"],
                                                                    grid_des_tar["xsize"])
        lat0_tar, lat1_tar = Preprocess_Unet_Tier1.get_slice_coords(grid_des_tar["yfirst"], grid_des_tar["yinc"],
                                                                    grid_des_tar["ysize"])
        # initialize tools
        cdo, ncrename, ncap2, ncks, ncea = CDO(), NCRENAME(), NCAP2(), NCKS(), NCEA()

        fname_base = nc_file_in.rstrip(".nc")

        # start processing chain
        # slice data to region of interest and relevant lead times
        nc_file_sd = fname_base + "_subdomain.nc"
        ncea.run([nc_file_in, nc_file_sd],
                 OrderedDict([("-O", ""), ("-d", ["time,0,11", "latitude,{0},{1}".format(lat0_b, lat1_b),
                                                  "longitude,{0},{1}".format(lon0_b, lon1_b)])]))

        ncrename.run([nc_file_sd], OrderedDict([("-d", ["latitude,lat", "longitude,lon"]),
                                                ("-v", ["latitude,lat", "longitude,lon"])]))
        ncap2.run([nc_file_sd, nc_file_sd], OrderedDict([("-O", ""), ("-s", "\"lat=double(lat); lon=double(lon)\"")]))

        # calculate dry static energy fir first-order conservative remapping
        nc_file_dse = fname_base + "_dse.nc"
        ncap2.run([nc_file_sd, nc_file_dse], OrderedDict([("-O", ""), ("-s", "\"s={0}*t2m+z+{1}*2\"".format(cpd, g)),
                                                          ("-v", "")]))
        # add surface geopotential to file
        ncks.run([nc_file_sd, nc_file_dse], OrderedDict([("-A", ""), ("-v", "z")]))

        # remap the data (first-order conservative approach)
        nc_file_crs = fname_base + "_coarse.nc"
        cdo.run([nc_file_dse, nc_file_crs], OrderedDict([("remapcon", fgrid_des_coarse), ("-setgrid", fgrid_des_base)]))

        # remap with extrapolation on the target high-resolved grid with bilinear remapping
        nc_file_remapped = fname_base + "_remapped.nc"
        cdo.run([nc_file_crs, nc_file_remapped], OrderedDict([("remapbil", fgrid_des_tar),
                                                              ("-setgrid", fgrid_des_coarse)]))
        # retransform dry static energy to t2m
        ncap2.run([nc_file_remapped, nc_file_remapped], OrderedDict([("-O", ""),
                                                                     ("-s", "\"t2m_in=(s-z-{0}*2)/{1}\"".format(g, cpd)),
                                                                     ("-o", "")]))
        # finally rename data to distinguish between input and target data
        # (the later must be copied over from previous files)
        ncrename.run([nc_file_remapped], OrderedDict([("-v", "z,z_in")]))
        ncks.run([nc_file_remapped, nc_file_remapped], OrderedDict([("-O", ""), ("-x", ""), ("-v", "s")]))
        # NCEA-bug with NCO/4.9.5: Add slide offset to lon1_tar to avoid corrupted data in appended file
        # (does not affect slicing at all)
        lon1_tar = lon1_tar + np.float(grid_des_tar["xinc"])/10.
        ncea.run([nc_file_sd, nc_file_remapped], OrderedDict([("-A", ""),
                                                              ("-d", ["lat,{0},{1}".format(lat0_tar, lat1_tar),
                                                                      "lon,{0},{1}".format(lon0_tar, lon1_tar)]),
                                                               ("-v", "t2m,z")]))
        ncrename.run([nc_file_remapped], OrderedDict([("-v", ["t2m,t2m_tar", "z,z_tar"])]))

        if os.path.isfile(nc_file_remapped):
            print("%{0}: Processed data successfully from '{1}' to '{2}'. Cleaning-up..."
                  .format(method, nc_file_in, nc_file_remapped))
            for f in [nc_file_sd, nc_file_dse, nc_file_crs]:
                os.remove(f)
        else:
            raise RuntimeError("%{0}: Something went wrong when processing '{1}'. Check intermediate files."
                               .format(method, nc_file_in))

        return True

    @staticmethod
    def get_slice_coords(coord0, dx, n, d=4):
        """
        Small helper to get coords for slicing
        """
        coord0 = np.float(coord0)
        coords = (np.round(coord0, decimals=d), np.round(coord0 + (np.int(n) - 1) * np.float(dx), decimals=d))
        return np.amin(coords), np.amax(coords)



