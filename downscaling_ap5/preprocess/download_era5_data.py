# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Script to download ERA5 data from the CDS API.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2023-11-21"
__update__ = "2023-08-22"

# import modules
import os, sys
import logging
import cdsapi
import numpy as np
import pandas as pd
from multiprocessing import Pool
from utils import to_list

# get logger
logger_module_name = f"main_download_era5.{__name__}"
module_logger = logging.getLogger(logger_module_name)

# known request parameters
knwon_req_keys = ["ml", "sfc"]


class ERA5_Data_Loader(object):
    """
    Class to download ERA5 data from the CDS API.
    """

    knwon_req_keys = ["ml", "sfc"]
    allowed_formats = ["netcdf", "grib"]
    # defaults 
    area = [75, -45, 20, 65]
    month_start = 1
    month_end = 12


    def __init__(self, nworkers) -> None:

        # get local logger
        func_logger = logging.getLogger(f"{logger_module_name}.{self.__init__.__name__}")
        
        self.nworkers = nworkers
        try: 
            self.cds = [cdsapi.Client() for _ in range(self.nworkers)]
        except Exception as e:
            func_logger.error(f"Could not initialize CDS API client: {e} \n" + \
                              "Please follow the instructions at https://cds.climate.copernicus.eu/api-how-to to install the CDS API.")                                                 ")
            raise e
        
    def __call__(self, req_dict, data_dir, start, end, format, **kwargs):
        """
        Run the requests to download the ERA5 data.
        :param req_dict: dictionary with data requests
        :param data_dir: directory where output data files will be stored
        :param start: start year of data request
        :param end: end year of data request
        :param format: format of downloaded data (netcdf or grib)
        :param kwargs: additional keyword arguments (options: area, month_start, month_end)
        """

        # get local logger
        func_logger = logging.getLogger(f"{logger_module_name}.{self.__call__.__name__}")

        # create output directory
        if not os.path.exists(data_dir):
            func_logger.info(f"Creating output directory {data_dir}")
            os.makedirs(data_dir)

        # validate request keys
        req_list = self.validate_request_keys(req_dict)

        # select data format
        if format not in self.allowed_formats:
            func_logger.warning(f"Unknown data format {format}. Using default format netcdf.")
            format = "netcdf"

        for req_key in req_list:
            if req_key == "sfc":
                out = self.download_sfc_data(req_dict[req_key], data_dir, start, end, format, **kwargs)
            elif req_key == "ml":
                out = self.download_ml_data(req_dict[req_key], data_dir, start, end, format, **kwargs)

            # check if all requests were successful
            _ = self.check_request_result(out)

        return out
    

    def download_sfc_data(self, varlist, data_dir, start, end, format, **kwargs):
        """
        Download ERA5 surface data.
        To obtain the varlist, please refer to the CDS API documentation at https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
        :param varlist: list of variables to download
        :param data_dir: directory where output data files will be stored
        :param start: start year of data request
        :param end: end year of data request
        :param format: format of downloaded data (netcdf or grib)
        :param kwargs: additional keyword arguments (options: area, month_start, month_end)
        :return: output of multiprocessing pool
        """
        # get local logger
        func_logger = logging.getLogger(f"{logger_module_name}.{self.download_sfc_data.__name__}")

        # get additional keyword arguments
        area = kwargs.get("area", self.area)
        month_start = kwargs.get("month_start", self.month_start)
        month_end = kwargs.get("month_end", self.month_end)

        # create base request dictionary (None-values will be set dynamically)
        req_dict_base = {"product_type": "reanalysis", "format": f"{format}", 
                         "variable": to_list(varlist), 
                         "day": None, "month": None, "time": [f"{h:02d}" for h in range(24)], "year": None,
                        "area": area}   

        # initialize multiprocessing pool
        func_logger.info(f"Downloading ERA5 surface data for variables {', '.join(varlist)} with {self.nworkers} workers.")
        pool = Pool(self.nworkers)

        # initialize dictionary for request results
        req_results = {}

        # create data requests for each month
        for year in range(start, end+1):
            req_dict = req_dict_base.copy()
            req_dict["year"] = [f"{year}"]
            for month in range(month_start, month_end+1):
                req_dict["month"] = [f"{month:02d}"]
                # get last day of month
                last_day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)
                req_dict["day"] = [f"{d:02d}" for d in range(1, last_day.day+1)]
                fout = f"era5_sfc_{year}-{month:02d}.{format}"
                
                func_logger.debug(f"Downloading ERA5 surface data for {year}-{month:02d} to {os.path.join(data_dir, fout)}")

                req_results[fout] = pool.apply_async(self.cds.retrieve, args=("reanalysis-era5-single-levels", req_dict,
                                                     os.path.join(data_dir, fout)))

        # run and close multiprocessing pool
        pool.close()
        pool.join()

        func_logger.info(f"Finished downloading ERA5 surface data.")

        return req_results

    def download_ml_data(self, var_dict, data_dir, start, end, format, **kwargs):
        """
        Download ERA5 data for multiple levels.
        To obtain the varlist, please refer to the CDS API documentation at https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-complete?tab=form
        :param var_dict: dictionary of variables to download for each level
        :param data_dir: directory where output data files will be stored
        :param start: start year of data request
        :param end: end year of data request
        :param format: format of downloaded data (netcdf or grib)
        :param kwargs: additional keyword arguments (options: area, month_start, month_end)
        :return: output of multiprocessing pool
        """
        # get local logger
        func_logger = logging.getLogger(f"{logger_module_name}.{self.download_ml_data.__name__}")

        # get additional keyword arguments
        area = kwargs.get("area", self.area)
        month_start = kwargs.get("month_start", self.month_start)
        month_end = kwargs.get("month_end", self.month_end)

        vars = var_dict.keys()
        vars_param = self.translate_mars_vars(vars)
        # ensure that data is downloaded for all levels (All-together approach -> overhead: additional download of data for levels that are not needed)
        collector = []
        _ = [collector.extend(ml_list) for ml_list in var_dict.values()]
        all_lvls = sorted([str(lvl) for lvl in set(collector)])

        # create base request dictionary (None-values will be set dynamically)
        req_dict_base = {"class": "ea", "date": None,
                         "expver": "1",
                         "levelist": "/".join(all_lvls),
                         "levtype": "ml",
                         "param": vars_param,
                         "stream": "oper",
                         "time": "00/to/23/by/1",
                         "type": "an", 
                         "area": area ,
                         "grid": "0.25/0.25",}

        # initialize multiprocessing pool
        func_logger.info(f"Downloading ERA5 model-level data for variables {', '.join(vars)} with {self.nworkers} workers.")
        pool = Pool(self.nworkers)
        
        # initialize dictionary for request results
        req_results = {}

        # create data requests for each month
        for year in range(start, end+1):
            req_dict = req_dict_base.copy()
            for month in range(month_start, month_end+1):
                # get last day of month
                last_day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)
                req_dict["date"] = f"{year}/{month:02d}/01/to/{year}/{month:02d}/{last_day.day}" 
                fout = f"era5_ml_{year}-{month:02d}.{format}"

                func_logger.debug(f"Downloading ERA5 model-level data for {year}-{month:02d} to {os.path.join(data_dir, fout)}")

                req_results[fout] = pool.apply_async(self.cds.retrieve, args=("reanalysis-era5-complete", req_dict,
                                                     os.path.join(data_dir, fout)))
            
        # run and close multiprocessing pool
        pool.close()
        pool.join()

        func_logger.info(f"Finished downloading ERA5 model-level data.")

        return req_results
    
    def check_request_result(self, results_dict):
        """
        Check if request was successful.
        :param results_dict: dictionary with request results (returned by download_ml_data and download_sfc_data)
        """
        # get local logger
        func_logger = logging.getLogger(f"{logger_module_name}.{self.check_request_result.__name__}")

        # check if all requests were successful
        stat = [o.get().check().__dict__["reply"]["state"] == "completed" for o in results_dict.values()]

        ok = True

        if all(stat):
            func_logger.info(f"All requests were successful.")
        else:
            ok = False
            bad_req = np.where(np.array(stat) == False)[0]
            results_arr = np.array(list(results_dict.keys()))
            func_logger.error(f"The following requests were not successful: {', '.join(results_arr[bad_req])}.")

        return ok


    def validate_request_keys(self, req_dict):
        """
        Validate request keys in data request file.
        :param req_dict: dictionary with data requests
        :return: list of valid request keys (filtered)
        """
        # get local logger
        func_logger = logging.getLogger(f"{logger_module_name}.{self.validate_request_keys.__name__}")

        # create list of requests
        req_list = []
        for req_key in req_dict.keys():
            if req_key in self.knwon_req_keys:
                req_list.append(req_key)
            else:
                func_logger.warning(f"Unknown request key {req_key} in data request file. Skipping this request.")

        return req_list

    def translate_mars_vars(self, vars):
        """
        Translate variable names to MARS parameter names.
        :param vars: list of variable names
        :return: list of MARS parameter names
        """

        # create dictionary with variable names and corresponding MARS parameter names
        var_dict = {"z": "129", "t": "130", "u": "131", "v": "132", "w": "135", "q": "133", 
                    "temperature": "130", "specific_humidity": "133", "geopotential": "129",
                    "vertical_velocity": "135", "u_component_of_wind": "131", "v_component_of_wind": "132",
                    "vorticity": "138", "divergence": "139", "logarithm_of_surface_pressure": "152", 
                    "fraction_of_cloud_cover": "164", "specific_cloud_liquid_water_content": "246",
                    "specific_cloud_ice_water_content": "247", "specific_rain_water_content": "248",
                    "specific_snow_water_content": "249", }

        # translate variable names
        vars_param = [var_dict[var] for var in vars]

        return vars_param





