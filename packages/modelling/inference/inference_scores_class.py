# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT
"""
Class for calculating scores for feature importance.
"""

__author__ = "Sebastian Lehner, Michael Langguth"
__email__ = "sebastian.lehner@geosphere.at, m.langguth@fz-juelich.de"
__date__ = "2025-10-03"
__update__ = "2025-10-03"

from typing import List
import logging
import numpy as np
import xarray as xr

# auxiliary variable for logger
logger_module_name = f"__main__.{__name__}"
module_logger = logging.getLogger(logger_module_name)

class InferenceScores:
    """
    Class to calculate scores and skill scores.
    """

    known_geodims = {"lon_dims": ["longitude", "lon", "rlon"],
                     "lat_dims": ["latitude", "lat", "rlat"]} 

    def __init__(self, data_fcst: xr.DataArray, data_ref: xr.DataArray, dims: List[str]):
        """
        :param data_fcst: forecast data to evaluate
        :param data_ref: reference or ground truth data
        """
        self.metrics_dict = {"mse": self.calc_mse, "rmse": self.calc_rmse,
            "rmse_relative": self.calc_rmse,
        }
        self.data_fcst = data_fcst
        self.data_dims = list(self.data_fcst.dims)
        self.data_ref = data_ref
        self.avg_dims = dims
        self.known_geodims = {"lat_dims": ["rlat", "lat", "latitude"], "lon_dims": ["rlon", "lon", "longitude"]}

    def __call__(self, score_name, **kwargs):
        try:
            score_func = self.metrics_dict[score_name]
        except:
            raise ValueError(f"{score_name} is not an implemented score." +
                             "Choose one of the following: {0}".format(", ".join(self.metrics_dict.keys())))

        return score_func(**kwargs)

    @property
    def data_fcst(self):
        return self._data_fcst

    @data_fcst.setter
    def data_fcst(self, da_fcst):
        if not isinstance(da_fcst, xr.DataArray):
            raise ValueError("data_fcst must be a xarray DataArray.")

        self._data_fcst = da_fcst

    @property
    def data_ref(self):
        return self._data_ref

    @data_ref.setter
    def data_ref(self, da_ref):
        if not isinstance(da_ref, xr.DataArray):
            raise ValueError("data_fcst must be a xarray DataArray.")

        if not list(da_ref.dims) == self.data_dims:
            raise ValueError("Dimensions of data_fcst and data_ref must match, but got:" +
                             "[{0}] vs. [{1}]".format(", ".join(list(da_ref.dims)),
                                                      ", ".join(self.data_dims)))

        self._data_ref = da_ref

    @property
    def avg_dims(self):
        return self._avg_dims

    @avg_dims.setter
    def avg_dims(self, dims):
        if dims is None:
            self.avg_dims = self.data_dims
            # print("Scores will be averaged across all data dimensions.")
        else:
            dim_stat = [avg_dim in self.data_dims for avg_dim in dims]
            if not all(dim_stat):
                ind_bad = [i for i, x in enumerate(dim_stat) if not x]
                raise ValueError("The following dimensions for score-averaging are not " +
                                 "part of the data: {0}".format(", ".join(np.array(dims)[ind_bad])))

            self._avg_dims = dims

    def calc_mse(self, relative: bool = False, **kwargs):
        """
        Calculate mse of forecast data w.r.t. reference data
        :param relative: when True, calculates the relative MSE, otherwise absolute
        :return: MSE
        """
        # get local logger
        func_logger = logging.getLogger(
            f"{logger_module_name}.Scores.{self.calc_mse.__name__}"
        )

        if kwargs:
            func_logger.debug(
                "Passed keyword arguments to calc_mse are without effect."
            )

        # calculate mse
        mse = np.square(self.data_fcst - self.data_ref).mean(dim=self.avg_dims)
        
        if relative:
            func_logger.info(f"kwarg {relative = } => calculating relative mse.")
            mse = mse / np.square(self.data_ref.mean(dim=self.avg_dims))
        else:
            func_logger.info(f"kwarg {relative = } => calculating absolute mse.")

        return mse

    def calc_rmse(self, relative: bool = False, **kwargs):
        """
        Calculate rmse of forecast data w.r.t. reference data
        :param relative: when True, calculates the relative RMSE, otherwise absolute
        :return: RMSE
        """
        # get local logger
        func_logger = logging.getLogger(
            f"{logger_module_name}.Scores.{self.calc_rmse.__name__}"
        )

        if kwargs:
            func_logger.debug(
                "Passed keyword arguments to calc_rmse are without effect."
            )

        if not relative:
            func_logger.info(f"kwarg {relative = } => calculating absolute rmse.")
            rmse = np.sqrt(self.calc_mse())
        else:
            func_logger.info(f"kwarg {relative = } => calculating relative rmse.")
            rmse = np.sqrt(self.calc_mse(relative=relative))

        return rmse

    def check_for_coords(
        self, coord_names_data, dim_query: str, return_index: bool = False):
        """
        Check if one of the known geographical coordinates is part of the passed list.
        :param coord_names_data: list of coordinate names
        :param dim_query: dimension to be checked for (either 'lat' or 'lon')
        :param return_index: flag to return the index of the found coordinate instead of name
        :return: index of the first found coordinate and the name of the coordinate
        """
        assert dim_query in ["lat", "lon"], "dim_query must be either 'lat' or 'lon'."

        dim_key = dim_query + "_dims"
        known_geodims = self.known_geodims[dim_key]

        stat = False
        for i, coord in enumerate(known_geodims):
            if coord in coord_names_data:
                stat = True
                break

        if stat:
            return_val = i if return_index else known_geodims[i]
            return return_val
        else:
            raise ValueError("Could not find one of the following coordinates in the passed dictionary: {0}"
                                .format(",".join(known_geodims)))
    @staticmethod
    def get_cdf_of_x(sample_in, prob_in):
        """
        Wrappper for interpolating CDF-value for given data
        :param sample_in : input values to derive discrete CDF
        :param prob_in   : corresponding CDF
        :return: lambda function converting arbitrary input values to corresponding CDF value
        """
        return lambda xin: np.interp(xin, sample_in, prob_in)
    

