# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
All implemented classes to perform normalization on data
"""

__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2022-10-06"
__update__ = "2023-01-31"

from typing import List, Union
from abstract_data_normalization import Normalize
import dask
import xarray as xr

da_or_ds = Union[xr.DataArray, xr.Dataset]


class ZScore(Normalize):
    def __init__(self, norm_dims: List):
        super().__init__("z_score", norm_dims)
        self.norm_stats = {"mu": None, "sigma": None}

    def get_required_stats(self, data: da_or_ds, varname: str= None, **stats):
        """
        Get required parameters for z-score normalization. They are either computed from the data
        or can be parsed as keyword arguments.
        :param data: the data to be (de-)normalized
        :param varname: retrieve parameters for specific varname only (without effect if parameters must be retrieved from data)
        :param stats: keyword arguments for mean (mu) and standard deviation (std) used for normalization
        :return (mu, sigma): Parameters for normalization
        """
        mu, std = stats.get("mu", self.norm_stats["mu"]), stats.get("sigma", self.norm_stats["sigma"])

        if mu is None or std is None:
            print("Retrieve mu and sigma from data...")
            mu, std = data.mean(self.norm_dims), data.std(self.norm_dims)
            # the following ensure that both parameters are computed in one graph!
            # This significantly reduces memory footprint as we don't end up having data duplicates
            # in memory due to multiple graphs (and also seem to enfore usage of data chunks as well)
            mu, std = dask.compute(mu, std)
            self.norm_stats = {"mu": mu, "sigma": std}
        else:
            if varname:
                if isinstance(mu, xr.DataArray):
                    mu, std = mu.sel({"variables": varname}), std.sel({"variables": varname})
                elif isinstance(mu, xr.Dataset):
                    mu, std = mu[varname], std[varname]
                else:
                    raise ValueError(f"Unexpected data type for mu and std: {type(mu)}, {type(std)}")
        #    print("Mu and sigma are parsed for (de-)normalization.")

        return mu, std

    @staticmethod
    def normalize_data(data, mu, std):
        """
        Perform z-score normalization on data
        :param data: Data array of interest
        :param mu: mean of data for normalization
        :param std: standard deviation of data for normalization
        :return data_norm: normalized data
        """
        data = (data - mu) / std

        return data

    @staticmethod
    def denormalize_data(data, mu, std):
        """
        Perform z-score denormalization on data.
        :param data: Data array of interest
        :param mu: mean of data for denormalization
        :param std: standard deviation of data for denormalization
        :return data_norm: denormalized data
        """
        data = data * std + mu

        return data
