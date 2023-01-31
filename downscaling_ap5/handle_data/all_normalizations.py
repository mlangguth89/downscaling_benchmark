# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
All implemented classes to perform normalization on data
"""

__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2022-10-06"

from typing import List
from abstract_data_normalization import Normalize
import dask
import xarray as xr


class ZScore(Normalize):
    def __init__(self, norm_dims: List):
        super().__init__("z_score", norm_dims)
        self.norm_stats = {"mu": None, "sigma": None}

    def get_required_stats(self, data: xr.DataArray, **stats):
        """
        Get required parameters for z-score normalization. They are either computed from the data
        or can be parsed as keyword arguments.
        :param data: the data to be (de-)normalized
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
        # else:
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
        data_norm = (data - mu) / std

        return data_norm

    @staticmethod
    def denormalize_data(data, mu, std):
        """
        Perform z-score denormalization on data.
        :param data: Data array of interest
        :param mu: mean of data for denormalization
        :param std: standard deviation of data for denormalization
        :return data_norm: denormalized data
        """
        data_denorm = data * std + mu

        return data_denorm
