# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

"""
General normalizer class which encapsulates all normalization based on abstract Normalize class
"""

__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2022-10-06"
__update__ = "2025-02-17"

import os
from typing import List, Union
import json as js
from abstract_data_normalization import Normalize
import dask
import xarray as xr

da_or_ds = Union[xr.DataArray, xr.Dataset]

#
### General normalization class encapsulating all normalizers
#
class GeneralNormalizer:
    def __init__(self, normalization_config: dict, norm_dims: list, **kwargs):
        """
        Initialize the GeneralNormalizer with a normalization configuration.
        :param normalization_config: Dictionary mapping variable names to normalization methods.
        :param norm_dims: List of dimensions to apply normalization over.
        :param kwargs: optional keyword arguments for respective normalizer
        """
        self.normalization_config = normalization_config
        self.norm_dims = norm_dims
        self.normalizers, self.vars_normalizers = self._initialize_normalizers(**kwargs)

    def _initialize_normalizers(self, **kwargs):
        """
        Create instances of normalization classes based on the config.
        :param kwargs: optional keyword arguments for respective normalizer
        :return: Dictionary of normalizers and dictionary of variables for each normalizer
        """
        method_groups = {}
        vars_in_groups = {}
        
        for var, method in self.normalization_config.items():
            if method not in method_groups:
                if method == "zscore":
                    method_groups[method] = ZScore(self.norm_dims, **kwargs)
                elif method == "log_zscore":
                    method_groups[method] = Log_ZScore(self.norm_dims, **kwargs)
                else:
                    raise ValueError(f"Unknown normalization method: {method}")
 
            if method not in vars_in_groups:
                vars_in_groups[method] = []
            
            vars_in_groups[method].append(var)

        return method_groups, vars_in_groups

    def get_stats_from_data(self, data):
        norm_stats = {}
        
        for method, normalizer in self.normalizers.items():
            norm_stats.update(normalizer.get_required_stats(data[self.vars_normalizers[method]]))

        return norm_stats
            

    def normalize(self, data: xr.Dataset):
        """
        Normalize the dataset based on the specified normalization methods.
        Note that normalizing is an in-place operation.
        :param data: xarray Dataset to normalize.
        :return: Normalized xarray Dataset.
        """
        for method, normalizer in self.normalizers.items():
            vars_to_normalize = [var for var, norm in self.normalization_config.items() if norm == method]
            print(f"Variables to normalize with method {method}: [{', '.join(vars_to_normalize)}]")
            data.update(normalizer.normalize(data[vars_to_normalize]))
        return data

    def denormalize(self, data: xr.Dataset):
        """
        Denormalize the dataset back to its original values.
        Note that denormalizing is an in-place operation.
        :param data: xarray Dataset to denormalize.
        :return: Denormalized xarray Dataset.
        """
        for method, normalizer in self.normalizers.items():
            vars_to_denormalize = [var for var, norm in self.normalization_config.items() if norm == method]
            data.update(normalizer.denormalize(data[vars_to_denormalize]))
        return data

    def read_norms_from_file(self, js_file):
        """
        Read normalization parameters from file. Inverse function to write_norms_from_file.
        :param js_file: Path to JSON-file to be read.
        :return: Parameters set to norm_stats of normalizers
        """
        with open(js_file, "r") as jsf:
            norm_data = js.load(jsf)

        data_type = norm_data.pop('data_type', None)

        if data_type == "data_array":
            xr_obj = xr.DataArray
        elif data_type == "data_set":
            xr_obj = xr.Dataset
        else:
            raise ValueError(
                f"Unknown data_type {data_type} in {js_file}. Only 'data_array' or 'data_set' are allowed.")

        norm_data.pop('data_type', None)

        for normalizer in self.normalizers.keys():
            # read required stats only
            norm_stats = self.normalizers[normalizer].norm_stats
            req_stats = list(norm_stats.keys())
            norm_dict_restored = {key: xr_obj.from_dict(da_dict) for key, da_dict in norm_data.items() if key in req_stats}
            # set required stats
            self.normalizers[normalizer].norm_stats = norm_dict_restored

    def save_norms_to_file(self, js_file, missdir_ok: bool = True):
        """
        Write normalization parameters to file
        :param js_file: Path to JSON-file to be created
        :param missdir_ok: If True, base-directory of JSON-file can be missing and will be created then
        :return: -
        """
        norm_serialized ={}
        data_type = None
        for normalizer in self.normalizers.keys():
            norm_stats = self.normalizers[normalizer].norm_stats
            if norm_stats is None:
                raise AttributeError(f"norm_stats for normalizer {normalizer} is still None. Please run (de-)normalization to get parameters.")

            if any([stat is None for stat in norm_stats.values()]):
                raise AttributeError(f"Some parameters of norm_stats for normalizer {normalizer} are None.")

            norm_serialized.update({key: da.to_dict() for key, da in norm_stats.items()})

            # check data type for consistency
            d0 = list(norm_stats.values())[0]
            if isinstance(d0, xr.DataArray):
                data_type_now = "data_array"
            elif isinstance(d0, xr.Dataset):
                data_type_now = "data_set"

            if data_type:
                if data_type_now != data_type:
                    raise ValueError(f"Inconsistent data type from {normalizer}. Expected {data_type}, but got {data_type_now}")
            else:
                data_type = data_type_now
            
        # serialization and (later) deserialization depends on data type.
        # Thus, we have to save it to the dictionary
        norm_serialized["data_type"] = data_type

        if missdir_ok: os.makedirs(os.path.dirname(js_file), exist_ok=True)

        # write to JSON-file
        with open(js_file, "w") as jsf:
            js.dump(norm_serialized, jsf)

#
### Normalizers
#
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


class Log_ZScore(Normalize):
    """
    Class to perform zscore-normalization on log transformed data.
    """
    def __init__(self, norm_dims: List, eps=0.01):
        super().__init__("log_zscore", norm_dims)
        self.norm_stats = {"log_mu": None, "log_sigma": None}
        self.eps = eps

    def get_required_stats(self, data: da_or_ds, varname: str= None, **stats):
        """
        Get required parameters for z-score normalization. They are either computed from the data
        or can be parsed as keyword arguments.
        :param data: the data to be (de-)normalized
        :param varname: retrieve parameters for specific varname only (without effect if parameters must be retrieved from data)
        :param stats: keyword arguments for mean (mu) and standard deviation (std) used for normalization
        :return (mu, sigma): Parameters for normalization
        """
        log_mu, log_std = stats.get("log_mu", self.norm_stats["log_mu"]), stats.get("log_sigma", self.norm_stats["log_sigma"])

        if log_mu is None or log_std is None:
            print("Retrieve mu and sigma from data...")
            log_data = np.log(data + self.eps) - np.log(self.eps)
            log_mu, log_std = log_data.mean(self.norm_dims), log_data.std(self.norm_dims)
            # the following ensure that both parameters are computed in one graph!
            # This significantly reduces memory footprint as we don't end up having data duplicates
            # in memory due to multiple graphs (and also seem to enfore usage of data chunks as well)
            log_mu, log_std = dask.compute(log_mu, log_std)
            self.norm_stats = {"log_mu": log_mu, "log_sigma": log_std}
        else:
            if varname:
                if isinstance(log_mu, xr.DataArray):
                    log_mu, log_std = log_mu.sel({"variables": varname}), log_std.sel({"variables": varname})
                elif isinstance(log_mu, xr.Dataset):
                    log_mu, log_std = log_mu[varname], log_std[varname]
                else:
                    raise ValueError(f"Unexpected data type for mu and std: {type(log_mu)}, {type(log_std)}")
        #    print("Mu and sigma are parsed for (de-)normalization.")

        return log_mu, log_std

    def normalize_data(self, data, log_mu, log_std):
        """
        Perform z-score normalization on data
        :param data: Data array of interest
        :param mu: mean of data for normalization
        :param std: standard deviation of data for normalization
        :return data_norm: normalized data
        """
        data = np.log(data + self.eps) - np.log(self.eps)
        data = (data - log_mu) / log_std

        return data

    def denormalize_data(self, data, log_mu, log_std):
        """
        Perform z-score denormalization on data.
        :param data: Data array of interest
        :param mu: mean of data for denormalization
        :param std: standard deviation of data for denormalization
        :return data_norm: denormalized data
        """
        data = np.exp( data + np.log(self.eps)) - self.eps
        data = data * log_std + log_mu

        return data
