# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Abstract class to perform normalization on data
"""

__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"
__update__ = "2023-01-31"

from abc import ABC, abstractmethod
from typing import Union, List
import os
import json as js
import numpy as np
import xarray as xr

da_or_ds = Union[xr.DataArray, xr.Dataset]


class Normalize(ABC):
    """
    Abstract class for normalizing data.
    """

    def __init__(self, method: str, norm_dims: List):
        self.method = method
        self.norm_dims = norm_dims
        self.norm_stats = None

    def normalize(self, data: xr.DataArray, **stats):
        """
        Normalize data
        :param data: The DataArray to be normalized
        :param stats: Optional parameters to perform normalization. Must fit to normalization type!
        :return: DataArray with normalized data
        """
        # sanity checks
        # if not isinstance(data, xr.DataArray):
        #    raise TypeError(f"Passed data must be a xarray.DataArray, but is of type {str(type(data))}.")

        _ = self._check_norm_dims(data)
        # do the computation
        norm_stats = self.get_required_stats(data, **stats)
        norm_stats = Normalize.match_datatype(data, *norm_stats)
        data_norm = self.normalize_data(data, *norm_stats)

        return data_norm

    def denormalize(self, data: da_or_ds, **stats):
        """
        Denormalize data.
        :param data: The DataArray to be denormalized.
        :param stats: Optional parameters to perform denormalization. Must fit to normalization type!
        :return: DataArray with denormalized data.
        """
        # sanity checks
        # if not isinstance(data, xr.DataArray):
        #    raise TypeError(f"Passed data must be a xarray.DataArray, but is of type {str(type(data))}.")

        _ = self._check_norm_dims(data)
        # do the computation
        norm_stats = self.get_required_stats(data, **stats)
        norm_stats = Normalize.match_datatype(data, *norm_stats)
        data_denorm = self.denormalize_data(data, *norm_stats)

        return data_denorm

    @property
    def norm_dims(self):
        return self._norm_dims

    @norm_dims.setter
    def norm_dims(self, norm_dims):
        if norm_dims is None:
            raise AttributeError("norm_dims must not be None. Please parse a list of dimensions" +
                                 "over which normalization should be applied.")

        self._norm_dims = list(norm_dims)

    def _check_norm_dims(self, data):
        """
        Check if dimension for normalization reside in dimensions of data.
        :param data: the data (xr.DataArray) to be normalized
        :return True: in case of passed check, a ValueError is risen else
        """
        data_dims = list(data.dims)
        norm_dims_check = [norm_dim in data_dims for norm_dim in self.norm_dims]
        if not all(norm_dims_check):
            imiss = np.where(~np.array(norm_dims_check))[0]
            miss_dims = list(np.array(self.norm_dims)[imiss])
            raise ValueError("The following dimensions do not reside in the data: " +
                             f"{', '.join(miss_dims)}")

        return True

    @staticmethod
    def match_datatype(data, *args, var_dim="variables"):
        """
        Ensures that the arguments have the same xarray datatype (either xr.DataArray or xr.Dataset) as data,
        i.e. coerces all arguments against type(data) if necessary.
        :param data: the reference data (must be either xr.Dataset or xr.DataArray)
        :param args: arbitrary number of arguments (all of them must also be either xr.Dataset or xr.DataArray,
                     but should not be mixed, e.g. type(args[0])=xr.Dataset and type(args[1])=xr.DataArray is not
                     allowed
        :param var_dim: dimension name to convert from/to xr.Dataset/xr.DataArray
        """

        # sanity check
        ds_or_da = (xr.Dataset, xr.DataArray)
        all_args = [data] + list(args)
        if not all(isinstance(arg, ds_or_da) for arg in all_args):
            flags = [not isinstance(arg, ds_or_da) for arg in all_args]
            inds = np.nonzero(flags)[0].tolist()
            if len(inds) == 1:
                err_str = f"The parsed argument at position {inds} is"
            else:
                err_str = f"The parsed arguments at positions {inds} are"
            raise ValueError(f"{err_str} not an xarray.DataArray or xarray.Dataset.")

        # align type of arguments if required
        if isinstance(data, type(args[0])):
            args_new = args
        elif isinstance(data, xr.Dataset) and isinstance(args[0], xr.DataArray):
            args_new = tuple(arg.to_dataset(dim=var_dim) for arg in args)
        elif isinstance(data, xr.DataArray) and isinstance(args[0], xr.Dataset):
            args_new = tuple(arg.to_array(dim=var_dim) for arg in args)
        else:
            raise ValueError("Unknown error occured. Please check all input parameters.")

        return args_new

    def save_norm_to_file(self, js_file, missdir_ok: bool = True):
        """
        Write normalization parameters to file
        :param js_file: Path to JSON-file to be created
        :param missdir_ok: If True, base-directory of JSON-file can be missing and will be created then
        :return: -
        """
        if self.norm_stats is None:
            raise AttributeError("norm_stats is still None. Please run (de-)normalization to get parameters.")

        if any([stat is None for stat in self.norm_stats.values()]):
            raise AttributeError("Some parameters of norm_stats are None.")

        norm_serialized = {key: da.to_dict() for key, da in self.norm_stats.items()}

        # serialization and (later) deserialization depends on data type.
        # Thus, we have to save it to the dictionary
        d0 = list(self.norm_stats.values())[0]
        if isinstance(d0, xr.DataArray):
            norm_serialized["data_type"] = "data_array"
        elif isinstance(d0, xr.Dataset):
            norm_serialized["data_type"] = "data_set"

        if missdir_ok: os.makedirs(os.path.dirname(js_file), exist_ok=True)

        with open(js_file, "w") as jsf:
            js.dump(norm_serialized, jsf)

    def read_norm_from_file(self, js_file):
        """
        Read normalization parameters from file. Inverse function to write_norm_from_file.
        :param js_file: Path to JSON-file to be read.
        :return: Parameters set to self.norm_stats
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

        norm_dict_restored = {key: xr_obj.from_dict(da_dict) for key, da_dict in norm_data.items()}

        self.norm_stats = norm_dict_restored

    @abstractmethod
    def get_required_stats(self, data, varname, *stats):
        """
        Function to retrieve either normalization parameters from data or from keyword arguments
        """
        pass

    @staticmethod
    @abstractmethod
    def normalize_data(data, *norm_param):
        """
        Function to normalize data.
        """
        pass

    @staticmethod
    @abstractmethod
    def denormalize_data(data, *norm_param):
        """
        Function to denormalize data.
        """
        pass

