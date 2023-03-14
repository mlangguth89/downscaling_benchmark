# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Simplified script to perform inference on a trained U-Net
"""

import os
from abc import ABC, abstractmethod
from typing import Union, List
import argparse
import json as js
from timeit import default_timer as timer
import dask
import numpy as np
import xarray as xr
import tensorflow.keras as keras

# Auxilary methods


def reshape_ds(ds):
    """
    Convert a xarray dataset to a data-array where the variables will constitute the last dimension (channel last)
    :param ds: the xarray dataset with dimensions (dims)
    :return da: the data-array with dimensions (dims, variables)
    """
    da = ds.to_array(dim="variables")
    da = da.transpose(..., "variables")
    return da


def split_in_tar(da: xr.DataArray, target_var: str = "t2m") -> (xr.DataArray, xr.DataArray):
    """
    Split data array with variables-dimension into input and target data for downscaling
    :param da: The unsplitted data array
    :param target_var: Name of target variable which should consttute the first channel
    :return: The split data array.
    """
    invars = [var for var in da["variables"].values if var.endswith("_in")]
    tarvars = [var for var in da["variables"].values if var.endswith("_tar")]

    # ensure that ds_tar has a channel coordinate even in case of single target variable
    roll = False
    if len(tarvars) == 1:
        sl_tarvars = tarvars
    else:
        sl_tarvars = slice(*tarvars)
        if tarvars[0] != target_var:     # ensure that target variable appears as first channel
            roll = True

    da_in, da_tar = da.sel({"variables": invars}), da.sel(variables=sl_tarvars)
    if roll: da_tar = da_tar.roll(variables=1, roll_coords=True)

    return da_in, da_tar


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
        Normalize data.
        :param data: The DataArray to be normalized.
        :param **stats: Parameters to perform normalization. Must fit to normalization type!
        :return: DataArray with normalized data.
        """
        # sanity checks
        if not isinstance(data, xr.DataArray):
            raise TypeError(f"Passed data must be a xarray.DataArray, but is of type {str(type(data))}.")

        _ = self._check_norm_dims(data)
        # do the computation
        norm_stats = self.get_required_stats(data, **stats)
        data_norm = self.normalize_data(data, *norm_stats)

        return data_norm

    def denormalize(self, data: da_or_ds, **stats):
        """
        Denormalize data.
        :param data: The DataArray to be denormalized.
        :param **stats: Parameters to perform denormalization. Must fit to normalization type!
        :return: DataArray with denormalized data.
        """
        # sanity checks
        if not isinstance(data, xr.DataArray):
            raise TypeError(f"Passed data must be a xarray.DataArray, but is of type {str(type(data))}.")

        _ = self._check_norm_dims(data)
        # do the computation
        norm_stats = self.get_required_stats(data, **stats)
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

    def save_norm_to_file(self, js_file, missdir_ok: bool = True):
        """
        Write normalization parameters to file.
        :param js_file: Path to JSON-file to be created.
        :param missdir_ok: If True, base-directory of JSON-file can be missing and will be created then.
        :return: -
        """
        if self.norm_stats is None:
            raise AttributeError("norm_stats is still None. Please run (de-)normalization to get parameters.")

        if any([stat is None for stat in self.norm_stats.values()]):
            raise AttributeError("Some parameters of norm_stats are None.")

        norm_serialized = {key: da.to_dict() for key, da in self.norm_stats.items()}

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

        norm_dict_restored = {key: xr.DataArray.from_dict(da_dict) for key, da_dict in norm_data.items()}

        self.norm_stats = norm_dict_restored

    @abstractmethod
    def get_required_stats(self, data, *stats):
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


class ZScore(Normalize):
    def __init__(self, norm_dims: List):
        super().__init__("z_score", norm_dims)
        self.norm_stats = {"mu": None, "sigma": None}

    def get_required_stats(self, data: da_or_ds, **stats):
        """
        Get required parameters for z-score normalization. They are either computed from the data
        or can be parsed as keyword arguments
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

# main-script


def main(parser_args):

    # Load trained model
    print(f"Load model from {parser_args.model_dir}")
    t0_load = timer()
    trained_model = keras.models.load_model(parser_args.model_dir, compile=False)
    print(f"Model was loaded successfully. Model loading time: {timer() - t0_load:.3f}s")

    # Load JSON-file providing normalization data
    js_norm = os.path.join(parser_args.model_dir, "norm.json")
    print("Read normalization file for subsequent data transformation.")
    data_norm = ZScore(["time", "rlat", "rlon"])
    data_norm.read_norm_from_file(js_norm)

    # Read the test dataset and pre-process it
    fdata_test = os.path.join(parser_args.data_dir, "downscaling_tier2_test.nc")
    print(f"Start opening datafile {fdata_test}...")
    t0_read = timer()

    ds_test = xr.open_dataset(fdata_test)
    da_test = data_norm.normalize(reshape_ds(ds_test))
    del ds_test

    da_test_in, da_test_tar = split_in_tar(da_test)
    tar_varname = da_test_tar['variables'].values[0]

    # add high-resolved surface topography as input
    da_test_in = xr.concat([da_test_in, da_test_tar.sel({"variables": "hsurf_tar"})], dim="variables")
    da_test_in = da_test_in.squeeze().values
    print(np.shape(da_test_in))

    # start inference
    print(f"Loading and preparation of test dataset finished. Data preparation time: {timer() - t0_read:.2f}s")
    t0_train = timer()
    print("Start inference on trained model...")
    _ = trained_model.predict(da_test_in, batch_size=32, verbose=2)
    tend = timer()
    print(f"Inference finished. Inference time: {tend - t0_train:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", "-data_dir", dest="data_dir", type=str, required=True,
                        help="Directory where test dataset (netCDF-file) is stored.")
    parser.add_argument("--model_directory", "-model_dir", dest="model_dir", type=str, required=True,
                        help="Base directory where trained models are saved.")

    args = parser.parse_args()
    main(args)

