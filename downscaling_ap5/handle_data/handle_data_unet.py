__author__ =  "Michael Langguth"
__date__ = "2022-01-20"
__update__ = "2022-02-01"

from typing import Union, List
from timeit import default_timer as timer
import datetime as dt
import numpy as np
import xarray as xr
import climetlab as cml
from handle_data_class import HandleDataClass
import os
import json
# basic data types
arr_xr_np = Union[xr.Dataset, xr.Dataset, np.ndarray]


class HandleUnetData(HandleDataClass):

    def __init__(self, datadir: str, query: str, purpose: str = None, hour: int = 12) -> None:
        app = "maelstrom-downscaling"
        super().__init__(datadir, app, query, purpose, hour=12)

        self.status_ok = True
        self.hour = hour
        self.sample_dim = "time"
        self.data_info["nsamples"] = {key: ds.dims[self.sample_dim] for (key, ds) in self.data.items()}

    def append_data(self, query: str, purpose: str = None):
        """
        Appends data-dictionary of the class and also tracks basic benchmark parameters
        :param query: the query-string to submit to the climetlab-API of the application
        :param purpose: the name/purpose of the retireved data (used to append the data-dictionary)
        :return: appended self.data-dictionary with {purpose: xr.Dataset}
        """
        super().append_data(query, purpose, hour=self.hour)

        self.data_info["nsamples"] = {key: ds.dims[self.sample_dim] for (key, ds) in self.data.items()}

    def get_data(self, query: str, datafile: str, hour: int = 12):
        """
        Depending on the flag ldownload_last, data is either downloaded from the s3-bucket or read from the file system.
        :param query: a query-string to retrieve the data from the s3-bucket
        :param datafile: the name of the datafile in which the retireved data is stored (will be either read or created)
        :return: xarray-Datasets for training, validation and testing (loaded to memory) and elapsed time
        """

        method = HandleDataClass.get_data.__name__

        if self.ldownload_last:
            try:
                print("%{0}: Start downloading the data...".format(method))
                # download the data from ECMWF's s3-bucket
                cmlds = cml.load_dataset(self.application, dataset=query)
                # convert to xarray datasets and...
                ds = cmlds.to_xarray()
                # ...save to disk
                _ = self.ds_to_netcdf(ds, datafile)
            except Exception as err:
                print("%{0}: Failed to download data files for query '{1}' for application {2} from s3-bucket."
                      .format(method, query, self.application))
                raise err
        else:
            try:
                print("%{0}: Start reading the data from '{1}'...".format(method, self.datadir))
                ds = xr.open_dataset(datafile)
                # Note 2022-02-01: don't slice here since this is done during preprocessing
                ds = ds.load()#.sel(time=dt.time(hour)).load()
            except Exception as err:
                print("%{0}: Failed to read file '{1}' corresponding to query '{2}'"
                                   .format(method, datafile, query))
                raise err

            print("%{0}: Dataset was retrieved succesfully.".format(method))

        return ds

    def normalize(self, ds_name: str, daytime: int = None, opt_norm: dict ={}):
        """
        Preprocess the data for feeding into the U-net, i.e. conversion to data arrays incl. z-score normalization
        :param ds_name: name of the dataset, i.e. one of the following "train", "val", "test"
        :param daytime: daytime in UTC for temporal slicing
        :param opt_norm: dictionary holding data for z-score normalization of data ("mu_in", "std_in", "mu_tar", "std_tar")
        :return: normalized data ready to be fed to U-net model
        """
        t0 = timer()

        norm_dims_t = ["time"]  # normalization of 2m temperature for each grid point
        norm_dims_z = ["time", "lat", "lon"]  # 'global' normalization of surface elevation

        # slice the dataset
        dsf = self.data[ds_name]
        if daytime is not None:
            dsf = dsf.sel(time=dt.time(daytime))

        # retrieve and normalize input and target data
        if not opt_norm:
            t2m_in, t2m_in_mu, t2m_in_std = self.z_norm_data(dsf["t2m_in"], dims=norm_dims_t, return_stat=True)
            t2m_tar, t2m_tar_mu, t2m_tar_std = self.z_norm_data(dsf["t2m_tar"], dims=norm_dims_t, return_stat=True)
        else:
            t2m_in = self.z_norm_data(dsf["t2m_in"], mu=opt_norm["mu_in"], std=opt_norm["std_in"])
            t2m_tar = self.z_norm_data(dsf["t2m_tar"], mu=opt_norm["mu_tar"], std=opt_norm["std_tar"])

        z_in, z_tar = self.z_norm_data(dsf["z_in"], dims=norm_dims_z), self.z_norm_data(dsf["z_tar"], dims=norm_dims_z)

        in_data = xr.concat([t2m_in, z_in, z_tar], dim="variable")
        tar_data = xr.concat([t2m_tar, z_tar], dim="variable")

        # re-order data
        in_data = in_data.transpose("time",...,"variable")
        tar_data = tar_data.transpose("time",...,"variable")
        if not opt_norm:
            opt_norm = {"mu_in": t2m_in_mu, "std_in": t2m_in_std,
                        "mu_tar": t2m_tar_mu, "std_tar": t2m_tar_std}
            self.timing["normalizing_{0}".format(ds_name)] = timer() - t0
            return in_data, tar_data, opt_norm
        else:
            self.timing["normalizing_{0}".format(ds_name)] = timer() - t0
            return in_data, tar_data

    @staticmethod
    def denormalize(data: arr_xr_np, mu: float, std: float):
        """
        Denoramlize data using z-score normalization.
        :param data: The data to be denormalized.
        :param mu: The mean of the data (first moment).
        :param std: The standard deviation of the data (second moment).
        :return: the denormalized data
        """
        data_denorm = HandleUnetData.z_norm_data(data, norm_method="denorm", mu=mu, std=std)

        return data_denorm

    @staticmethod
    def z_norm_data(data: arr_xr_np, norm_method="norm",save_path:str=None, dims=None, return_stat: bool = False):
        """
        Perform z-score normalization on the data
        :param data: the data as xarray-Dataset
        :param norm_method: 'direction' of normalization, i.e. "norm" for normalization, "denorm" for denormalization
        :param save_path: the path to the json file that save the statstic information
        :param dims: list of dimension over which statistical quantities for normalization are calculated
        :param return_stat: flag if normalization statistics are returned
        :return: the normalized data
        """
        method = HandleUnetData.z_norm_data.__name__
        js_file = os.path.join(save_path,"norm_dict.json")  
        norm_dict = {}
        assert isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray), \
                   "%{0}: data must be a xarray Dataset or Array.".format(method)

        if not dims:
            dims = list(data.dims)
        

        if os.path.exists(js_file):
            print("Loading file:",js_file)
            with open(js_file, "r") as f:
                norm_dict = json.load(f)
                print("norm_dict mu", norm_dict["mu"] )
                keys = list(norm_dict["mu"].keys())
                mu = np.asarray(list(norm_dict["mu"].values()))
                std = np.asarray(list(norm_dict["std"].values()))
                print("Mu here after loading file:",mu)
                mu = xr.DataArray(mu, coords={"variables": keys}, dims=["variables"])
                std = xr.DataArray(std, coords={"variables": keys}, dims=["variables"])
        else:
           mu = data.mean(dim=dims)
           std = data.std(dim=dims)
           mu_dict, std_dict = mu.to_dict(), std.to_dict()
           print("dict_test", mu_dict)
           dict_mu, dict_std = {}, {}
           dt_mu, dt_std = mu_dict["data"], std_dict["data"]
           keys =  mu_dict['coords']['variables']['data']

           for i, var in enumerate(keys):
               dict_mu[var] = dt_mu[i]
               dict_std[var] = dt_std[i]

           norm_dict["mu"] = dict_mu
           norm_dict["std"] = dict_std
           with open(js_file,"w") as f:
               json.dump(norm_dict,f)
               print("The training stats mu and std are saved to ", js_file)
                   

        if norm_method == "norm":
            data_out = (data - mu) / std
        elif norm_method == "denorm":
            data_out = data * std + mu
        else:
            raise ValueError("%{0}: norm_metod must be either 'norm' or 'denorm'.".format(method))

        if return_stat:
            return data_out, mu, std
        else:
            return data_out


