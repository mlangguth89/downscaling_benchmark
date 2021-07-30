# ********** Info **********
# @Creation: 2021-07-28
# @Update: 2021-07-28
# @Author: Michael Langguth
# @Site: Juelich supercomputing Centre (JSC) @ FZJ
# @File: tfrecords_utils.py
# ********** Info **********

import os
import glob
import subprocess as sp
import sys
import datetime as dt
import numpy as np
import xarray as xr
import pandas as pd
import json as js
import tensorflow as tf
from helper_utils import ensure_datetime, extract_date, subset_files_on_date


class IFS2TFRecords(object):
    class_name = "IFS2TFRecords"

    date_fmt = "%Y-%m-%dT%H:%M"

    def __init__(self, tfr_dir: str, example_nc_file: str, create_tfr_dir: bool = True):

        method = "%{0}->{1}".format(IFS2TFRecords.class_name, IFS2TFRecords.__init__.__name__)
        self.tfr_dir = tfr_dir
        self.meta_data = os.path.join(self.tfr_dir, "metadata.json")
        self.example_nc_data_file = example_nc_file
        meta_dict = self.get_and_write_metadata()
        self.variables = meta_dict["coordinates"]["variable"]
        self.data_dim = (meta_dict["shape"]["nvars"], meta_dict["shape"]["nlat"], meta_dict["shape"]["nlon"])

        if not os.path.isdir(self.tfr_dir):
            if create_tfr_dir:
                os.makedirs(tfr_dir)
                print("%{0}: TFRecords directory has been created since it was not existing.".format(method))
            else:
                raise NotADirectoryError("%{0}: TFRecords-directory does not exist.".format(method) +
                                         "Either create it manually or set create_tfr_dir to True.")

    def get_and_write_metadata(self):

        method = IFS2TFRecords.get_and_write_metadata.__name__

        if not os.path.isfile(self.example_nc_data_file):
            raise FileNotFoundError("%{0}: netCDF-file '{1} does not exist.".format(method, self.example_nc_data_file))

        with xr.open_dataset(self.example_nc_data_file) as ds:
            da = ds.to_array().squeeze(drop=True)
            vars_nc = da["variable"].values
            lat, lon = da["lat"].values, da["lon"].values
            nlat, nlon, nvars = len(lat), len(lon), len(vars_nc)

        meta_dict = {"coordinates": {"variable": vars_nc.to_list(),
                                     "lat": np.round(lat, decimals=2).tolist(),
                                     "lon": np.round(lon, decimals=2).tolist()},
                     "shape": {"nvars": nvars, "nlat": nlat, "nlon": nlon}}

        if not os.path.isfile(self.meta_data):
            with open(self.meta_data, "w") as js_file:
                js.dump(meta_dict, js_file)

        return meta_dict

    def get_data_from_file(self, fname):

        method = IFS2TFRecords.get_data_from_file.__name__

        suffix_tfr = ".tfrecords"
        tfr_file = os.path.join(self.tfr_dir, fname+".tfrecords" if fname.endswith(suffix_tfr) else fname)

        if not os.path.isfile(tfr_file):
            raise FileNotFoundError("%{0}: TFRecord-file '{1}' does not exist.".format(method, tfr_file))

        data = tf.data.TFRecordDataset(tfr_file)

        data = data.map(IFS2TFRecords.parse_one_data_arr)

        return data

    def write_monthly_data_to_tfr(self, dir_in, hour=None):
        """
        Use dates=pd.date_range(start_date, end_date, freq="M", normalize=True)
        and then dates_red = dates[dates.quarter.isin([2,3])] for generating year_months
        """

        method = "%{0}->{1}".format(IFS2TFRecords.class_name, IFS2TFRecords.write_monthly_data_to_tfr.__name__)

        if not os.path.isdir(dir_in):
            raise NotADirectoryError("%{0}: Passed directory does not exist.".format(method))

        nc_files = glob.glob(os.path.join(dir_in, "*.nc"))

        if not nc_files:
            raise FileNotFoundError("%{0}: No netCDF-files found in '{1}'".format(method, dir_in))

        if hour is None:
            pass
        else:
            nc_files = subset_files_on_date(nc_files, int(hour))
        # create temprorary working directory to merge netCDF-files into a single one
        #tmp_dir = os.path.join(dir_in, os.path.basename(dir_in) + "_subset")
        #os.mkdir(tmp_dir)
        #for nc_file in nc_files:
        #    os.symlink(nc_file, os.path.join(tmp_dir, os.path.basename(nc_file)))

        #dest_file = os.path.join(tmp_dir, "merged_data.nc")
        #cmd = "cdo mergetime ${0}/*.nc ${1}".format(tmp_dir, dest_file)

        #_ = sp.check_output(cmd, shell=True)

        with xr.open_mfdataset(nc_files) as dataset:
            data_arr_all = dataset.to_array()
            data_arr_all = data_arr_all.transpose("time", "variable", ...)

        dims2check = data_arr_all.isel(time=0).squeeze().shape()
        vars2check = list(data_arr_all["variables"].values)
        assert dims2check == self.data_dim, \
               "%{0}: Shape of data from netCDF-file list {1} does not match expected shape {2}"\
               .format(method, dims2check, self.data_dim)

        assert vars2check == self.variables, "%{0} Unexpected set of varibales {1}".format(method, ",".join(vars2check))

        IFS2TFRecords.write_dataset_to_tfr(data_arr_all, self.tfr_dir)

    @staticmethod
    def write_dataset_to_tfr(data_arr: xr.DataArray, dirout:str):

        method = IFS2TFRecords.write_dataset_to_tfr.__name__

        assert isinstance(data_arr, xr.DataArray), "%{0}: Input data must be a data array, but is of type {1}."\
                                                   .format(method, type(data_arr))

        assert os.path.isdir(dirout), "%{0}: Output directory '{1}' does not exist.".format(method, dirout)

        date_fmt = "%Y%m%d%H"

        try:
            times = data_arr["time"]
            ntimes = len(times)
            date_start, date_end = ensure_datetime(times[0]), ensure_datetime(times[-1])
            tfr_file = os.path.join(dirout, "ifs_data_{0}_{1}.tfrecords".format(date_start.strftime(date_fmt),
                                                                                date_end.strftime(date_fmt)))

            with tf.io.TFRecordWriter(tfr_file) as tfr_writer:
                for time in times:
                    out = IFS2TFRecords.parse_one_data_arr()
                    tfr_writer.write(out.SerializeToString())

            print("%{0}: Wrote {1:d} elements to TFRecord-file '{2}'".format(method, ntimes, tfr_file))
        except Exception as err:
            print("%{0}: Failed to write DataArray to TFRecord-file. See error below.".format(method))
            raise err

    @staticmethod
    def parse_one_data_arr(data_arr):

        method = "%{0}->{1}".format(IFS2TFRecords.class_name, IFS2TFRecords.parse_one_data_arr.__name__)

        date_fmt = IFS2TFRecords.date_fmt
        # sanity checks
        if not isinstance(data_arr, xr.DataArray):
            raise ValueError("%{0}: Input dataset must be a xarray Dataset, but is of type '{1}'"
                             .format(method, type(data_arr)))

        assert data_arr.ndim == 3, "%{0}: Data array must have rank 3, but is of rank {1:d}.".format(method,
                                                                                                     data_arr.ndim)
        dim_sh = data_arr.shape

        data_dict = {"nvars": IFS2TFRecords._int64_feature(dim_sh[0]),
                     "nlat": IFS2TFRecords._int64_feature(dim_sh[1]),
                     "nlon": IFS2TFRecords._int64_feature(dim_sh[2]),
                     "variable": IFS2TFRecords._bytes_list_feature(data_arr["variable"].values),
                     "time": IFS2TFRecords._bytes_feature(ensure_datetime(data_arr["time"][0].values)
                                                          .strftime(date_fmt)),
                     "data_array": IFS2TFRecords._bytes_feature(IFS2TFRecords.serialize_array(data_arr.values))
                     }

        # create an Example with wrapped features
        out = tf.train.Example(features=tf.train.Features(feature=data_dict))

        return out

    # Methods to convert data to TF protocol buffer messages
    @staticmethod
    def serialize_array(array):
        method = IFS2TFRecords.serialize_array.__name__

        # method that should be used locally only
        def map_np_dtype(arr, name="arr"):
            dict_map = {name: tf.dtypes.as_dtype(arr[0].dtype)}
            return dict_map

        try:
            dtype_dict = map_np_dtype(array)
            new = tf.io.serialize_tensor(array)
        except Exception as err:
            assert type(array) == np.array, "%{0}: Input data must be a numpy array.".format(method)
            raise err
        return new, dtype_dict

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _bytes_list_feature(values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    @staticmethod
    def _floats_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
