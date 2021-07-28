# ********** Info **********
# @Creation: 2021-07-28
# @Update: 2021-07-28
# @Author: Michael Langguth
# @Site: Juelich supercomputing Centre (JSC) @ FZJ
# @File: tfrecords_utils.py
# ********** Info **********

import os
import glob
import sys
import datetime as dt
import numpy as np
import xarray as xr
import pandas as pd
import json as js
import tensorflow as tf
from helper_utils import ensure_datetime


class IFS2TFRecords(object):
    class_name = "IFS2TFRecords"

    date_fmt = "%Y-%m-%dT%H:%M"

    def __init__(self, tfr_dir: str, create_tfr_dir: bool = True):

        method = "%{0}->{1}".format(IFS2TFRecords.class_name, IFS2TFRecords.__init__.__name__)
        self.tfr_dir = tfr_dir
        self.meta_data = os.path.join(self.tfr_dir, "metadata.json")
        self.data_dim = None

        if not os.path.isdir(self.tfr_dir):
            if create_tfr_dir:
                os.makedirs(tfr_dir)
                print("%{0}: TFRecords directory has been created since it was not existing.".format(method))
            else:
                raise NotADirectoryError("%{0}: TFRecords-directory does not exist.".format(method) +
                                         "Either create it manually or set create_tfr_dir to True.")

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
            nc_files_dates = [(extract_date(nc_file)).strftime("%H") for nc_file in nc_files]
            inds = [idx for idx, s in enumerate(dates_hour) if "05" in s]

            nc_files = nc_files[inds]
            tmp_dir = os.path.join(dir_in), os.path.basename(dir_in)+"_test")
            os.mkdir(tmp_dir)
            for nc_file in nc_files:
                os.symlink(nc_file, os.path.join(tmp_dir, os.path.basename(nc_file)))







    def write_dataset_to_tfr(self, dataset):

    def parse_one_dataset(self, dataset):

        method = "%{0}->{1}".format(IFS2TFRecords.class_name, IFS2TFRecords.parse_one_dataset.__name__)

        date_fmt = IFS2TFRecords.date_fmt
        # sanity checks
        if not isinstance(dataset, xr.Dataset):
            raise ValueError("%{0}: Input dataset must be a xarray Dataset, but is of type '{1}'"
                             .format(method, type(dataset)))

        assert np.shape(dataset) == self.data_dim, "%{0}: Shape of data {1} does not fit expected shape of data {2}" \
            .format(method, np.shape(dataset), self.data_dim)

        data_dict = {"nvars": IFS2TFRecords._int64_feature(self.data_dim[0]),
                     "nlat": IFS2TFRecords._int64_feature(self.data_dim[1]),
                     "nlon": IFS2TFRecords._int64_feature(self.data_dim[2]),
                     "variable": IFS2TFRecords._bytes_feature(self.variables),
                     "time": IFS2TFRecords._bytes_list_feature(ensure_datetime(dataset["time"][0]).strftime(date_fmt))
                     "data_array": IFS2TFRecords._bytes_feature(IFS2TFRecords.serialize_array(dataset.values))
                     }

        if not os.path.isfile(self.meta_data):
            _ = self.write_metadata(dataset)

        # create an Example with wrapped features
        out = tf.train.Example(features=tf.train.Features(feature=data_dict))

        return out


    def write_metadata(self, dataset: xr.Dataset):
        a = 5

        return True

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
