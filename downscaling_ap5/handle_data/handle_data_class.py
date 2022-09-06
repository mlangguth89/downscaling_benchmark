__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2022-02-01"

import os, sys
import socket
import gc
from collections import OrderedDict
from timeit import default_timer as timer
import xarray as xr
import tensorflow as tf


class HandleDataClass(object):

    def __init__(self, datadir: str, application: str, query: str, purpose: str = None, **kwargs) -> None:
        """
        Initialize Input data object by reading data from netCDF-files
        :param datadir: the directory from where netCDF-files are located (or should be located if downloaded)
        :param application: name of application (must coincide with name in s3-bucket)
        :param query: query string which can be used to load data from the s3-bucket of the application
        :param purpose: optional name to indicate the purpose of queried data (used as key for the data-dictionary)
        """
        method = HandleDataClass.__init__.__name__

        self.host = os.getenv("HOSTNAME") if os.getenv("HOSTNAME") is not None else "unknown"
        purpose = query if purpose is None else purpose
        self.application = application
        self.datadir = datadir
        if not os.path.isdir(datadir):
            os.makedirs(datadir)
        self.ldownload_last = None

        self.data, self.timing, self.data_info = self.handle_data_req(query, purpose, **kwargs)

    def handle_data_req(self, query: str, purpose, **kwargs):
        """
        Handles a data-query by parsing it to the get_data function
        :param query: the query-string to submit to the climetlab-API of the application
        :param purpose: the name/purpose of the retireved data (used to append the data-dictionary)
        :return: the xr.Dataset retireved from get_data and dictionaries for the loading time and the memory consumption
        """
        method = HandleDataClass.handle_data_req.__name__

        datafile = os.path.join(self.datadir, "{0}_{1}.nc".format(self.application, purpose))
        self.ldownload_last = self.set_download_flag(datafile)
        # time data retrieval
        t0_load = timer()
        ds = self.get_data(query, datafile, **kwargs)
        load_time = timer() - t0_load
        if self.ldownload_last:
            print("%{0}: Downloading took {1:.2f}s.".format(method, load_time))
            _ = HandleDataClass.ds_to_netcdf(ds, datafile)

        data = OrderedDict({purpose: ds})
        timing = {"loading_times": {purpose: load_time}}
        data_info = {"memory_datasets": {purpose: ds.nbytes}}

        return data, timing, data_info

    def append_data(self, query: str, purpose: str = None, **kwargs):
        """
        Appends data-dictionary of the class and also tracks basic benchmark parameters
        :param query: the query-string to submit to the climetlab-API of the application
        :param purpose: the name/purpose of the retireved data (used to append the data-dictionary)
        :return: appended self.data-dictionary with {purpose: xr.Dataset}
        """
        purpose = query if purpose is None else purpose
        ds_app, timing_app, data_info_app = self.handle_data_req(query, purpose, **kwargs)

        self.data.update(ds_app)
        self.timing["loading_times"].update(timing_app["loading_times"])
        self.data_info["memory_datasets"].update(data_info_app["memory_datasets"])
        
    def set_download_flag(self, datafile):
        """
        Depending on the hosting system and on the availability of the dataset on the filesystem
        (stored under self.datadir), the download flag is set to False or True. Also returns a dictionary for the
        respective netCDF-filenames.
        :return: Boolean flag for downloading and dictionary of data-filenames
        """
        method = HandleDataClass.set_download_flag.__name__

        ldownload = True if "login" in self.host else False        
        stat_file = os.path.isfile(datafile)

        if stat_file and ldownload:
            print("%{0}: Datafiles are already available under '{1}'".format(method, self.datadir))
            ldownload = False
        elif not stat_file and not ldownload:
            raise ValueError("%{0}: Data is not available under '{1}',".format(method, self.datadir) +
                             "but downloading on computing node '{0}' is not possible.".format(self.host))

        return ldownload

    def get_data(self, *args):
        """
        Function to either downlaod data from the s3-bucket or to read from file.
        """
        raise NotImplementedError("Please set-up a customized get_data-function.")

    @staticmethod
    def split_in_tar(da: xr.DataArray, target_var: str = "t2m") -> (xr.DataArray, xr.DataArray):
        """
        Split data array with variables-dimension into input and target data for downscaling.
        :param da: The unsplitted data array.
        :param target_var: Name of target variable which should consttute the first channel
        :return: The splitted data array.
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

    @staticmethod
    def make_tf_dataset(da: xr.DataArray, batch_size: int, lshuffle: bool = True, shuffle_samples: int = 20000,
                        lembed: bool = None) -> tf.data.Dataset:
        """
        Build-up TensorFlow dataset from a generator based on the xarray-data array.
        Note that all data is loaded into memory.
        :param da: the data-array from which the dataset should be cretaed
        :param batch_size: number of samples per mini-batch
        :param lshuffle: flag if shuffling should be applied to dataset
        :param shuffle_samples: number of samples to load before applying shuffling
        :param lembed: flag to trigger temporal embedding (not implemented yet!)
        """
        da = da.load()
        da_in, da_tar = HandleDataClass.split_in_tar(da)

        def gen(darr_in, darr_tar):
            # darr_in, darr_tar = darr_in.load(), darr_tar.load()
            ntimes = len(darr_in["time"])
            for t in range(ntimes):
                yield tuple((darr_in.isel({"time": t}).values, darr_tar.isel({"time": t}).values))

        s0 = next(iter(gen(da_in, da_tar)))
        sample_spec = {"shape_in": s0[0].shape, "type_in": s0[0].dtype,
                       "shape_tar": s0[1].shape, "type_tar": s0[0].dtype}

        gen_train = gen(da_in, da_tar)

        if lembed:
            if not lembed:
                raise ValueError("Time embedding is not supported yet.")
            # data_iter = tf.data.Dataset.from_tensor_slices((ds_in, ds_tar, embed))
        else:
            data_iter = tf.data.Dataset.from_generator(lambda: gen_train,
                                                       output_signature=(tf.TensorSpec(sample_spec["shape_in"],
                                                                                       dtype=sample_spec["type_in"]),
                                                                         tf.TensorSpec(sample_spec["shape_tar"],
                                                                                       dtype=sample_spec["type_tar"])))
        # Notes:
        # * cache is reuqired to make repeat work properly on datasets based on generators
        #   (see https://stackoverflow.com/questions/60226022/tf-data-generator-keras-repeat-does-not-work-why)
        # * repeat must be applied after shuffle to get varying mini-batches per epoch
        # * batch-size is increaded to allow substepping in train_step
        if lshuffle:
            data_iter = data_iter.cache().shuffle(shuffle_samples).batch(batch_size, drop_remainder=True).repeat()
        else:
            data_iter = data_iter.cache().batch(batch_size, drop_remainder=True).repeat()

        # clean-up to free some memory
        del da
        gc.collect()

        return data_iter

    @staticmethod
    def ds_to_netcdf(ds: xr.Dataset, fname: str, comp_lvl=5):
        """
        Create dictionary for compressing all variables of dataset in netCDF-files
        :param ds: the xarray-dataset
        :param fname: name of the target netCDF-file
        :param comp_lvl: the compression level
        :return: True in case of success
        """
        method = HandleDataClass.ds_to_netcdf.__name__

        comp = dict(zlib=True, complevel=comp_lvl)
        try:
            encoding_ds = {var: comp for var in ds.data_vars}
            print("%{0}: Save dataset to netCDF-file '{1}'".format(method, fname))
            ds.to_netcdf(path=fname, encoding=encoding_ds)  # , engine="scipy")
        except Exception as err:
            print("%{0}: Failed to handle and save input dataset.".format(method))
            raise err

        return True

    @staticmethod
    def has_internet():
        """
        Checks if Internet connection is available.
        :return: True if connected, False else.
        """
        try:
            # connect to the host -- tells us if the host is actually
            # reachable
            socket.create_connection(("1.1.1.1", 53), timeout=5)
            return True
        except OSError:
            pass
        return False



