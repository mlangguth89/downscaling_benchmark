# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Methods to handle data for the neural networks.

To-Dos:
    - Shuffle indices for sharding in make_tf_dataset_all-method
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2024-07-30"

import os, glob
from typing import List, Tuple, Union, Dict
from pathlib import Path
import re
from operator import itemgetter
from functools import partial
import socket
import gc
from collections import OrderedDict
from timeit import default_timer as timer
import random
import numpy as np
import xarray as xr
import tensorflow as tf
from abstract_data_normalization import Normalize
import multiprocessing
try:
    from multiprocessing import Pool as ThreadPool
except:
    from multiprocessing.pool import ThreadPool
from all_normalizations import ZScore
from other_utils import to_list, find_closest_divisor, finditem


class HandleDataClass(object):

    def __init__(self, datadir: str, application: str, query: str, purpose: str = None, **kwargs) -> None:
        """
        Initialize Input data object by reading data from netCDF-files
        :param datadir: the directory from where netCDF-files are located (or should be located if downloaded)
        :param application: name of application (must coincide with name in s3-bucket)
        :param query: query string which can be used to load data from the s3-bucket of the application
        :param purpose: optional name to indicate the purpose of queried data (used as key for the data-dictionary)
        """
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
    def reshape_ds(ds):
        """
        Convert a xarray dataset to a data-array where the variables will constitute the last dimension (channel last)
        :param ds: the xarray dataset with dimensions (dims)
        :return da: the data-array with dimensions (dims, variables)
        """
        da = ds.to_array(dim="variables")
        da = da.transpose(..., "variables")
        return da

    @staticmethod
    def split_in_tar(ds: xr.Dataset, predictands: List = None, predictors: List = None, static_vars: List = None) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Split data array with variables-dimension into input and target data for downscaling
        :param ds: The unsplitted dataset
        :param predictands: List of selected predictand variables; parse None to use
                            all predictands (vars with suffix _tar)
        :param predictors: List of selected predictor variables; parse None to use all predictors (vars with suffix _in)
        :param static_predictors: List of selected static (high-resolved) predictors, the corresponding splitted dataset ds_stat will be None of None is parsed
        :return: Tuple of splitted datasets.
        """
        varnames = list(ds.data_vars)

        if predictors is None:
            invars = [var for var in varnames if var.endswith("_in")]
        else:
            assert all(
                [predictor in varnames for predictor in predictors]
            ), f"At least one predictor is not a data variable. Available variables are {*varnames,}"
            invars = list(predictors)
        if predictands is None:
            tarvars = [var for var in varnames if var.endswith("_tar")]
        else:
            assert all(
                [predictand in varnames for predictand in predictands]
            ), f"At least one predictor is not a data variable. Available variables are {*varnames,}"
            tarvars = list(predictands)

        ds_in, ds_tar = ds[invars], ds[tarvars]

        if static_vars is None:
            ds_stat = None
        else:
            assert all(
                [static_var in varnames for static_var in static_vars]
            ), f"At least ostatic high-res is not a data variable. Available variables are {*varnames,}"
            statvars = list(static_vars)

            ds_stat = ds[statvars]

        return ds_in, ds_tar, ds_stat

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


def get_dataset_filename(datadir: str, dataset_name: str, subset: str, laugmented: bool = False):

    allowed_subsets = ("train", "val", "test")

    if subset in allowed_subsets:
        pass
    else:
        raise ValueError(f"Unknown dataset subset '{subset}' chosen. Allowed subsets are {*allowed_subsets,}")

    fname_suffix = "downscaling"

    if dataset_name == "tier1":
        fname_suffix = f"{fname_suffix}_{dataset_name}_{subset}"
        if laugmented: fname_suffix = f"{fname_suffix}_aug"
    elif dataset_name == "tier2":
        fname_suffix = f"{fname_suffix}_{dataset_name}_{subset}"
        if subset == "train":
            fname_suffix = f"{fname_suffix}*"
        if laugmented: raise ValueError("No augmented dataset available for Tier-2.")
    elif dataset_name == "atmorep":
        fname_suffix = f"{fname_suffix}_{dataset_name}_{subset}"
        if subset == "train":
            fname_suffix = f"{fname_suffix}*"
        if laugmented: raise ValueError("No augmented dataset available for AtmoRep.")
    elif dataset_name == "benchmark_t2m":
        fname_suffix = f"{fname_suffix}_{dataset_name}_{subset}"
        if subset == "train":
            fname_suffix = f"{fname_suffix}*"
        if laugmented: raise ValueError("No augmented dataset available for benchmark_t2m.")
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}' passed.")

    if "*" in fname_suffix:
        ds_filename = fname_suffix
    else:
        ds_filename = os.path.join(datadir, f"{fname_suffix}.nc")

        if not os.path.isfile(ds_filename):
            raise FileNotFoundError(f"Could not find requested dataset file '{ds_filename}'")

    return ds_filename


def prepare_dataset(datadir: str, dataset_name: str, ds_dict: dict, hparams_dict: dict, mode: str,
                    norm_dims: List=None, norm_obj=None, shuffle: bool = True, nworkers: int = 10, lrepeat: bool = True,
                    drop_remainder: bool = True, with_horovod: bool = False, seed: int = None):
    """
    Prepare training data for downscaling
    :param datadir: directory where netCDF-files for TF dataset are strored
    :param dataset_name: name of dataset to be loaded
    :param ds_dict: dictionary of dataset names and their subsets
    :param hparams_dict: dictionary of hyperparameters
    :param mode: mode of dataset (train, val, test)
    :param norm_dims: names of dimension over which normalization is applied. Should be None if norm_obj is parsed
    :param norm_obj: normalization instance used to normalize the data.
                     If not passed, the normalization instance is retrieved from the data
    :param shuffle: flag if shuffling should be applied to dataset
    :param nworkers: numbers of workers to read in netCDF-files (for the case where NOT all data is loaded into memory)
    :param lrepeat: flag if dataset should be repeated
    :param drop_remainder: flag if samples will be dropped in case batch size is not a divisor of # data samples
    :param with_horovod: flag if horovod is used for distributed training
    :param seed: seed for random shuffling of datafiles
    :return: tuple of (TensorFlow dataset object, dictionary of dataset information)
    """
    # check parsed mode
    allowed_modes = ["train", "val", "test"]
    assert mode in allowed_modes, f"{mode} is not a valid mode. Allowed modes are {*allowed_modes,}" 

    # check if normalization object is provided (mandatory for validation and test mode)
    if mode != "train" and not norm_obj:
        raise ValueError(f"Normalization object norm_obj must be provided for mode {mode}.")
    else:
        assert norm_obj or norm_dims, f"Neither norm_obj nor norm_dims has been provided."

    if norm_obj and norm_dims:
        print("WARNING: norm_obj and norm_dims have been passed. norm_dims will be ignored.")
        norm_dims = None

    if norm_obj: assert isinstance(norm_obj, Normalize), "norm_obj is not an instance of the Normalize-class."

    # check if target variables are provided
    varnames_tar_all = to_list(ds_dict["predictands"]).copy()
    if finditem(hparams_dict, "z_branch", False):
        varnames_tar_all += to_list(ds_dict["varname_z"])

    # Note: bs_train is introduced to allow substepping in the training loop, e.g. for WGAN where n optimization steps
    # are applied to train the critic, before the generator is trained once.
    # The validation and test dataset however do not perform substeeping and thus don't require an increased mini-batch size.
    if mode == "train":
        bs_train = ds_dict["batch_size"] * (hparams_dict["d_steps"] + 1) if "d_steps" in hparams_dict else ds_dict["batch_size"]
        nepochs = hparams_dict["nepochs"] * (hparams_dict["d_steps"] + 1) if "d_steps" in hparams_dict else hparams_dict["nepochs"]
    else:
        bs_train = ds_dict["batch_size"]
        nepochs = hparams_dict["nepochs"]

    
    fname_or_pattern = get_dataset_filename(datadir, dataset_name, mode)

    # backward compatibility for deprecated keys var_tar2in and named_targets in ds_dict and hparams_dict, respectively
    if "var_tar2in" in ds_dict:
        static_predictors = list(ds_dict["var_tar2in"]) + ds_dict.get("static_predictors", [])
        print("Warning: The usage of 'var_tar2in' is deprecated. Use 'static_predictors' instead. \n"
              +f"List of updated static predictors: {','.join(static_predictors)}")
    else:
        static_predictors = ds_dict.get("static_predictors", None)
    
    if hparams_dict.get("named_targets", False):
        stream_mode = "hi_input_named_target"
    else:
        stream_mode = hparams_dict.get("stream_mode", "hi_input")

    if not "stream_mode" in hparams_dict:
        print(f"Warning: stream_mode not provided in hparams_dict. Autmotically set to '{stream_mode}'.")

    if "*" in fname_or_pattern:                                             # do not load all data into memory
        ds_obj = StreamMonthlyNetCDF(stream_mode, datadir, fname_or_pattern, nfiles_merge=ds_dict["num_files"],
                                     predictands=varnames_tar_all, predictors=ds_dict.get("predictors", None),
                                     static_predictors=static_predictors, sample_dim=ds_dict.get("sample_dim", "time"),
                                     norm_obj=norm_obj, norm_dims=norm_dims, with_horovod=with_horovod, seed=seed, nworkers=nworkers)
        
        if shuffle:
            nshuffle = ds_obj.samples_merged
        else:
            nshuffle = 1          # equivalent to no shuffling

        tfds = make_tf_dataset_dyn(ds_obj, bs_train, nepochs, nshuffle=nshuffle, lrepeat=lrepeat, drop_remainder=drop_remainder)
        
        tfds_info = {"nsamples": ds_obj.nsamples, "data_norm": ds_obj.data_norm, "shape_in": (*ds_obj.data_dim[::-1], ds_obj.n_predictors),
                     "dataset_size": ds_obj.dataset_size, "ds_obj": ds_obj, "all_predictands": varnames_tar_all, "file": ds_obj.file_list,
                     "effective_dataset_size": ds_obj.effective_dataset_size, "predictors": ds_obj.predictor_list, 
                     "static_predictors": static_predictors, "stream_mode": stream_mode}
    else:                                                                   # load all data into memory
        ds = xr.open_dataset(fname_or_pattern)

        if not norm_obj:
            # norm_obj must be freshly instantiated (triggering later parameter retrieval)
            norm_obj = ZScore(ds_dict["norm_dims"])

        ds = norm_obj.normalize(ds)

        nsamples = len(ds["time"])

        # get list of predictors (same approach as in split_in_tar-method) if it's None
        predictors=ds_dict.get("predictors", None)
        if predictors is None:
            predictors = [var for var in ds.data_vars if var.endswith("_in")]

        # create TensorFlow dataset
        tfds = make_tf_dataset_allmem(stream_mode, ds, bs_train, varnames_tar_all, predictors=predictors, 
                                      static_predictors=static_predictors, lrepeat=lrepeat, drop_remainder=drop_remainder,
                                      lshuffle=shuffle, with_horovod=with_horovod)
    
        # provide dict for later use
        tfds_info = {"nsamples": nsamples, "data_norm": norm_obj, "shape_in": tfds.element_spec[0].shape[1:].as_list(),
                     "dataset_size": ds.nbytes, "all_predictands": varnames_tar_all, "file": fname_or_pattern, 
                     "effective_dataset_size": ds.nbytes, "predictors": predictors, "static_predictors": static_predictors,
                     "stream_mode": stream_mode}
        
    return tfds, tfds_info

def make_tf_dataset_dyn(ds_obj, batch_size: int, nepochs: int, nshuffle: int, lrepeat: bool = True, drop_remainder: bool = True) -> tf.data.Dataset:
    """
    Build TensorFlow dataset by streaming from netCDF using xarray's open_mfdatset-method.
    To fit into memory, only a subset of all netCDF-files is processed at once (nfiles2merge-parameter).
    :param ds_obj: StreamMonthlyNetCDF-object
    :param batch_size: desired mini-batch size
    :param nepochs: (effective) number of epochs for training
    :param nshuffle: number of samples to shuffle (set to 1 to disable shuffling)
    :param lrepeat: flag if dataset should be repeated
    :param drop_remainder: flag if samples will be dropped in case batch size is not a divisor of # data samples
    :return: TensorFlow dataset object that streams data from subset of many netCDF-files
    """
    tf_read_nc = lambda ind_set: tf.py_function(ds_obj.read_netcdf, [ind_set], tf.int64)
    tf_choose_data = lambda il: tf.py_function(ds_obj.choose_data, [il], tf.bool)
    tf_getdata = lambda i: tf.numpy_function(ds_obj.getitems, [i], tf.float32)
    
    mode = ds_obj.stream_mode
    
    if mode in ["hi_input", "hi_input_named_target"]:
        tf_getdata = lambda i: tf.numpy_function(ds_obj.getitems, [i], tf.float32)
        if mode == "hi_input":
            tf_split = lambda arr: (arr[..., 0:-ds_obj.n_predictands], arr[..., -ds_obj.n_predictands:])
        else:
            varnames = ds_obj.predictand_list
            tf_split = lambda arr: (arr[..., 0:-ds_obj.n_predictands],
                                    {var: arr[..., -ds_obj.n_predictands + i] for i, var in enumerate(varnames)})
    else: 
        def make_dict(darr_in, darr_stat, darr_out):
            return {"lo_res_inputs": darr_in, "hi_res_inputs": darr_stat, "output": darr_out}
                                         
        tf_getdata = lambda i: tf.numpy_function(ds_obj.getitems, [i], [tf.float32, tf.float32, tf.float32])
        tf_split = lambda arr_in, arr_stat, arr_out: make_dict(arr_in, arr_stat, arr_out)

    # enable flexibility in factor for range
    n_reads = int(ds_obj.nfiles_merged*nepochs)
    if ds_obj.with_horovod:
        import horovod.tensorflow as hvd
        tfds = tf.data.Dataset.range(n_reads).shard(hvd.size(), hvd.rank()).map(tf_read_nc).prefetch(1)
    else:
        tfds = tf.data.Dataset.range(n_reads).map(tf_read_nc).prefetch(1)

    tfds = tfds.flat_map(lambda x: tf.data.Dataset.from_tensors(x).map(tf_choose_data))
    tfds = tfds.flat_map(
        lambda x: tf.data.Dataset.range(ds_obj.samples_merged).shuffle(nshuffle)
        .batch(batch_size, drop_remainder=drop_remainder).map(tf_getdata, num_parallel_calls=tf.data.AUTOTUNE))

    tfds = tfds.map(tf_split, num_parallel_calls=tf.data.AUTOTUNE)
    
    if lrepeat:
        tfds = tfds.repeat()

    return tfds

def make_tf_dataset_allmem(stream_mode: str, ds: xr.Dataset, batch_size: int, predictands: List, predictors: List = None,
                           static_predictors: List = None, lshuffle: bool = True, shuffle_samples: int = 20000,
                           lrepeat: bool = True, drop_remainder: bool = True, with_horovod: bool = False) -> tf.data.Dataset:
    """
    Build-up TensorFlow dataset from a generator based on the xarray-data array.
    NOTE: All data is loaded into memory
    :param stream_mode: choices: ['hi_input', 'hi_input_named_target', 'lo_input']
                'hi_input': (bilinearly) upscaled input data available from data file
                'hi_input_named_target': as 'hi_input', but with named target (output as dictionary with variable names as key)
                'lo_input': input data on coarse grid, but high-resolved static predictors available
                            data pipeline yield a dictionary with 'lo_input', 'static' and 'output' as keys
    :param ds: the xarray dataset. Input variable names must carry the suffix '_in', whereas it must be '_tar' for target variables
    :param batch_size: number of samples per mini-batch
    :param predictands: List of selected predictand variables
    :param predictors: List of selected predictor variables; parse None to use all predictors (vars with suffix _in)
    :param static_predictors: List of static (high-resolved) variables serving as predictors 
    :param lshuffle: flag if shuffling should be applied to dataset
    :param shuffle_samples: number of samples to load before applying shuffling
    :param lrepeat: flag if dataset should be repeated
    :param drop_remainder: flag if samples will be dropped in case batch size is not a divisor of # data samples
    :param with_horovod: flag to trigger horovod-based distributed dataset creation
    """
    # add static predictors to predictors-list unless lo_input-streaming mode is chosen
    if stream_mode == "lo_input":
        assert static_predictors is not None, "Provide high-resolved static input predictors for stream_mode 'lo_input'"
    else:
        if static_predictors:
            predictors = to_list(static_predictors) + predictors
            static_predictors = None
            print(f"Static predictors added to predictors for data pipeline mode '{stream_mode}'")

    if with_horovod:
        import horovod.tensorflow as hvd
        ntimes = len(ds["time"])
        # To-Do: Indices should be shuffled to avoid daytime dependencies
        inds = list(range(hvd.rank(), ntimes, hvd.size()))
        ds = ds.isel({"time": inds})

    # add time dimension to constant variables
    for var in ds.data_vars:
        if "time" not in ds[var].dims:
            ds[var] = ds[var].expand_dims({"time": ds["time"]}, axis=0)

    ds_in, ds_tar, ds_stat = HandleDataClass.split_in_tar(ds, predictands=predictands, predictors=predictors,
                                                          static_predictors=static_predictors)    
    
    ds_list = [ds_in, ds_tar, ds_stat] if static_predictors is not None else [ds_in, ds_tar]

    # convert dataset to data arrays and load into memory
    da_list = [HandleDataClass.reshape_ds(ds).astype("float32", copy=True) for ds in ds_list]

    varnames_tar = da_list[1]["variables"].values

    def gen_named(darr_in, darr_tar):
        # darr_in, darr_tar = darr_in.load(), darr_tar.load()
        ntimes = len(darr_in["time"])
        for t in range(ntimes):
            tar_now = darr_tar.isel({"time": t})
            yield tuple((darr_in.isel({"time": t}).values,
                            {var: tar_now.sel({"variables": var}).values for var in varnames_tar}))

    def gen_unnamed(darr_in, darr_tar):
        # darr_in, darr_tar = darr_in.load(), darr_tar.load()
        ntimes = len(darr_in["time"])
        for t in range(ntimes):
            yield tuple((darr_in.isel({"time": t}).values, darr_tar.isel({"time": t}).values))

    def gen_dict(darr_in, darr_tar, darr_stat):
        ntimes = len(darr_in["time"])
        for t in range(ntimes):
            yield tuple(
                (
                    {
                        "lo_res_inputs": darr_in.isel({"time": t}).values,
                        "hi_res_inputs": darr_stat.isel({"time": t}).values,
                    },
                    {"output": darr_tar.isel({"time": t}).values},
                )
            )

    if stream_mode == "hi_input":
        gen_now = gen_unnamed
    elif stream_mode == "hi_input_named_target":
        gen_now = gen_named
    elif stream_mode == "lo_input":
        gen_now = gen_dict
    else: 
        raise ValueError(f"Mode {stream_mode} is not supported. Possible choices: 'hi_input', 'hi_input_named_target' and 'lo_input'")

    # create output signatures from first sample
    s0 = next(iter(gen_now(*da_list)))
    if stream_mode == "lo_input":
        sample_spec_in = {
            "lo_res_inputs": tf.TensorSpec(
                s0[0]["lo_res_inputs"].shape, dtype=s0[0]["lo_res_inputs"].dtype
            ),
            "hi_res_inputs": tf.TensorSpec(
                s0[0]["hi_res_inputs"].shape, dtype=s0[0]["hi_res_inputs"].dtype
            ),
        }

        sample_spec_tar = {
            "output": tf.TensorSpec(s0[1]["output"].shape, dtype=s0[1]["output"].dtype)
        }
    else:
        sample_spec_in = tf.TensorSpec(s0[0].shape, dtype=s0[0].dtype)
        if stream_mode == "hi_input_named_target":
            sample_spec_tar = {
                var: tf.TensorSpec(s0[1][var].shape, dtype=s0[1][var].dtype)
                for var in varnames_tar
            }
        else: 
            sample_spec_tar = tf.TensorSpec(s0[1].shape, dtype=s0[1].dtype)

    # re-instantiate the generator and build TF dataset
    gen_train = gen_now(*da_list)
    data_iter = tf.data.Dataset.from_generator(lambda: gen_train, output_signature=(sample_spec_in, sample_spec_tar))

    # Notes:
    # * cache is reuqired to make repeat work properly on datasets based on generators
    #   (see https://stackoverflow.com/questions/60226022/tf-data-generator-keras-repeat-does-not-work-why)
    # * repeat must be applied after shuffle to get varying mini-batches per epoch
    # * batch-size is increased to allow substepping in train_step
    if lshuffle:
        data_iter = data_iter.cache().shuffle(shuffle_samples).batch(batch_size, drop_remainder=drop_remainder)
    else:
        data_iter = data_iter.cache().batch(batch_size, drop_remainder=drop_remainder)

    if lrepeat:
        data_iter = data_iter.repeat()

    # clean-up to free some memory
    # free_mem([da, da_in, da_tar, varnames_tar])
    del ds
    del ds_list
    del da_list
    gc.collect()

    return data_iter

class StreamMonthlyNetCDF(object):
    def __init__(self, mode: str, datadir: Path, patt: str, nfiles_merge: Union[int, Dict], predictands: List,
                 predictors: List = None, static_predictors: List = None, sample_dim: str = "time", norm_dims: List = None,
                 norm_obj=None ,with_horovod: bool = False, seed: int = None, nworkers: int = 10):
        """
        Class object providing all methods to create a TF dataset that iterates over a set of (monthly) netCDF-files
        rather than loading all into memory. Instead, only a subset of all netCDF-files is loaded into memory.
        Furthermore, the class attributes provide key information on the handled dataset
        :param datadir: directory where set of netCDF-files are located
        :param patt: filename pattern to allow globbing for netCDF-files
        :param nfiles_merge: number of files per data subset loaded into memory (can be an integer or a dictionary like {"#GPUS=1": 33})
        :param predictands: list of predictand variables names to be obtained
        :param predictors: list of predictor variable names to be obtained, pass None
                           if all vars with suffix _in should be chosen
        :param static_predictors: list of static predictor variable names to be obtained
        :param sample_dim: name of dimension in the data over which sampling should be performed
        :param var_tar2in: predictand (target) variable that can be inputted as well
                          (e.g. static variables known a priori such as the surface topography)
        :param norm_dims: list of dimensions over which data will be normalized
        :param norm_obj: normalization object providing parameters for (de-)normalization
        :param with_horovod: flag to trigger horovod-based distributed dataset creation
        :param seed: seed for random sampling of netCDF-files
        :param nworkers: number of threads to read the netCDF-files
        """
        self.with_horovod = with_horovod
        if self.with_horovod:
            import horovod.tensorflow as hvd
        self.seed = seed
        self.stream_mode = mode
        self.data_dir = datadir
        # get file list and number of files to be merged for data subse
        self.file_list = patt
        self.nfiles = len(self.file_list)
        # get relevant data dimensions
        self.ds_all = xr.open_mfdataset(list(self.file_list), decode_cf=False, cache=False)  # , parallel=True)
        self.sample_dim = sample_dim
        self.nsamples = self.ds_all.dims[sample_dim]
        self.data_dim = self.get_data_dim()
        self.dataset_size = self.get_dataset_size()
        # sampling of datafiles
        self.file_list_random = random.sample(self.file_list, self.nfiles)
        self.nfiles2merge = nfiles_merge                                # number of files to be merged for data subset  
        self.nfiles_merged = int(self.nfiles / self.nfiles2merge)       # number of data subsets
        self.samples_merged = self.get_samples_per_merged_file()
        # list of files can be larger for distributed training, i.e. effective dataset size can be increased
        if self.with_horovod:
            # re-do dataset calculation since file list is potentially larger for distributed training
            self.effective_dataset_size = self.get_dataset_size(random_list=True)
        else:
            self.effective_dataset_size = self.dataset_size
        # handle selected variables
        self.varnames_list = self.get_all_varnames()
        self.predictor_list = predictors
        self.static_predictor_list = static_predictors
        self.predictand_list = predictands
        self.n_predictands, self.n_predictors = len(self.predictand_list), len(self.predictor_list)
        self.all_vars = self.predictor_list + self.predictand_list 
        if self.static_predictor_list is not None:
            self.all_vars = self.static_predictor_list + self.all_vars     # ordering important to ensure that predictors come first (cf. make_tf_dataset_allmem-method)!
            self.n_predictors += len(self.static_predictor_list)      
        # get normalization object
        t0 = timer()
        # check if normalization object is provided
        self.normalization_time = -999.
        if norm_obj is None:
            print("Start computing normalization parameters.")
            self.data_norm = ZScore(norm_dims)  # TO-DO: Allow for arbitrary normalization
            self.norm_params = self.data_norm.get_required_stats(self.ds_all)
            self.normalization_time = timer() - t0
        else:
            self.data_norm = norm_obj
            self.norm_params = norm_obj.norm_stats

        # initialize data loading
        self.data_loaded = [xr.Dataset, xr.Dataset]        # two datasets will be cached
        self.iload_next, self.iuse_next = 0, 0
        self.reading_times = []
        self.ds_proc_size = 0.
        self.data_now = None
        if not nworkers:
            nworkers = min((multiprocessing.cpu_count(), self.nfiles2merge))
        self.pool = ThreadPool(nworkers)

    @property
    def stream_mode(self):
        return self._stream_mode
    
    @stream_mode.setter
    def stream_mode(self, mode):
        known_modes = ["hi_input", "hi_input_named_target", "lo_input"]
        if mode not in known_modes:
            raise ValueError(f"Streaming mode {mode} is not supported. Known modes are {', '.join(known_modes)}")
        
        self._stream_mode = mode
        
    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, datadir):
        if not os.path.isdir(datadir):
            raise NotADirectoryError(f"Parsed data directory '{datadir}' does not exist.")

        self._data_dir = datadir

    @property
    def file_list(self):
        return self._file_list

    @file_list.setter
    def file_list(self, patt):
        patt = patt if patt.endswith(".nc") else f"{patt}.nc"
        files = glob.glob(os.path.join(self.data_dir, patt))

        if not files:
            raise FileNotFoundError(f"Could not find any files with pattern '{patt}' under '{self.data_dir}'.")

        self._file_list = list(
            np.asarray(sorted(files, key=lambda s: int(re.search(r'\d+', os.path.basename(s)).group()))))

    @property
    def seed(self):
        return self._seed 

    @seed.setter 
    def seed(self, seed_int):
        if self.with_horovod and seed_int is None:
            raise ValueError(f"Seed integer must be provided for distitributed training with Horovod,")
        
        # set seed
        random.seed(seed_int)
        self._seed = seed_int

    @property
    def nfiles2merge(self):
        return self._nfiles2merge
    
    @nfiles2merge.setter
    def nfiles2merge(self, n2merge: Union[int, Dict]):

        if isinstance(n2merge, int):
            n = n2merge
        else: 
            n = n2merge[f"#GPUS={hvd.size()}"] if self.with_horovod else n2merge[f"#GPUS=1"]

        # ensure that n is a divisor of the total number of files
        n = find_closest_divisor(self.nfiles, n)

        self._nfiles2merge = n
        # for distributed training, data files must be distributed over workers
        if self.with_horovod:
            if hvd.rank() == 0:
                if n != n2merge:
                    print(f"{n2merge} is not a divisor of the total number of files. Value is changed to {n}")
                print(f"Distributed streaming over {hvd.size()} workers.")

            assert n > hvd.size(), f"Number of files to merge {n} must be larger than number of workers {hvd.size()}."
            self._nfiles2merge = int(n / hvd.size())
            if n % hvd.size() > 0:
                # In case that the modulo is non-zero, nfiles2merge is incremented and the file list is appended
                # so that each work processes the same number of files.
                # Note that duplicated files only occur in the last data subset. 
                # To avoid duplicates in the last subset itself, only files from the preceiding subsets are appended.
                self._nfiles2merge += 1
                nfiles_req = int(hvd.size() * self._nfiles2merge * self.nfiles/n)
                if hvd.rank() == 0: 
                    print(f"Append file list by {nfiles_req - self.nfiles} files to get {nfiles_req} files ({self._nfiles2merge} files per worker).")
                self.file_list_random += random.sample(self.file_list_random[0:self.nfiles-n], nfiles_req - self.nfiles)
                self.nfiles = len(self.file_list_random)
        else:
            if n != n2merge:
                print(f"{n2merge} is not a divisor of the total number of files. Value is changed to {n}")


    @property
    def sample_dim(self):
        return self._sample_dim

    @sample_dim.setter
    def sample_dim(self, sample_dim):
        if not sample_dim in self.ds_all.dims:
            raise KeyError(f"Could not find dimension '{sample_dim}' in data.")

        self._sample_dim = sample_dim

    @property
    def predictor_list(self):
        return self._predictor_list

    @predictor_list.setter
    def predictor_list(self, selected_predictors: List):
        """
        Initalizes predictor list. In case that selected_predictors is set to None, all variables with suffix `_in`
        in their names are selected.
        In case that a list of selected_predictors is parsed, their availability is checked
        :param selected_predictors: list of predictor variables or None
        """
        self._predictor_list = self.check_and_choose_vars(selected_predictors, "_in")
        
    @property
    def static_predictor_list(self):
        return self._static_predictor_list
    
    @static_predictor_list.setter
    def static_predictor_list(self, selected_static_predictors: List):
        if selected_static_predictors is None:
            # if no static, high-res predictors are added, set to None
            self._static_predictor_list = None
        else:
            self._static_predictor_list = self.check_and_choose_vars(selected_static_predictors)

    @property
    def predictand_list(self):
        return self._predictand_list

    @predictand_list.setter
    def predictand_list(self, selected_predictands: List):
        """
        Similar to predictor_list-setter, but does not allow for parsing None.
        """
        assert isinstance(selected_predictands, list), "Selected predictands must be a list of variable names"
        self._predictand_list = self.check_and_choose_vars(selected_predictands, "_tar")

    def __len__(self):
        return self.nsamples

    def getitems(self, indices):
        """
        Return samples from loaded dataset, either as single array (mode: 'hi_input' and 'hi_input_named_target')
        or as tuple of arrays (mode: 'lo_input')
        :param indices: sample indices 
        """
        if self.stream_mode == "lo_input":
            da_now = self.getitems_as_tuple(indices)
        else:
            da_now = self.getitems_as_array(indices)
        
        return da_now
    
    def getitems_as_array(self, indices):
        """
        Retrieves samples from dataset and returns an ordered array for later data handling.
        :param indices: sample indices 
        :return: Ordered array of variables with variables as last dimension. Order is: [static_predictors], predictors, predictands
        """
        da_now = self.data_now.isel({self.sample_dim: indices}).to_array("variables").sel({"variables": self.all_vars})
        
        return da_now.transpose(..., "variables")        
        
    def getitems_as_tuple(self, indices):
        """
        Retrieves samples from dataset and returns an ordered tuple of arrays for later data handling.
        :param indices: sample indices 
        :return: Ordered tuple of arrays with variables as last dimension. Order is: static_predictors, predictors, predictands
        """
        da_in_coa, da_in_static, da_out = self.data_now[self.predictor_list].isel({self.sample_dim: indices}).to_array("variables"), \
                                          self.data_now[self.static_predictor_list].isel({self.sample_dim: indices}).to_array("variables"), \
                                          self.data_now[self.predictand_list].isel({self.sample_dim: indices}).to_array("variables")
        
        da_tuple = (da_in_coa.transpose(..., "variables"), da_in_static.transpose(..., "variables"), da_out.transpose(..., "variables"))
        return da_tuple

    def get_dataset_size(self, random_list: bool = False):
        """
        Sum the size of all dataset files in bytes.
        :param random_list: if True, the size computation is based on randomized list which might be longer for distributed training
        :return: size of dataset files in bytes 
        """
        dataset_size = 0.
        # iterate over file_list_random-attribute since this comprises the actual files that are streamed 
        # incl. duplicates in case of distributed training
        flist = self.file_list_random if random_list else self.file_list 
        for datafile in flist:
            dataset_size += os.path.getsize(datafile)

        return dataset_size

    def get_data_dim(self):
        """
        Retrieve the dimensionality of the data to be handled, i.e. without sample_dim which will be batched in a
        data stream.
        :return: tuple of data dimensions
        """
        # get existing dimension names and remove sample_dim
        dimnames = list(self.ds_all.coords)
        dimnames.remove(self.sample_dim)

        # get the dimensionality of the data of interest
        all_dims = dict(self.ds_all.dims)
        data_dim = itemgetter(*dimnames)(all_dims)

        return data_dim

    def get_samples_per_merged_file(self):
        nsamples_merged = []

        for i in range(self.nfiles_merged):
            file_list_now = self.file_list_random[i * self.nfiles2merge: (i + 1) * self.nfiles2merge]
            ds_now = xr.open_mfdataset(list(file_list_now), decode_cf=False)
            nsamples_merged.append(ds_now.dims[self.sample_dim])  

        return max(nsamples_merged)

    def get_all_varnames(self):
        ds_test = xr.open_dataset(self.file_list[0])
        return list(ds_test.variables)

    def check_and_choose_vars(self, var_list: List[str], suffix: str = "*"):
        """
        Checks list of variables for availability or retrieves all variables named with a given suffix
        (for var_list = None)
        :param var_list: list of predictor variables or None
        :param suffix: optional suffix of variables to selected. Only effective if var_list is None
        :return selected_vars: list of selected variables
        """
        if var_list is None:
            selected_vars = [var for var in self.varnames_list if var.endswith(suffix)]
        else:
            stat_list = [var in self.varnames_list for var in var_list]
            if all(stat_list):
                selected_vars = var_list
            else:
                miss_inds = [i for i, x in enumerate(stat_list) if not x]
                miss_vars = [var_list[i] for i in miss_inds]
                raise ValueError(f"Could not find the following variables in the dataset: {*miss_vars,}")

        return selected_vars

    @staticmethod
    def _process_one_netcdf(fname, data_norm, engine: str = "netcdf4", var_list: List = None, **kwargs):
        with xr.open_dataset(fname, decode_cf=False, engine=engine, **kwargs) as ds_now:
            if var_list: ds_now = ds_now[var_list]
            ds_now = StreamMonthlyNetCDF._preprocess_ds(ds_now, data_norm)
            ds_now = ds_now.load()
            return ds_now

    @staticmethod
    def _preprocess_ds(ds, data_norm):
        ds = data_norm.normalize(ds)
        return ds.astype("float32")

    def _read_mfdataset(self, files, **kwargs):
        # parallel processing of files incl. normalization
        datasets = self.pool.map(partial(self._process_one_netcdf, data_norm=self.data_norm, **kwargs), files)
        ds_all = xr.concat(datasets, dim=self.sample_dim)
        # clean-up
        del datasets
        gc.collect()

        return ds_all

    def read_netcdf(self, set_ind):
        set_ind = tf.keras.backend.get_value(set_ind)
        set_ind = int(str(set_ind).lstrip("b'").rstrip("'"))
        set_ind = int(set_ind % self.nfiles_merged)
        file_list_now = self.file_list_random[set_ind * self.nfiles2merge:(set_ind + 1) * self.nfiles2merge]
        il = int(self.iload_next % 2)
        # read the normalized data into memory
        t0 = timer()
        # Restriction to read dynamic variables is not required currently,
        # since constant data get automatically broadcasted with the _read_mfdataset-method
        data_now = self._read_mfdataset(file_list_now, var_list=self.all_vars).copy()
        nsamples = data_now.sizes[self.sample_dim]

        if nsamples < self.samples_merged:
            t1 = timer()
            add_samples = self.samples_merged - nsamples
            istart = random.randint(0, self.samples_merged - add_samples - 1)
            # slice data from data_now...
            ds_add = data_now.isel({self.sample_dim: slice(istart, istart+add_samples)})
            if ds_add.sizes[self.sample_dim] != add_samples:
                print("WARNING: ds_add contains inconsistent number of samples. Re-try...")
                add_samples = self.samples_merged - nsamples
                istart = random.randint(0, self.samples_merged - add_samples - 1)
                ds_add = data_now.isel({self.sample_dim: slice(istart, istart + add_samples)})
            # ... and modify underlying sample-dimension to allow clean concatenation
            ds_add[self.sample_dim] = data_now[self.sample_dim][-1].values + 1 + np.arange(add_samples)
            ds_add[self.sample_dim] = ds_add[self.sample_dim].assign_attrs(data_now[self.sample_dim].attrs)
            data_now = xr.concat([data_now, ds_add], dim=self.sample_dim)
            print(f"Appending data with {add_samples:d} samples took {timer() - t1:.2f}s" +
                  f"(total #samples: {data_now.sizes[self.sample_dim]})")
            
        # Appending with constant variables is not required since they are read and broadcast to data_now already (see above)
        #if self.const_vars:
        #    ds_const_append = self.ds_const.copy().expand_dims({self.sample_dim: data_now[self.sample_dim]})
        #    data_now = xr.merge([data_now, ds_const_append])

        # write to class attribute
        self.data_loaded[il] = data_now
        # timing
        t_read = timer() - t0
        self.reading_times.append(t_read)
        self.ds_proc_size += data_now.nbytes
        print(f"Dataset #{set_ind:d} ({il+1:d}/2) reading time: {t_read:.2f}s.")
        self.iload_next = il + 1

        return il

    def choose_data(self, _):
        ik = int(self.iuse_next % 2)
        self.data_now = self.data_loaded[ik]
        print(f"Use data subset {ik:d}...")
        self.iuse_next = ik + 1
        return True
