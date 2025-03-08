# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

"""
Methods to handle data for the neural networks.

Provides:
    - get_dataset_filenames: get files for known datasets  
    - prepare_dataset: set up TF data pipeline for model training
    - make_tf_dataset_dyn: Dynmaical data streaming from bunch of netCDF-files
    - make_tf_dataset_allmem: Data streaming from a single netCDF-file that fits into memory
    - reshape_ds: reshape dataset 
    - split_in_tar: split dataset between input and target data:
    - StreamMonthlyNetCDF: class for dynamical data streaming from netCDF-files

To-Dos:
    - Shuffle indices for sharding in make_tf_dataset_all-method
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2025-03-08"

import os, glob
from typing import List, Tuple, Union, Dict
from pathlib import Path
import re
from operator import itemgetter
from functools import partial
import gc
from timeit import default_timer as timer
import random
import numpy as np
import xarray as xr
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
except:
    print("Horovod is not installed. Distributed training is not supported.")
    pass
import multiprocessing
try:
    from multiprocessing import Pool as ThreadPool
except:
    from multiprocessing.pool import ThreadPool
from all_normalizations import GeneralNormalizer
from other_utils import to_list, find_closest_divisor, finditem



def get_dataset_filename(datadir: str, dataset_name: str, subset: str, laugmented: bool = False):
    """
    Get files in directory corresponding to known known dataset (e.g. "benchmark_t2m") and its subset (e.g. "train", "val", "test").
    :param datadir: data directory under which files are expected
    :param dataset_name: known dataset name. Valid choices are: 'tier1', 'tier2', 'atmorep', 'benchmark_t2m', 'benchmark_wind'
    :param subset: dataset subset for training ML models. Valid choices are 'train', 'val', 'test'
    :param laugmented: boolean if augmented dataset should be used (if available)
    :return: filename or list of filenames corresponding to desired dataset and its subset
    """

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
    elif dataset_name in ["benchmark_t2m", "benchmark_wind"]:
        fname_suffix = f"{fname_suffix}_{dataset_name}_{subset}"
        if subset == "train":
            fname_suffix = f"{fname_suffix}*"
        if laugmented: raise ValueError(f"No augmented dataset available for {dataset_name}.")
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
    :param ds_dict: dictionary of dataset configuration.
                    Must comprise: 
                    - predictors, predictands: Dictionaries of predictors and predictands with variable names as keys,
                                               and respective normalization-method as values
                    - num_files: Number of files num_files to load into memory if data is distributed over several netCDF-files
                    - norm_dims: List of dimensions over which normalization parameters are derived
                    Optional:
                    - static_predictors: Same as predictors, but for temporally invariant variables 
                    - varname_z: Dictionary for topography data in the form {variable name: normalization method}
                    - var_tar2in: Dictionary for (static) target data that can (additionally) be used as predictor (deprecated!)                    
    :param hparams_dict: dictionary of model hyperparameters
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
    main_process = True
    if with_horovod:
        main_process = hvd.rank() == 0

    # Check parsed dataset mode
    allowed_modes = ["train", "val", "test"]
    assert mode in allowed_modes, f"{mode} is not a valid mode. Allowed modes are {*allowed_modes,}" 

    # Check if normalization object is provided (mandatory for validation and test mode)
    if mode != "train" and not norm_obj:
        raise ValueError(f"Normalization object norm_obj must be provided for mode {mode}.")
    else:
        assert norm_obj or norm_dims, f"Neither norm_obj nor norm_dims has been provided."

    if norm_obj and norm_dims:
        if main_process: print("WARNING: norm_obj and norm_dims have been passed. norm_dims will be ignored.")
        norm_dims = None

    if norm_obj: 
        assert isinstance(norm_obj, GeneralNormalizer), "norm_obj is not an instance of the GeneralNormalizer-class."

    # Handle predictands
    varnames_tar_all = ds_dict["predictands"].copy()
    # Apppend predictands in case of separate z_branch in model configuration 
    if finditem(hparams_dict, "z_branch", False):
        varnames_tar_all = {**varnames_tar_all, **ds_dict["varname_z"]}

    # Handle dynamic and static predictors
    predictors = ds_dict["predictors"].copy()           # predictors-dictionary must be provided

    # Backward compatibility for deprecated keys var_tar2in and named_targets in ds_dict and hparams_dict, respectively
    if "var_tar2in" in ds_dict:
        static_predictors = {**ds_dict.get['var_tar2in'], **ds_dict.get("static_predictors", {})}
        if main_process: print("Warning: The usage of 'var_tar2in' is deprecated. Use 'static_predictors' instead. \n"
                               +f"Dictionary of updated static predictors: {','.join(static_predictors)}")
    else:
        static_predictors = ds_dict.get("static_predictors", None).copy()

    # Get filenames for training dataset...    
    fname_or_pattern = get_dataset_filename(datadir, dataset_name, mode)
    
    # ... and set streaming mode
    if hparams_dict.get("named_targets", False):
        stream_mode = "hi_input_named_target"
    else:
        stream_mode = hparams_dict.get("stream_mode", "hi_input")

    if not "stream_mode" in hparams_dict and main_process:
        print(f"Warning: stream_mode not provided in hparams_dict. Autmotically set to '{stream_mode}'.")
    elif main_process:
        print(f"Selected stream mode for {mode} dataset: {stream_mode}")
    else:
        pass

    # Get (effective) batch size and desired number of epochs from model configuration 

    # Note: bs_train is introduced to allow substepping in the training loop, e.g. for WGAN where n optimization steps
    # are applied to train the critic, before the generator is trained once.
    # The validation and test dataset however do not perform substeeping and thus don't require an increased mini-batch size.
    if mode == "train":
        bs_train = ds_dict["batch_size"] * (hparams_dict["d_steps"] + 1) if "d_steps" in hparams_dict else ds_dict["batch_size"]
        nepochs = hparams_dict["nepochs"] * (hparams_dict["d_steps"] + 1) if "d_steps" in hparams_dict else hparams_dict["nepochs"]
    else:
        bs_train = ds_dict["batch_size"]
        nepochs = hparams_dict["nepochs"]

    # Get TensorFlow datasets
    if "*" in fname_or_pattern:                                             # do not load all data into memory
        ds_obj = StreamMonthlyNetCDF(stream_mode, datadir, fname_or_pattern, nfiles_merge=ds_dict["num_files"],
                                     predictands=varnames_tar_all, predictors=predictors,
                                     static_predictors=static_predictors, sample_dim=ds_dict.get("sample_dim", "time"),
                                     norm_obj=norm_obj, norm_dims=norm_dims, with_horovod=with_horovod, seed=seed, nworkers=nworkers)
        
        if shuffle:
            nshuffle = ds_obj.samples_merged
        else:
            nshuffle = 1          # equivalent to no shuffling

        tfds = make_tf_dataset_dyn(ds_obj, bs_train, nepochs, nshuffle=nshuffle, lrepeat=lrepeat, drop_remainder=drop_remainder)

        # get input shape depending on streaming mode and processed data
        if stream_mode == "lo_input":
            shape_in = [*ds_obj.data_xy_dim["input"], len(ds_obj.predictor_list), len(ds_obj.static_predictor_list)]
        else:
            shape_in = [*ds_obj.data_xy_dim["input"], len(ds_obj.predictor_list + ds_obj.static_predictor_list)]
        
        
        tfds_info = {"nsamples": ds_obj.nsamples, "data_norm": ds_obj.data_norm, "shape_in": tuple(shape_in),
                     "dataset_size": ds_obj.dataset_size, "ds_obj": ds_obj, "all_predictands": varnames_tar_all, "file": ds_obj.file_list,
                     "effective_dataset_size": ds_obj.effective_dataset_size, "predictors": predictors, 
                     "static_predictors": static_predictors, "stream_mode": stream_mode}
    else:                                                                   # load all data into memory
        ds = xr.open_dataset(fname_or_pattern)

        vars2norm = {**predictors, **static_predictors, **varnames_tar_all}
        if not norm_obj:
            # norm_obj must be freshly instantiated (triggering later parameter retrieval)
            norm_obj = GeneralNormalizer(ds_dict["norm_dims"], vars2norm)

        ds = norm_obj.normalize(ds)

        nsamples = len(ds["time"])

        # create TensorFlow dataset
        tfds = make_tf_dataset_allmem(stream_mode, ds, bs_train, varnames_tar_all, predictors=predictors, 
                                      static_predictors=static_predictors, lrepeat=lrepeat, drop_remainder=drop_remainder,
                                      lshuffle=shuffle, with_horovod=with_horovod)
        
        # get input shape depending on streaming mode and processed data
        if stream_mode == "lo_input":
            shape_in = tfds.element_spec[0]["lo_res_inputs"].shape[1:].as_list() + [len(static_predictors)]
        else:
            shape_in = tfds.element_spec[0].shape[1:].as_list()
    
        # provide dict for later use
        tfds_info = {"nsamples": nsamples, "data_norm": norm_obj, "shape_in": tuple(shape_in),
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
            return ({"lo_res_inputs": darr_in, "hi_res_inputs": darr_stat}, {"output": darr_out})
                                         
        tf_getdata = lambda i: tf.numpy_function(ds_obj.getitems, [i], [tf.float32, tf.float32, tf.float32])
        tf_split = lambda arr_in, arr_stat, arr_out: make_dict(arr_in, arr_stat, arr_out)

    # enable flexibility in factor for range
    n_reads = int(ds_obj.nds*nepochs)
    if ds_obj.with_horovod:
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
    main_process = True
    if with_horovod:
        main_process = hvd.rank() == 0

    # add static predictors to predictors-list unless lo_input-streaming mode is chosen
    if stream_mode == "lo_input":
        assert static_predictors is not None, "Provide high-resolved static input predictors for stream_mode 'lo_input'"
    else:
        if static_predictors:
            predictors = {**static_predictors, **predictors}
            static_predictors = None
            if main_process: print(f"Static predictors added to predictors for data pipeline mode '{stream_mode}'")

    if with_horovod:
        ntimes = len(ds["time"])
        # To-Do: Indices should be shuffled to avoid daytime dependencies
        inds = list(range(hvd.rank(), ntimes, hvd.size()))
        ds = ds.isel({"time": inds})

    # add time dimension to constant variables
    for var in ds.data_vars:
        if "time" not in ds[var].dims:
            ds[var] = ds[var].expand_dims({"time": ds["time"]}, axis=0)

    ds_in, ds_tar, ds_stat = split_in_tar(ds, predictands=predictands, predictors=predictors,
                                                          static_predictors=static_predictors)    
    
    ds_list = [ds_in, ds_tar, ds_stat] if static_predictors is not None else [ds_in, ds_tar]

    # convert dataset to data arrays and load into memory
    da_list = [reshape_ds(ds).astype("float32", copy=True) for ds in ds_list]

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
            yield tuple(({"lo_res_inputs": darr_in.isel({"time": t}).values,
                          "hi_res_inputs": darr_stat.isel({"time": t}).values},
                         {"output": darr_tar.isel({"time": t}).values}))

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

def reshape_ds(ds):
    """
    Convert a xarray dataset to a data-array where the variables will constitute the last dimension (channel last)
    :param ds: the xarray dataset with dimensions (dims)
    :return da: the data-array with dimensions (dims, variables)
    """
    da = ds.to_array(dim="variables")
    da = da.transpose(..., "variables")
    return da


def split_in_tar(ds: xr.Dataset, predictands: List = None, predictors: List = None, static_predictors: List = None) -> Tuple[xr.Dataset, xr.Dataset]:
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
        raise ValueError(f"Automatic detection of predictors is not supported anymore")
    else:
        assert all(
            [predictor in varnames for predictor in predictors.keys()]
        ), f"At least one predictor is not a data variable. Available variables are {*varnames,}"
        invars = list(predictors.keys())

    if predictands is None:
        raise ValueError(f"Automatic detection of predictands is not supported anymore")
    else:
        assert all(
            [predictand in varnames for predictand in predictands.keys()]
        ), f"At least one predictor is not a data variable. Available variables are {*varnames,}"
        tarvars = list(predictands.keys())

    ds_in, ds_tar = ds[invars], ds[tarvars]

    if static_predictors is None:
        ds_stat = None
    else:
        assert all([static_predictor in varnames for static_predictor in static_predictors.keys()]), \
                f"At least one static high-res predictor is not a data variable. Available variables are {*varnames,}"
        statvars = list(static_predictors.keys())

        ds_stat = ds[statvars]

    return ds_in, ds_tar, ds_stat

class StreamMonthlyNetCDF(object):
    def __init__(self, mode: str, datadir: Path, patt: str, nfiles_merge: Union[int, Dict], predictands: Dict,
                 predictors: Dict, static_predictors: Dict = None, sample_dim: str = "time", norm_dims: List = None,
                 norm_obj=None ,with_horovod: bool = False, seed: int = None, nworkers: int = 10, max_tries: int = 3):
        """
        Class object providing all methods to create a TF dataset that iterates over a set of (monthly) netCDF-files
        rather than loading all into memory. Instead, only a subset of all netCDF-files is loaded into memory.
        Furthermore, the class attributes provide key information on the handled dataset
        :param datadir: directory where set of netCDF-files are located
        :param patt: filename pattern to allow globbing for netCDF-files
        :param nfiles_merge: number of files per data subset loaded into memory (can be an integer or a dictionary like {"#GPUS=1": 33})
        :param predictands: Dictionary of predictands with variable names as keys and normalization method as values
        :param predictors: Dictionary of predictors with variable names as keys and normalization method as values
        :param static_predictors: Dictionary of static predictors with variable names as keys and normalization method as values
        :param sample_dim: name of dimension in the data over which sampling should be performed
        :param norm_dims: list of dimensions over which data will be normalized
        :param norm_obj: normalization object providing parameters for (de-)normalization
        :param with_horovod: flag to trigger horovod-based distributed dataset creation
        :param seed: seed for random sampling of netCDF-files
        :param nworkers: number of threads to read the netCDF-files
        :param max_tries: maximum number of tries to append data to fixed number of samples in read_netcdf-method
        """
        self.with_horovod = with_horovod
        self.main_process = True
        if self.with_horovod:
            self.main_process = hvd.rank() == 0
        self.seed = seed
        self.stream_mode = mode
        self.data_dir = datadir
        # get file list and number of files to be merged for data subse
        self.file_list = patt
        self.nfiles = len(self.file_list)
        # get relevant data dimensions
        ds_all = xr.open_mfdataset(list(self.file_list), decode_cf=False, cache=False)  # , parallel=True)
        self.all_dims = ds_all.dims
        self.sample_dim = sample_dim
        self.nsamples = ds_all.dims[sample_dim]
        self.dataset_size = self.get_dataset_size()
        # sampling of datafiles
        self.file_list_random = random.sample(self.file_list, self.nfiles)
        self.nfiles2merge = nfiles_merge                                # number of files to be merged for data subset  
        self.nds = int(self.nfiles / self.nfiles2merge)                 # number of data subsets
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

        self.data_xy_dim = self.get_nxy_dim(ds_all) 
        # sanity check on shapes of predictors, predictands and static predictors depending on stream_mode
        self.check_data_shapes()    
        # get normalization object
        t0 = timer()
        # check if normalization object is provided
        self.normalization_time = -999.
        if not norm_obj:
            vars2norm = {**predictors, **static_predictors, **predictands}
            # norm_obj must be freshly instantiated (triggering later parameter retrieval)
            self.data_norm = GeneralNormalizer(norm_dims, vars2norm)  # TO-DO: Allow for arbitrary normalization
            _ = self.data_norm.get_stats_from_data(ds_all)
            self.normalization_time = timer() - t0
        else:
            if not isinstance(norm_obj, GeneralNormalizer):
                raise ValueError("norm_obj is not an instance of the GeneralNormalizer-class.")
            self.data_norm = norm_obj

        self.max_tries = max_tries

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
            if self.with_horovod: 
                n = n2merge[f"#GPUS={hvd.size()}"]
            else:
                n = n2merge[f"#GPUS=1"]

        # ensure that n is a divisor of the total number of files
        n = find_closest_divisor(self.nfiles, n)

        self._nfiles2merge = n
        # for distributed training, data files must be distributed over workers
        if self.with_horovod:
            if self.main_process: print(f"Distributed streaming over {hvd.size()} workers.")

            #assert n > hvd.size(), f"Number of files to merge {n} must be larger than number of workers {hvd.size()}."
            #self._nfiles2merge = int(n / hvd.size())
            if n % hvd.size() > 0:
                # In case that the modulo is non-zero, nfiles2merge is incremented and the file list is appended
                # so that each work processes the same number of files.
                # Note that duplicated files only occur in the last data subset. 
                # To avoid duplicates in the last subset itself, only files from the preceiding subsets are appended.
                self._nfiles2merge += 1
                nfiles_req = int(hvd.size() * self._nfiles2merge * self.nfiles/n)
                if self.main_process: 
                    print(f"Append file list by {nfiles_req - self.nfiles} files to get {nfiles_req} files ({self._nfiles2merge} files per worker).")
                self.file_list_random += random.sample(self.file_list_random[0:self.nfiles-n], nfiles_req - self.nfiles)
                self.nfiles = len(self.file_list_random)
        else:
            if n != n2merge and self.main_process:
                print(f"{n2merge} is not a divisor of the total number of files. Value is changed to {n}")


    @property
    def sample_dim(self):
        return self._sample_dim

    @sample_dim.setter
    def sample_dim(self, sample_dim):
        if not sample_dim in list(self.all_dims):
            raise KeyError(f"Could not find dimension '{sample_dim}' in data.")

        self._sample_dim = sample_dim

    @property
    def predictor_list(self):
        return self._predictor_list

    @predictor_list.setter
    def predictor_list(self, selected_predictors: Dict):
        """
        Initalizes predictor list. In case that selected_predictors is set to None, all variables with suffix `_in`
        in their names are selected.
        In case that a list of selected_predictors is parsed, their availability is checked
        :param selected_predictors: list of predictor variables or None
        """
        assert isinstance(selected_predictors, dict), \
            "Selected predictors must be a dictionary of variable names as keys and normalization method as values"
        self._predictor_list = self.check_and_choose_vars(selected_predictors.keys())
        
    @property
    def static_predictor_list(self):
        return self._static_predictor_list
    
    @static_predictor_list.setter
    def static_predictor_list(self, selected_static_predictors: Dict):
        if selected_static_predictors == {}:
            # if no static, high-res predictors are added, set to None
            self._static_predictor_list = None
        else:
            assert isinstance(selected_static_predictors, dict), \
                "Selected static predictors must be a dictionary of variable names as keys and normalization method as values"
            self._static_predictor_list = self.check_and_choose_vars(selected_static_predictors.keys())

    @property
    def predictand_list(self):
        return self._predictand_list

    @predictand_list.setter
    def predictand_list(self, selected_predictands: Dict):
        """
        Similar to predictor_list-setter, but does not allow for parsing None.
        """
        assert isinstance(selected_predictands, dict), \
            "Selected predictands must be a dictionary of variable names as keys and normalization method as values"
        self._predictand_list = self.check_and_choose_vars(selected_predictands.keys())

    def __len__(self):
        return self.nsamples

    def getitems(self, indices):
        """
        Return samples from loaded dataset, either as single array (mode: 'hi_input' and 'hi_input_named_target')
        or as tuple of arrays (mode: 'lo_input')
        :param indices: sample indices 
        """
        indices = np.array(indices) % self.data_now.sizes[self.sample_dim]

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

    def get_nxy_dim(self, ds):
        """
        Retrieve the spatial dimensionality of the input and target data.
        :return: Dictionary of spatial dimensions of the predictands, predictors and, if available, static predictors
        """
        data_dims_keys = ["output", "input",]
        infer_vars = [self.predictand_list[0], self.predictor_list[0]]
        
        if self.static_predictor_list is not None:
            data_dims_keys += ["input_static"]
            infer_vars += [self.static_predictor_list[0]]

        dim_dict = {}
        for key, var in zip(data_dims_keys, infer_vars):
            dimnames = list(ds[var].dims)
            dimnames.remove(self.sample_dim)

            data_dim = itemgetter(*dimnames)(self.all_dims)
            dim_dict[key] = data_dim
            
        return dim_dict

    def get_samples_per_merged_file(self):
        nsamples_merged = []

        for i in range(self.nds):
            file_list_now = self.file_list_random[i * self.nfiles2merge: (i + 1) * self.nfiles2merge]
            ds_now = xr.open_mfdataset(list(file_list_now), decode_cf=False)
            nsamples_merged.append(ds_now.dims[self.sample_dim])  

        return max(nsamples_merged)

    def get_all_varnames(self):
        ds_test = xr.open_dataset(self.file_list[0])
        return list(ds_test.variables)

    def check_and_choose_vars(self, var_list: List[str]):
        """
        Checks list of variables for availability 
        :param var_list: list of variables
        :return selected_vars: sanity checked list of variables
        """
        if var_list is None:
            raise ValueError(f"Automatic detection of predictors is not supported anymore")
        else:
            stat_list = [var in self.varnames_list for var in var_list]
            if all(stat_list):
                selected_vars = list(var_list)
            else:
                miss_inds = [i for i, x in enumerate(stat_list) if not x]
                miss_vars = [list(var_list)[i] for i in miss_inds]
                raise ValueError(f"Could not find the following variables in the dataset: {*miss_vars,}")

        return selected_vars
    
    def check_data_shapes(self):
        """
        Check if the spatial data dimensions are consistent w.r.t. to the streaming mode.
        """
        nxy_in_str, nxy_stat_str = [str(n) for n in self.data_xy_dim['input']], [str(n) for n in self.data_xy_dim['input_static']]
        nxy_out_str = [str(n) for n in self.data_xy_dim['output']]
        
        if self.stream_mode == "lo_input":
            assert self.data_xy_dim["input"] != self.data_xy_dim["input_static"], f"Predictors and static predictors must have different spatial shapes. " + \
                                                                                      f"predictors: [{','.join(nxy_in_str)}], " + \
                                                                                      f"static_predictors: [{','.join(nxy_stat_str)}]"
            assert self.data_xy_dim["output"] == self.data_xy_dim["input_static"], f"Predictands and static predictors must have the same spatial shapes. " + \
                                                                                      f"predictands: [{','.join(nxy_out_str)}], " + \
                                                                                      f"static_predictors: [{','.join(nxy_stat_str)}]"
        else:
            mess = f"The spatial shapes of all variables must be the same. predictands: [{','.join(nxy_out_str)}], predictors: [{','.join(nxy_in_str)}]"
            if self.static_predictor_list is not None:
                mess += f" static_predictors: [{','.join(nxy_stat_str)}]"
            assert self.data_xy_dim["input"] == self.data_xy_dim["input_static"] == self.data_xy_dim["output"], mess        

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
        # get start index for data subset
        set_ind = tf.keras.backend.get_value(set_ind)
        set_ind = int(str(set_ind).lstrip("b'").rstrip("'"))
        set_ind = int(set_ind % self.nds)
        file_list_now = self.file_list_random[set_ind * self.nfiles2merge:(set_ind + 1) * self.nfiles2merge]
        if self.main_process: print(f"ifiles: {set_ind * self.nfiles2merge} -> {(set_ind + 1) * self.nfiles2merge}")
        
        il = int(self.iload_next % 2)
        # read the normalized data into memory
        t0 = timer()
        # Restriction to read dynamic variables is not required currently,
        # since constant data get automatically broadcasted with the _read_mfdataset-method
        data_now = self._read_mfdataset(file_list_now, var_list=self.all_vars).copy()
        # appending to fixed number of samples is not performed anymore
        # Instead, indices are wrapped to avoid index out-of-range errors (cf. getitems-method)
        #nsamples = data_now.sizes[self.sample_dim]
        #add_samples = self.samples_merged - nsamples
       
        # write to class attribute
        self.data_loaded[il] = data_now
        # timing
        t_read = timer() - t0
        self.reading_times.append(t_read)
        self.ds_proc_size += data_now.nbytes
        if self.main_process: print(f"Dataset #{set_ind:d} ({il+1:d}/2) reading time: {t_read:.2f}s.")
        self.iload_next = il + 1

        return il

    def choose_data(self, _):
        ik = int(self.iuse_next % 2)
        self.data_now = self.data_loaded[ik]
        if self.main_process: print(f"Use data subset {ik:d}...")
        self.iuse_next = ik + 1
        return True
