# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2023-02-07"

import os, glob
from typing import List
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
from other_utils import to_list, find_closest_divisor, free_mem


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

    @staticmethod
    def make_tf_dataset_dyn(datadir: str, file_patt: str, batch_size: int, nepochs: int, nfiles2merge: int,
                            lshuffle: bool = True, named_targets: bool = False, predictands: List = None,
                            predictors: List = None, var_tar2in: str = None, norm_obj=None, norm_dims: List = None,
                            nworkers: int = 10):
        """
        Build TensorFlow dataset by streaming from netCDF using xarray's open_mfdatset-method.
        To fit into memory, only a subset of all netCDF-files is processed at once (nfiles2merge-parameter).
        TO-DO: Add flags for repeat and drop_remainder (cf. make_tf_dataset_allmem-method)
        :param datadir: directory where netCDF-files for TF dataset are strored
        :param file_patt: filename pattern to glob files from datadir
        :param batch_size: desired mini-batch size
        :param nepochs: (effective) number of epochs for training
        :param nfiles2merge: number if files to merge for streaming
        :param lshuffle: boolean to enable sample shuffling
        :param named_targets: boolean if targets will be provided as dictionary with named variables for data stream
        :param predictors: List of selected predictor variables; parse None to use all data
        :param predictands: List of selected predictor variables; parse None to use all data
        :param var_tar2in: name of target variable to be added to input (used e.g. for adding high-resolved topography
                                                                         to the input)
        :param norm_dims: names of dimension over which normalization is applied. Should be None if norm_obj is parsed
        :param norm_obj: normalization instance used to normalize the data.
                         If not passed, the normalization instance is retrieved from the data
        :param nworkers: numbers of workers to read in netCDF-files
        :return: tuple of (normalization object, TensorFlow dataset object)
        """
        assert norm_obj or norm_dims, f"Neither norm_obj nor norm_dims has been provided."

        if norm_obj and norm_dims:
            print("WARNING: norm_obj and norm_dims have been passed. norm_dims will be ignored.")
            norm_dims = None

        if norm_obj: assert isinstance(norm_obj, Normalize), "norm_obj is not an instance of the Normalize-class."

        ds_obj = StreamMonthlyNetCDF(datadir, file_patt, nfiles_merge=nfiles2merge, selected_predictors=predictors,
                                     selected_predictands=predictands, var_tar2in=var_tar2in,
                                     norm_obj=norm_obj, norm_dims=norm_dims, nworkers=nworkers)

        tf_read_nc = lambda ind_set: tf.py_function(ds_obj.read_netcdf, [ind_set], tf.int64)
        tf_choose_data = lambda il: tf.py_function(ds_obj.choose_data, [il], tf.bool)
        tf_getdata = lambda i: tf.numpy_function(ds_obj.getitems, [i], tf.float32)
        if named_targets:
            varnames = ds_obj.predictand_list
            tf_split = lambda arr: (arr[..., 0:-ds_obj.n_predictands],
                                    {var: arr[..., -ds_obj.n_predictands + i] for i, var in enumerate(varnames)})
        else:
            tf_split = lambda arr: (arr[..., 0:-ds_obj.n_predictands], arr[..., -ds_obj.n_predictands:])

        if lshuffle:
            nshuffle = ds_obj.samples_merged
        else:
            nshuffle = 1          # equivalent to no shuffling

        # enable flexibility in factor for range
        n_reads = int(ds_obj.nfiles_merged*nepochs)
        tfds = tf.data.Dataset.range(n_reads).map(tf_read_nc).prefetch(1)
        tfds = tfds.flat_map(lambda x: tf.data.Dataset.from_tensors(x).map(tf_choose_data))
        tfds = tfds.flat_map(
            lambda x: tf.data.Dataset.range(ds_obj.samples_merged).shuffle(nshuffle)
            .batch(batch_size, drop_remainder=True).map(tf_getdata, num_parallel_calls=tf.data.AUTOTUNE))

        tfds = tfds.map(tf_split, num_parallel_calls=tf.data.AUTOTUNE).repeat()

        return ds_obj, tfds

    @staticmethod
    def make_tf_dataset_allmem(da: xr.DataArray, batch_size: int, lshuffle: bool = True, shuffle_samples: int = 20000,
            named_targets: bool = False, var_tar2in: str = None, lrepeat: bool = True, drop_remainder: bool = True, 
            lembed: bool = False) -> tf.data.Dataset:
        """
        Build-up TensorFlow dataset from a generator based on the xarray-data array.
        NOTE: All data is loaded into memory
        :param da: the data-array from which the dataset should be cretaed. Must have dimensions [time, ..., variables].
                   Input variable names must carry the suffix '_in', whereas it must be '_tar' for target variables
        :param batch_size: number of samples per mini-batch
        :param lshuffle: flag if shuffling should be applied to dataset
        :param shuffle_samples: number of samples to load before applying shuffling
        :param named_targets: flag if target of TF dataset should be dictionary with named target variables
        :param var_tar2in: name of target variable to be added to input (used e.g. for adding high-resolved topography
                                                                         to the input)
        :param lrepeat: flag if dataset should be repeated
        :param drop_remainder: flag if samples will be dropped in case batch size is not a divisor of # data samples
        :param lembed: flag to trigger temporal embedding (not implemented yet!)
        """
        da = da.load()
        da_in, da_tar = HandleDataClass.split_in_tar(da)
        if var_tar2in is not None:
            # NOTE: * The order of the following operation must be the same as in StreamMonthlyNetCDF.getitems
            #       * The following operation order must concatenate var_tar2in by da_in to ensure
            #         that the variable appears at first place. This is required to avoid
            #         that var_tar2in becomes a predeictand when slicing takes place in tf_split
            da_in = xr.concat([da_tar.sel({"variables": var_tar2in}), da_in], "variables")

        varnames_tar = da_tar["variables"].values

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

        if named_targets is True:
            gen_now = gen_named
        else:
            gen_now = gen_unnamed

        # create output signatures from first sample
        s0 = next(iter(gen_now(da_in, da_tar)))
        sample_spec_in = tf.TensorSpec(s0[0].shape, dtype=s0[0].dtype)
        if named_targets is True:
            sample_spec_tar = {var: tf.TensorSpec(s0[1][var].shape, dtype=s0[1][var].dtype) for var in varnames_tar}
        else:
            sample_spec_tar = tf.TensorSpec(s0[1].shape, dtype=s0[1].dtype)

        # re-instantiate the generator and build TF dataset
        gen_train = gen_now(da_in, da_tar)

        if lembed is True:
            raise ValueError("Time embedding is not supported yet.")
        else:
            data_iter = tf.data.Dataset.from_generator(lambda: gen_train,
                                                       output_signature=(sample_spec_in, sample_spec_tar))

        # Notes:
        # * cache is reuqired to make repeat work properly on datasets based on generators
        #   (see https://stackoverflow.com/questions/60226022/tf-data-generator-keras-repeat-does-not-work-why)
        # * repeat must be applied after shuffle to get varying mini-batches per epoch
        # * batch-size is increased to allow substepping in train_step
        if lshuffle is True:
            data_iter = data_iter.cache().shuffle(shuffle_samples).batch(batch_size, drop_remainder=drop_remainder)
        else:
            data_iter = data_iter.cache().batch(batch_size, drop_remainder=drop_remainder)

        if lrepeat:
            data_iter = data_iter.repeat()

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
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}' passed.")

    if "*" in fname_suffix:
        ds_filename = fname_suffix
    else:
        ds_filename = os.path.join(datadir, f"{fname_suffix}.nc")

        if not os.path.isfile(ds_filename):
            raise FileNotFoundError(f"Could not find requested dataset file '{ds_filename}'")

    return ds_filename


class StreamMonthlyNetCDF(object):
    def __init__(self, datadir, patt, nfiles_merge: int, sample_dim: str = "time", selected_predictors: List = None,
                 selected_predictands: List = None, var_tar2in: str = None, norm_dims: List = None, norm_obj=None,
                 nworkers: int = 10):
        """
        Class object providing all methods to create a TF dataset that iterates over a set of (monthly) netCDF-files
        rather than loading all into memory. Instead, only a subset of all netCDF-files is loaded into memory.
        Furthermore, the class attributes provide key information on the handled dataset.
        :param datadir: directory where set of netCDF-files are located
        :param patt: filename pattern to allow globbing for netCDF-files
        :param nfiles_merge: number of files that will be loaded into memory (corresponding to one dataset subset)
        :param sample_dim: name of dimension in the data over which sampling should be performed
        :param selected_predictors: list of predictor variable names to be obtained
        :param selected_predictands: list of predictand variables names to be obtained
        :param var_tar2in: predictand (target) variable that can be inputted as well
                          (e.g. static variables known a priori such as the surface topography)
        :param norm_dims: list of dimensions over which data will be normalized
        :param norm_obj: normalization object providing parameters for (de-)normalization
        :param nworkers: number of threads to read the netCDF-files
        """
        self.data_dir = datadir
        self.file_list = patt
        self.nfiles = len(self.file_list)
        self.dataset_size = self.get_dataset_size()
        self.file_list_random = random.sample(self.file_list, self.nfiles)
        self.nfiles2merge = nfiles_merge
        self.nfiles_merged = int(self.nfiles / self.nfiles2merge)
        self.samples_merged = self.get_samples_per_merged_file()
        print(f"Data subsets will comprise {self.samples_merged} samples.")
        self.predictor_list = selected_predictors
        self.predictand_list = selected_predictands
        self.n_predictands, self.n_predictors = len(self.predictand_list), len(self.predictor_list)
        self.all_vars = self.predictor_list + self.predictand_list
        self.ds_all = xr.open_mfdataset(list(self.file_list), decode_cf=False, cache=False)  # , parallel=True)
        self.var_tar2in = var_tar2in
        if self.var_tar2in is not None:
            self.n_predictors += len(to_list(self.var_tar2in))
        self.sample_dim = sample_dim
        self.nsamples = self.ds_all.dims[sample_dim]
        self.data_dim = self.get_data_dim()
        t0 = timer()
        self.normalization_time = -999.
        if norm_obj is None:
            print("Start computing normalization parameters.")
            self.data_norm = ZScore(norm_dims)  # TO-DO: Allow for arbitrary normalization
            self.norm_params = self.data_norm.get_required_stats(self.ds_all)
            self.normalization_time = timer() - t0
        else:
            self.data_norm = norm_obj
            self.norm_params = norm_obj.norm_stats

        self.data_loaded = [xr.Dataset, xr.Dataset]
        self.iload_next, self.iuse_next = 0, 0
        self.reading_times = []
        self.ds_proc_size = 0.
        self.data_now = None
        if not nworkers:
            nworkers = min((multiprocessing.cpu_count(), self.nfiles2merge))
        self.pool = ThreadPool(nworkers)

    def __len__(self):
        return self.nsamples

    def getitems(self, indices):
        da_now = self.data_now.isel({self.sample_dim: indices}).to_array("variables")
        if self.var_tar2in is not None:
            # NOTE: * The order of the following operation must be the same as in make_tf_dataset_allmem
            #       * The following operation order must concatenate var_tar2in by da_in to ensure
            #         that the variable appears at first place. This is required to avoid
            #         that var_tar2in becomes a predeictand when slicing takes place in tf_split
            da_now = xr.concat([da_now.sel({"variables": self.var_tar2in}), da_now], dim="variables")

        return da_now.transpose(..., "variables")

    def get_dataset_size(self):
        dataset_size = 0.
        for datafile in self.file_list:
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
            nsamples_merged.append(ds_now.dims["time"])  # To-Do: avoid hard-coding

        return max(nsamples_merged)

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
    def nfiles2merge(self):
        return self._nfiles2merge

    @nfiles2merge.setter
    def nfiles2merge(self, n2merge):
        n = find_closest_divisor(self.nfiles, n2merge)
        if n != n2merge:
            print(f"{n2merge} is not a divisor of the total number of files. Value is changed to {n}")

        self._nfiles2merge = n

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
        Initalizes predictor list. In case that selected_predictors is set to None, all variables with suffix `_in` in their names are selected.
        In case that a list of selected_predictors is parsed, their availability is checked.
        :param selected_predictors: list of predictor variables or None
        """
        self._predictor_list = self.check_and_choose_vars(selected_predictors, "_in")

    @property
    def predictand_list(self):
        return self._predictand_list

    @predictand_list.setter
    def predictand_list(self, selected_predictands: List):
        self._predictand_list = self.check_and_choose_vars(selected_predictands, "_tar")

    def check_and_choose_vars(self, var_list: List[str], suffix: str = "*"):
        """
        Checks list of variables for availability or retrieves all variables named with a given suffix (for var_list = None)
        :param var_list: list of predictor variables or None
        :param suffix: optional suffix of variables to selected. Only effective if var_list is None.
        :return selected_vars: list of selected variables
        """
        ds_test = xr.open_dataset(self.file_list[0])
        all_vars = list(ds_test.variables)

        if var_list is None:
            selected_vars = [var for var in all_vars if var.endswith(suffix)]
        else:
            stat_list = [var in all_vars for var in var_list]
            if all(stat_list):
                selected_vars = var_list
            else:
                miss_inds = [i for i, x in enumerate(stat_list) if x]
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
        set_ind = int(set_ind%self.nfiles_merged)
        file_list_now = self.file_list_random[set_ind * self.nfiles2merge:(set_ind + 1) * self.nfiles2merge]
        il = int(self.iload_next%2)
        # read the normalized data into memory
        # ds_now = xr.open_mfdataset(list(file_list_now), decode_cf=False, data_vars=self.all_vars,
        #                           preprocess=partial(self._preprocess_ds, data_norm=self.data_norm),
        #                           parallel=True).load()
        t0 = timer()
        data_now = self._read_mfdataset(file_list_now, var_list=self.all_vars).copy()
        nsamples = data_now.dims[self.sample_dim]

        if nsamples < self.samples_merged:
            t1 = timer()
            add_samples = self.samples_merged - nsamples
            istart = random.randint(0, self.samples_merged - add_samples - 1)
            # slice data from data_now...
            ds_add = data_now.isel({self.sample_dim: slice(istart, istart+add_samples)})
            if ds_add.dims[self.sample_dim] != add_samples:
                print("WARNING: ds_add contains inconsistent number of samples. Re-try...")
                add_samples = self.samples_merged - nsamples
                istart = random.randint(0, self.samples_merged - add_samples - 1)
                ds_add = data_now.isel({self.sample_dim: slice(istart, istart + add_samples)})
            # ... and modify underlying sample-dimension to allow clean concatenation
            ds_add[self.sample_dim] = data_now[self.sample_dim][-1].values + 1 + np.arange(add_samples)
            ds_add[self.sample_dim] = ds_add[self.sample_dim].assign_attrs(data_now[self.sample_dim].attrs)
            data_now = xr.concat([data_now, ds_add], dim=self.sample_dim)
            print(f"Appending data with {add_samples:d} samples took {timer() - t1:.2f}s" +
                  f"(total #samples: {data_now.dims[self.sample_dim]})")
            # free memory
            free_mem([ds_add, add_samples, istart])

        # free memory
        free_mem([nsamples])
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
