import os, sys, glob
import argparse
from typing import List, Union
from operator import itemgetter
from functools import partial
import re
import gc
import random
from timeit import default_timer as timer
import numpy as np
import xarray as xr
import dask
from multiprocessing import Pool as ThreadPool
import tensorflow as tf
import tensorflow.keras as keras

da_or_ds = Union[xr.DataArray, xr.Dataset]

def main():
    parser = argparse.ArgumentParser("Program that test the MAELSTROM AP5 data pipeline")
    parser.add_argument("--datadir", "-d", dest="datadir", type=str,
                       default="/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_tier2/monthly_files_copy/", 
                       help="Directory where monthly netCDF-files are stored")
    parser.add_argument("--file_pattern", "-f", dest="file_patt", type=str, default="downscaling_tier2_train_*.nc", help="Filename pattern to glob netCDF-files")    
    parser.add_argument("--nfiles_load", "-n", default=30, type=int, dest="nfiles_load",
                        help="Number of netCDF-files to load into memory (2x with prefetching).")
    parser.add_argument("--lprefetch", "-lprefetch", dest="lprefetch", action="store_true",
                       help="Enable prefetching.")
    parser.add_argument("--batch_size", "-b", dest="batch_size", default=192, type=int, 
                        help="Batch size for TF dataset.")
    args = parser.parse_args()
    
    # get data handler
    ds_obj = StreamMonthlyNetCDF(args.datadir, args.file_patt, nfiles_merge=args.nfiles_load,
                                 norm_dims=["time", "rlat", "rlon"])
    
    # set-up TF dataset
    ## map-funcs
    tf_read_nc = lambda ind_set: tf.py_function(ds_obj.read_netcdf, [ind_set], tf.int64)
    tf_choose_data = lambda il: tf.py_function(ds_obj.choose_data, [il], tf.bool)
    tf_getdata = lambda i: tf.numpy_function(ds_obj.getitems, [i], tf.float32)
    tf_split = lambda arr: (arr[..., 0:-ds_obj.n_predictands], arr[..., -ds_obj.n_predictands:])
    
    ## process chain
    tfds = tf.data.Dataset.range(int(ds_obj.nfiles_merged*6*10)).map(tf_read_nc) # 6*10 corresponds to (d_steps + 1)*n_epochs with d_steps=5, n_epochs=10
    if args.lprefetch:
        tfds = tfds.prefetch(1)     # .prefetch(1) ensures that one data subset (=n files) is prefetched
    tfds = tfds.flat_map(lambda x: tf.data.Dataset.from_tensors(x).map(tf_choose_data))
    tfds = tfds.flat_map(
        lambda x: tf.data.Dataset.range(ds_obj.samples_merged).shuffle(ds_obj.samples_merged)
        .batch(args.batch_size, drop_remainder=True).map(tf_getdata))

    tfds = tfds.prefetch(int(ds_obj.samples_merged))
    tfds = tfds.map(tf_split).repeat()
    
    t0 = timer()
    for i, x in enumerate(tfds):
        if i == int(ds_obj.nsamples/args.batch_size) - 1:
            break
            
    print(f"Processing one epoch of data lasted {timer() - t0:.1f} seconds.")
    

class ZScore(object):
    """
    Class for performing zscore-normalization and denormalization.
    Also computes normalization parameters from data if necessary.
    """
    def __init__(self, norm_dims: List):
        self.method = "zscore"
        self.norm_dims = norm_dims
        self.norm_stats = {"mu": None, "sigma": None}

    def get_required_stats(self, data: da_or_ds, **stats):
        """
        Get required parameters for z-score normalization. They are either computed from the data
        or can be parsed as keyword arguments.
        :param data: the data to be (de-)normalized
        :param stats: keyword arguments for mean (mu) and standard deviation (std) used for normalization
        :return (mu, sigma): Parameters for normalization
        """
        mu, std = stats.get("mu", self.norm_stats["mu"]), stats.get("sigma", self.norm_stats["sigma"])

        if mu is None or std is None:
            print("Retrieve mu and sigma from data...")
            mu, std = data.mean(self.norm_dims), data.std(self.norm_dims)
            # The following ensure that both parameters are computed in one graph!
            # This significantly reduces memory footprint as we don't end up having data duplicates
            # in memory due to multiple graphs (and also seem to enfore usage of data chunks as well)
            mu, std = dask.compute(mu, std)
            self.norm_stats = {"mu": mu, "sigma": std}
        # else:
        #    print("Mu and sigma are parsed for (de-)normalization.")

        return mu, std

    def normalize(self, data, **stats):
        """
        Perform z-score normalization on data. 
        Either computes the normalization parameters from the data or applies pre-existing ones.
        :param data: Data array of interest
        :param mu: mean of data for normalization
        :param std: standard deviation of data for normalization
        :return data_norm: normalized data
        """
        mu, std = self.get_required_stats(data, **stats)
        data_norm = (data - mu) / std

        return data_norm

    def denormalize(self, data, **stats):
        """
        Perform z-score denormalization on data.
        :param data: Data array of interest
        :param mu: mean of data for denormalization
        :param std: standard deviation of data for denormalization
        :return data_norm: denormalized data
        """
        if self.norm_stats["mu"] is None or self.norm_stats["std"] is None:
            raise ValueError("Normalization parameters mu and std are unknown.")
        else:
            norm_stats = self.get_required_stats(data, **stats)
        
        
        data_denorm = data * norm_stats["std"] + norm_stats["mu"]

        return data_denorm


class StreamMonthlyNetCDF(object):
    """
    Data handler for monthly netCDF-files which provides methods for setting up 
    a TF dataset that is too large to fit into memory. 
    """    
    def __init__(self, datadir, patt: str, nfiles_merge: int, sample_dim: str = "time", selected_predictors: List = None,
                 selected_predictands: List = None, var_tar2in: str = None, norm_dims: List = None, norm_obj=None):
        """
        Initialize data handler.
        :param datadir: path to directory where netCDF-files are located
        :param patt: file name pattern for globbing
        :param nfiles_merge: number of netCDF-files that get loaded into memory 
        :param sample_dim: dimension from which samples will be drawn
        :param selected_predictors: list of predictor variable names
        :param selected_predictands: list of predictand variable names
        :param var_tar2in: target variable that should be added to input as well (e.g. surface topography)
        :param norm_dims: dimenions over which data will be normalized
        :param norm_obj: normalization object 
        """
        self.data_dir = datadir
        self.file_list = patt
        self.nfiles = len(self.file_list)
        self.file_list_random = random.sample(self.file_list, self.nfiles)
        self.nfiles2merge = nfiles_merge
        self.nfiles_merged = int(self.nfiles / self.nfiles2merge)
        self.samples_merged = self.get_samples_per_merged_file()
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
        self.i_loaded = 0
        self.data_now = None
        self.pool = ThreadPool(10)        # To-Do: remove hard-coded number of threads (-> support contact)

    def __len__(self):
        return self.nsamples

    def getitems(self, indices):
        da_now = self.data_now.isel({self.sample_dim: indices}).to_array("variables")
        if self.var_tar2in is not None:
            da_now = xr.concat([da_now, da_now.sel({"variables": self.var_tar2in})], dim="variables")
        return da_now.transpose(..., "variables")

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
            nsamples_merged.append(ds_now.dims["time"])  # To-Do avoid hard-coding

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
        #n = find_closest_divisor(self.nfiles, n2merge)
        #if n != n2merge:
        #    print(f"{n2merge} is not a divisor of the total number of files. Value is changed to {n}")
        if self.nfiles%n2merge != 0:
            raise ValueError(f"Chosen number of files ({n2merge:d}) to read must be a divisor " +
                             f" of total number of files ({self.nfiles:d}).")
        
        self._nfiles2merge = n2merge

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
        t0 = timer()
        # parallel processing of files incl. normalization
        datasets = self.pool.map(partial(self._process_one_netcdf, data_norm=self.data_norm, **kwargs), files)
        ds_all = xr.concat(datasets, dim=self.sample_dim)
        # clean-up
        del datasets
        gc.collect()
        # timing
        print(f"Reading dataset of {len(files)} files took {timer() - t0:.2f}s.")

        return ds_all

    def read_netcdf(self, set_ind):
        set_ind = tf.keras.backend.get_value(set_ind)
        set_ind = int(str(set_ind).lstrip("b'").rstrip("'"))
        set_ind = int(set_ind%self.nfiles_merged)
        il = int((self.i_loaded + 1)%2)
        file_list_now = self.file_list_random[set_ind * self.nfiles2merge:(set_ind + 1) * self.nfiles2merge]
        # read the normalized data into memory
        # ds_now = xr.open_mfdataset(list(file_list_now), decode_cf=False, data_vars=self.all_vars,
        #                           preprocess=partial(self._preprocess_ds, data_norm=self.data_norm),
        #                           parallel=True).load()
        self.data_loaded[il] = self._read_mfdataset(file_list_now, var_list=self.all_vars).copy()
        nsamples = self.data_loaded[il].dims[self.sample_dim]
        if nsamples < self.samples_merged:
            t0 = timer()
            add_samples = self.samples_merged - nsamples
            add_inds = random.sample(range(nsamples), add_samples)
            ds_add = self.data_loaded[il].isel({self.sample_dim: add_inds})
            ds_add[self.sample_dim] = ds_add[self.sample_dim] + 1.
            self.data_loaded[il] = xr.concat([self.data_loaded[il], ds_add], dim=self.sample_dim)
            print(f"Appending data with {add_samples:d} samples took {timer() - t0:.2f}s.")

        print(f"Currently loaded dataset has {self.data_loaded[il].dims[self.sample_dim]} samples.")

        return il

    def choose_data(self, il):
        self.data_now = self.data_loaded[il]
        self.i_loaded = il
        return True
    
if __name__ == "__main__":
    main()
