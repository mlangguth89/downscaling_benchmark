#!/usr/bin/env python
# coding: utf-8

import os, glob
import argparse
from timeit import default_timer as timer
import json as js
import pandas as pd
import numpy as np
import xarray as xr
import tensorflow as tf
import multiprocessing


class StreamMonthlyNetCDF(object):
    def __init__(self, datadir, patt, workers=4, sample_dim: str = "time"):
        self.data_dir = datadir
        self.file_list = patt
        self.ds = xr.open_mfdataset(self.file_list)
        self.sample_dim = sample_dim
        self.times = self.ds[sample_dim].load()

        self.pool = multiprocessing.pool.ThreadPool(workers)

    def __len__(self):
        return self.ds.dims[self.sample_dim]

    def __getitem__(self, i):
        data = self.index_to_sample(i)
        return data
    
    def getitems(self, indices):
        print(indices)
        return np.array(self.pool.map(self.__getitem__, indices))
    
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
            
        self._file_list = sorted(files)
        
    @property
    def sample_dim(self):
        return self._sample_dim 
    
    @sample_dim.setter
    def sample_dim(self, sample_dim):
        if not sample_dim in self.ds.dims:
            raise KeyError(f"Could not find dimension '{sample_dim}' in data.")
            
        self._sample_dim = sample_dim 
        
    def index_to_sample(self, index):
        curr_time = pd.to_datetime(self.times[index].values)
        
        fname = [s for s in self.file_list if curr_time.strftime("%Y-%m") in s]
        if not fname:
            raise FileNotFoundError(f"Could not find a file matching requested date {curr_time}")
        elif len(fname) > 1:
            raise ValueError(f"Files found for requested date {curr_time} is not unique.")
         
        ds = xr.open_dataset(fname[0])
        return ds.sel({self.sample_dim: curr_time}).to_array()


def main(parser_args):
    # path to netCDF-files
    datadir = parser_args.data_dir
    patt = parser_args.fname_pattern
    ds_name = parser_args.dataset_name
    # other parameters
    batch_size = parser_args.batch_size
    ntests = parser_args.ntests
    # JSON-file to track results
    js_file = os.path.join("./", f"performance_test_{ds_name}.json")

    # get number of available (virtual) CPUs
    max_workers = multiprocessing.cpu_count()
    workers = min(batch_size, max_workers)
    print(f"Number of available (virtual) CPUs: {max_workers:d}. " +
          f"Batch generation will be distributed over {workers:d} workers.")

    # instantiate an example monthly data stream
    all_data = StreamMonthlyNetCDF(datadir, patt, workers=workers)

    # check timestamps and available number of samples
    nsamples = len(all_data)
    print(f"Available samples in dataset: {nsamples:d}.")
    print(all_data.times)

    tf_fun = lambda i: tf.numpy_function(all_data.getitems, [i], tf.float64)

    nworkers_list = [int(workers/n) for n in range(1, 5)]

    log_dict = {}

    for nworkers in nworkers_list:
        elapsed_times = []
        for j in range(ntests):
            time0 = timer()

            # set-up TF sataset
            ds = tf.data.Dataset.range(len(all_data)).shuffle(buffer_size=20000).batch(nworkers) \
                                .map(tf_fun).unbatch().batch(batch_size)

            for k, x in enumerate(ds):
                if k == 0 and j == 0:
                    print(f"Sanity check for test with {nworkers}:")
                    print(tf.shape(x))
                else:
                    pass

            elapsed_times.append(timer() - time0)

        print(f"Averaged elapsed time with {nworkers:d}: {np.mean(elapsed_times):.1f})")
        log_dict[f"elapsed_times_{nworkers:d}"] = elapsed_times

        with open(js_file, "w") as jsf:
            js.dump(log_dict, jsf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", "-data_dir", dest="data_dir", type=str, required=True,
                        help="Directory where input netCDF-files are stored.")
    parser.add_argument("--filename_pattern", "-fname_patt", dest="fname_pattern", type=str, required=True,
                        help="File pattern of netCDF-files to include into TF dataset.")
    parser.add_argument("--dataset_name", "-ds_name", dest="dataset_name", type=str, required=True,
                        help="Name of dataset used for testing data stream performance.")
    parser.add_argument("--mini_batch_size", "-batch_size", dest="batch_size", type=int, default=32,
                        help="Number of samples for mini-batch when building TF dataset.")
    parser.add_argument("--num_tests", "-ntests", dest="ntests", type=int, default=5,
                        help="Number of tests to calculate performance.")

    args = parser.parse_args()
    main(args)
