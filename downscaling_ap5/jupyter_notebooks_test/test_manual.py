import os, sys
sys.path.append("../handle_data/")
import glob
import gc
import psutil
from multiprocessing import Pool as ThreadPool
import xarray as xr
from functools import partial
from all_normalizations import ZScore
import random
import time
from timeit import default_timer as timer

datadir = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_tier2/monthly_files/"
file_list = glob.glob(os.path.join(datadir, "downscaling_tier2_train*.nc"))


class Test:
    def __init__(self, file_list, data_norm, lparallel):
        self.file_list = file_list
        self.data_norm = data_norm
        self.data = None
        self.samples_merged = 21348
        self.lparallel = lparallel
        if lparallel:
            print("Process will be parallelized.")
            #self.pool = multiprocessing.pool.ThreadPool(multiprocessing.cpu_count())
            self.pool = ThreadPool(10)
        else:
            print("Process will run sequentially.")

    def _preprocess_ds(self, ds):
        ds = self.data_norm.normalize(ds)
        return ds.astype("float32")

    @staticmethod
    def _process_one_netcdf(fname):
        with xr.open_dataset(fname, decode_cf=False) as ds_now:
            #ds_now = self._preprocess_ds(ds_now)
            ds_now = ds_now.load()    
        #    nsamples = ds_now.dims["time"]
            return ds_now
            

    def read_netcdf(self, ind_s, ind_e):
        del self.data
        gc.collect()
        file_list_now = self.file_list[ind_s:ind_e]
        t0 = timer()
        # read the normalized data into memory
        def _process_one_netcdf(fname):
            with xr.open_dataset(fname, decode_cf=False) as ds_now:
                #ds_now = self._preprocess_ds(ds_now)
                ds_now = ds_now.load()    
                nsamples = ds_now.dims["time"]
                return ds_now
            
        if self.lparallel:
            datasets = self.pool.map(self._process_one_netcdf, file_list_now)
        else:
            datasets = [_process_one_netcdf(p) for p in file_list_now]
        self.data = xr.concat(datasets, dim="time")
        print(f"Reading datasets of files {ind_s} to {ind_e} took {timer()-t0:.2f}s")
        print(f"Size of loaded dataset: {self.data.nbytes/(1024**3):.2f} GB")
        del datasets
        gc.collect()
        time.sleep(60)

        return True

data_norm = ZScore(["time", "rlat", "rlon"])
data_norm.read_norm_from_file("./norm_test.json")

test = Test(file_list, data_norm, lparallel=True)

print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
test.read_netcdf(0, 30)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
test.read_netcdf(30, 60)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
test.read_netcdf(60, 90)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
