import os, sys
sys.path.append("../handle_data/")
import argparse
import glob
import gc
import psutil
import multiprocessing
from multiprocessing import Pool as ThreadPool
import xarray as xr
from functools import partial
from all_normalizations import ZScore
import time
from timeit import default_timer as timer

datadir = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_tier2/monthly_files/"
file_list = glob.glob(os.path.join(datadir, "downscaling_tier2_train*.nc"))
nfiles=30

class Test:
    def __init__(self, file_list, data_norm, lparallel):
        self.file_list = file_list
        self.data_norm = data_norm
        self.data = None
        self.lparallel = lparallel
        if lparallel:
            print("Process will be parallelized.")
            # NOTE on assumed bug:
            # Increasing thread pool to size of available CPUs results into a serious memory leak
            self.pool = ThreadPool(10)
            #self.pool = ThreadPool(multiprocessing.cpu_count())
        else:
            print("Process will run sequentially.")

    @staticmethod
    def _preprocess_ds(ds, data_norm):
        ds = data_norm.normalize(ds)
        return ds.astype("float32")

    @staticmethod
    def _process_one_netcdf(fname, data_norm):
        with xr.open_dataset(fname, decode_cf=False) as ds_now:
            ds_now = Test._preprocess_ds(ds_now, data_norm)
            ds_now = ds_now.load()
            return ds_now
            

    def read_netcdf(self, ind_s, ind_e):
        file_list_now = self.file_list[ind_s:ind_e]
        t0 = timer()           
        if self.lparallel:
            datasets = self.pool.map(partial(self._process_one_netcdf, data_norm=self.data_norm), file_list_now)
            #datasets = self.pool.map(self._process_one_netcdf, file_list_now)
        else:
            datasets = [self._process_one_netcdf(p, self.data_norm) for p in file_list_now]
        self.data = xr.concat(datasets, dim="time")
        nsamples = self.data.dims["time"]
        print(f"Reading datasets of files {ind_s} to {ind_e} took {timer()-t0:.2f}s")
        print(f"Size of loaded dataset: {self.data.nbytes/(1024**3):.2f} GB")
        del datasets
        gc.collect()
        time.sleep(60)

        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lparallel", "-parallel", dest="parallel", default=False, action="store_true", help="Flag to parallelize reading data with multiprocessing.")

    args = parser.parse_args()

    data_norm = ZScore(["time", "rlat", "rlon"])
    data_norm.read_norm_from_file("./norm_test.json")

    print('Start - RAM Used (GB):', psutil.virtual_memory()[3]/1024**3)
    test = Test(file_list, data_norm, lparallel=args.parallel)
    print('After init - RAM Used (GB):', psutil.virtual_memory()[3]/1024**3)

    for i in range(3):
        test.read_netcdf(i*nfiles, (i+1)*nfiles )
        print(f"Read step {i} - RAM Used (GB): {psutil.virtual_memory()[3]/1024**3:.2f}")
