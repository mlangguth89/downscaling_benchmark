import os, sys
sys.path.append("../handle_data/")
import glob
import xarray as xr
from functools import partial
from all_normalizations import ZScore
import random
import time
from timeit import default_timer as timer

datadir = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_tier2/monthly_files/"
file_list = glob.glob(os.path.join(datadir, "downscaling_tier2_train*.nc"))

class Test:
    def __init__(self, file_list, data_norm):
        self.file_list = file_list
        self.data_norm = data_norm
        self.data = None
        self.samples_merged = 21348
        self.lparallel = lparallel
        if lparallel:
            self.pool = multiprocessing.pool.ThreadPool(multiprocessing.cpu_count())

    def _preprocess_ds(self, ds, data_norm):
        ds = data_norm.normalize(ds)
        return ds.astype("float32")

    def read_netcdf(self, ind_s, ind_e):
        file_list_now = file_list[ind_s:ind_e]
        # read the normalized data into memory
        t0 = timer()
        def _process_netcdf(file_list_now):
            with xr.open_mfdataset(list(file_list_now), decode_cf=False, 
                                       preprocess=partial(self._preprocess_ds, data_norm=self.data_norm), parallel=True) as ds_now:
                ds_now = ds_now.load()    
                nsamples = ds_now.dims["time"]
                return ds_now
        self.data = _process_netcdf(file_list_now)
        print(f"Reading datasets of files {ind_s} to {ind_e} took {timer()-t0:.2f}s")
        print(f"Size of loaded dataset: {self.data.nbytes/(1024**3):.2f} GB")

        time.sleep(120)

        return True

data_norm = ZScore(["time", "rlat", "rlon"])
data_norm.read_norm_from_file("./norm_test.json")

test = Test(file_list, data_norm)

test.read_netcdf(0, 30)
test.read_netcdf(30, 60)
test.read_netcdf(60, 90)
