import os, sys
sys.path.append("../handle_data/")
import glob
import gc
import psutil
import argparse
import xarray as xr
from functools import partial
from all_normalizations import ZScore
import time
from timeit import default_timer as timer

datadir = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_tier2/monthly_files/"
file_list = glob.glob(os.path.join(datadir, "downscaling_tier2_train*.nc"))
nfiles=30

class Test:
    def __init__(self, file_list, data_norm, lparallel=True):
        self.file_list = file_list
        self.data_norm = data_norm
        self.data = None
        self.lparallel = lparallel
        
    def _preprocess_ds(self, ds):
        ds = self.data_norm.normalize(ds)
        return ds.astype("float32")
    
    def _process_netcdf(self, file_list):
        with xr.open_mfdataset(list(file_list), decode_cf=False, preprocess=partial(self._preprocess_ds),
                               parallel=self.lparallel, engine="h5netcdf") as ds_now:
                return ds_now.load()
            
    def read_netcdf(self, ind_s, ind_e):
        del self.data
        gc.collect()
        file_list_now = file_list[ind_s:ind_e]
        # read the normalized data into memory
        t0 = timer()
        self.data = self._process_netcdf(file_list_now)
        nsamples = self.data.dims["time"]
        print(f"Reading datasets of files {ind_s} to {ind_e} took {timer()-t0:.2f}s")
        print(f"Size of loaded dataset: {self.data.nbytes/(1024**3):.2f} GB")

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

