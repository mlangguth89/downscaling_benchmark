# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-02-15"
__update__ = "2023-03-03"

import argparse
from typing import List, Union
from timeit import default_timer as timer
import xarray as xr
from handle_data_class import HandleDataClass
from all_normalizations import ZScore

da_or_ds = Union[xr.DataArray, xr.Dataset]

make_tf_dataset_dyn = HandleDataClass.make_tf_dataset_dyn


def main():
    parser = argparse.ArgumentParser("Program that test the MAELSTROM AP5 data pipeline")
    parser.add_argument("--datadir", "-d", dest="datadir", type=str,
                        default="/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_tier2/monthly_files_copy/",
                        help="Directory where monthly netCDF-files are stored")
    parser.add_argument("--file_pattern", "-f", dest="file_patt", type=str, default="downscaling_tier2_train_*.nc", help="Filename pattern to glob netCDF-files")    
    parser.add_argument("--nfiles_load", "-n", default=30, type=int, dest="nfiles_load",
                        help="Number of netCDF-files to load into memory (2x with prefetching).")
    parser.add_argument("--lshuffle", "-lshuffle", dest="lshuffle", action="store_true", help="Enable shuffling.")
    parser.add_argument("--nepochs", "-ne", dest="nepochs", default=1, type=int,
                        help="Number of epochs to iterate over dataset.")
    parser.add_argument("--batch_size", "-b", dest="batch_size", default=192, type=int, 
                        help="Batch size for TF dataset.")
    parser.add_argument("--json_norm", "-js_norm", dest="js_norm", type=str, default=None,
                        help="Path to normalization file providing normalization parameters for given dataset.")
    parser.add_argument("--norm_dims", "-nd", dest="norm_dims", type=List, default=["time", "rlat", "rlon"],
                        help="Dimension names over which dataset should be normalized (if json_norm is unset).")
    parser.add_argument("--var_tar2in", "-tar2in", dest="var_tar2in", type=str, default=None,
                        help="Static target variable that can be used as input variable as well.")

    args = parser.parse_args()
  
    norm_dims = args.norm_dims

    if args.js_norm:
        print(f"Read normalization data from '{args.js_norm}'")
        data_norm = ZScore(args.norm_dims)
        data_norm.read_norm_from_file(args.js_norm)
        norm_dims = None
    else:
        data_norm = None


    # set-up dynamic TF dataset
    ds_obj, tfds = make_tf_dataset_dyn(args.datadir, args.file_patt, args.batch_size, args.nepochs, args.nfiles_load,
                                       args.lshuffle, var_tar2in=args.var_tar2in, norm_obj=data_norm, norm_dims=norm_dims)
    
    niter = int(args.nepochs*(ds_obj.nsamples/args.batch_size) - 1)
    t0 = timer()
    print(f"Start processing dataset with size {ds_obj.dataset_size/1.e+09:.3f} GB")
    for i, x in enumerate(tfds):
        if i == niter:
            break
    telapsed = timer() - t0

    print(f"Processing {args.nepochs} epochs of data lasted {telapsed:.1f} seconds.")
    print(f"Average throughput: {ds_obj.dataset_size*args.nepochs/1.e+06/telapsed:.3f} MB/s")

    
if __name__ == "__main__":
    main()
