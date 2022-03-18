# ********** Info **********
# @Creation: 2021-08-01
# @Update: 2021-08-01
# @Author: Michael Langguth
# @Site: Juelich supercomputing Centre (JSC) @ FZJ
# @File: preproces_downscaling_data.py
# ********** Info **********

# doc-string
"""
Main script to preprocess IFS HRES data for downscaling with UNet-architecture.
"""
# doc-string

import os, glob
import shutil
import argparse
import logging
import subprocess as sp
import datetime as dt
from preprocess_data_unet_tier1 import Preprocess_Unet_Tier1

scr_name = "preprocess_downsclaing_data"

known_methods = {"Unet_Tier1": Preprocess_Unet_Tier1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_parent_dir", "-src_dir", dest="src_dir", type=str,
                        default="/p/scratch/deepacf/maelstrom/maelstrom_data/ifs_hres",
                        help="Top-level directory under which IFS HRES data are stored with subdirectories " +
                             "<year>/<month>.")
    parser.add_argument("--out_parent_dir", "-out_dir", dest="out_dir", type=str, required=True,
                        help="Top-level directory under which remapped data will be stored.")
    parser.add_argument("--years", "-y", dest="years", type=int, nargs="+", default=[2016, 2017, 2018, 2019, 2020],
                        help="Years of data to be preprocessed.")
    parser.add_argument("--months", "-m", dest="months", type=int, nargs="+", default=range(3, 10),
                        help="Months of data to be preprocessed.")
    parser.add_argument("--grid_description_target", "-grid_des_tar", dest="grid_des_tar", type=str, required=True,
                        help="Grid description file for target data.")
    parser.add_argument("--preprocess_method", "-method", dest="method", type=str, required=True,
                        help="Preprocessing method to generate dataset for training, validation and testing.")

    args = parser.parse_args()
    dir_in = args.src_dir
    dir_out = args.out_dir
    years = args.years
    months = args.months
    grid_des_tar = args.grid_des_tar
    preprocess_method = args.method

    time_str = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    if preprocess_method in known_methods:
        preproc_class = known_methods[preprocess_method]
        preproc_instance = preproc_class(dir_in, dir_out, grid_des_tar)
        print("Preprocessing starts at: {0}".format(time_str.replace("T", " ")))
        preproc_instance(years, months, 
                         jobname = "{0}_{1}".format(preprocess_method, time_str.replace("-", "").replace(":", "")))
    else:
        raise ValueError("Preprocessing method '{0}' is unknown. Please choose a known method: {1}."
                         .format(preprocess_method, ", ".join(known_methods.keys())))


if __name__ == "__main__":
    main()
