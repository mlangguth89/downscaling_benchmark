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

scr_name = "preprocess_downsclaing_data"

import os, sys, glob
import argparse
import subprocess as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_parent_dir", "-src_dir", dest="src_dir", type=str,
                        default="/p/scratch/deepacf/maelstrom/maelstrom_data/ifs_hres",
                        help="Top-level directory under which IFS HRES data are stored with subdirectories " +
                             "<year>/<month>.")
    parser.add_argument("--years", "-y", dest="years", type=int, nargs="+", default=[2016, 2017, 2018, 2019, 2020],
                        help="Years of data to be preprocessed.")
    parser.add_argument("--months", "-m", dest="months", type=int, nargs="+", default=range(3, 10),
                        help="Months of data to be preprocessed.")

    args = parser.parse_args()
    dir_in = args.src_dir
    years = args.years
    months = args.months

    if os.path.isdir(dir_in):
        raise NotADirectoryError("%{0}: Parsed source directory does not exist.".format(scr_name))


def preprocess_worker(year, month, dir_in, logger):



