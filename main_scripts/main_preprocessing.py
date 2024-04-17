# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2021-08-01"
__update__ = "2024-03-07"

"""
Main script to preprocess data for downscaling applications in MAELSTROM.
"""

import argparse
import json as js
import datetime as dt
from preprocess_data_unet_tier1 import Preprocess_Unet_Tier1
from preprocess_data_era5_to_ifs import PreprocessERA5toIFS
from preprocess_data_era5_to_crea6 import PreprocessERA5toCREA6

known_methods = {"Unet_Tier1": Preprocess_Unet_Tier1,
                 "ERA5_to_IFS": PreprocessERA5toIFS,
                 "ERA5_to_CREA6": PreprocessERA5toCREA6}

def main():
    # NOTE: Only arguments whose value is not None are actually parsed!
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_datadir", "-in_datadir", dest="in_datadir", type=str, default=None,
                        help="Top-level directory under which downscaling input data are stored in subdirectories " +
                             "with <year>/<month>. Should be None for synthetic downscaling task")
    parser.add_argument("--target_datadir", "-tar_datadir", dest="tar_datadir", type=str,
                        default="/p/scratch/deepacf/maelstrom/maelstrom_data/ifs_hres",
                        help="Top-level directory under which downscaling target data are stored in subdirectories " +
                             "with <year>/<month>.")
    parser.add_argument("--input_constant_file", "-in_constfile", dest="in_constfile", type=str, default=None,
                        help="File providing invariant data for the input data.")
    parser.add_argument("--target_constant_file", "-tar_constfile", dest="tar_constfile", type=str, default=None,
                        help="File providing invariant data for the target data.")
    parser.add_argument("--preproc_out_dir", "-out_dir", dest="out_dir", type=str, required=True,
                        help="Top-level directory under which preprocessed data will be stored.")
    parser.add_argument("--predictors", "-predictors", dest="predictors", type=js.loads, default=None,
                        help="Dictionary-like string to parse predictor variables (keys) together with variable types" +
                             '(values), e.g. {"2t": {"sfc": ""}}. Should be None if predictors are predefined in' +
                             'preprocessing method, e.g. for method "Unet_Tier1"')
    parser.add_argument("--predictands", "-predictands", dest="predictands", type=js.loads, default=None,
                        help="Dictionary-like string to parse predictand variables (keys) together with variable " +
                             'types (values), e.g. {"2t": {"sfc": ""}}. Should be None if predictands are predefined ' +
                             'in preprocessing method, e.g. for method "Unet_Tier1"')
    parser.add_argument("--years", "-y", dest="years", type=int, nargs="+", default=[2016, 2017, 2018, 2019, 2020],
                        help="Years of data to be preprocessed.")
    parser.add_argument("--months", "-m", dest="months", type=maybe_str_or_intlist, nargs="+", default="all",
                        help="Months of data to be preprocessed.")
    parser.add_argument("--grid_description_target", "-grid_des_tar", dest="grid_des_tar", type=str, required=True,
                        help="Grid description file to define domain of interest (target domain).")
    parser.add_argument("--preprocess_method", "-method", dest="method", type=str, required=True,
                        help="Preprocessing method to generate dataset for training, validation and testing.")
    parser.add_argument("--downscaling_factor", "-down_fac", dest="downscaling_fac", type=int, default=None, 
                        help="Downscaling factor for downscaling task.")

    args_dict = vars(parser.parse_args())

    # get and remove arguments that are not parsed when instancing preprocessing class
    preprocess_method = args_dict.pop("method")
    years = args_dict.pop("years")
    months = args_dict.pop("months")


    time_str = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    if preprocess_method in known_methods:
        preproc_cls = known_methods[preprocess_method]
        preproc_instance = preproc_cls(**{k: v for k, v in args_dict.items() if v is not None})  # only parse valid args
        
        print("Preprocessing starts at: {0}".format(time_str.replace("T", " ")))
        preproc_instance(years, months, 
                         jobname = "{0}_{1}".format(preprocess_method, time_str.replace("-", "").replace(":", "")))
    else:
        raise ValueError("Preprocessing method '{0}' is unknown. Please choose a known method: {1}."
                         .format(preprocess_method, ", ".join(known_methods.keys())))


# enable arguments that are either a string "all" or a list of integers
def maybe_str_or_intlist(arg):
    try:
       return int(arg)    # try convert to int
    except ValueError:
        pass
    if arg == "all":
        return arg
    raise argparse.ArgumentTypeError("--months/-m must be an int or 'all'")

if __name__ == "__main__":
    main()
