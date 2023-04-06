# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Demo to perform inference on trained downscaling models. In contrast to main_postprocess.py, the test dataset is run
through the trained model only, but no plots are created.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2023-03-10"
__update__ = "2023-03-10"

import os, glob
import argparse
from timeit import default_timer as timer
import json as js
# WTF, the following does not work to whatever reasons in conjunction with dt.now() on Juwels (others as well?)
# from datetime import datatime as dt
# ... thus, we go for the dirty way
import datetime as dt
import xarray as xr
import tensorflow.keras as keras
from handle_data_unet import *
from handle_data_class import HandleDataClass, get_dataset_filename
from all_normalizations import ZScore
from postprocess import get_model_info
from other_utils import print_gpu_usage, print_cpu_usage


def main(parser_args):
    t0 = timer()
    # construct model directory path and infer model type
    model_base = os.path.join(parser_args.model_base_dir, parser_args.exp_name)

    model_dir, plt_dir, norm_dir, model_type = get_model_info(model_base, "./",
                                                              parser_args.exp_name, parser_args.last,
                                                              parser_args.model_type)

    print(f"Start postprocessingi with job-ID  {parser_args.id} at " +
          f"{dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S')}")

    # read configuration files
    md_config_pattern, ds_config_pattern = f"config_{model_type}*.json", f"config_ds_{parser_args.dataset}.json"
    md_config_file, ds_config_file = glob.glob(os.path.join(model_base, md_config_pattern)), \
                                     glob.glob(os.path.join(model_base, ds_config_pattern))
    if not ds_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for dataset '{ds_config_pattern}' " +
                                f"under '{model_dir}'")
    else:
        with open(ds_config_file[0]) as dsf:
            print(f"Read dataset configuration file '{ds_config_file[0]}'.")
            ds_dict = js.load(dsf)

    if not md_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for model '{md_config_pattern}' " +
                                f"under '{model_dir}'")
    else:
        with open(md_config_file[0]) as mdf:
            print(f"Read model configuration file '{md_config_file[0]}'.")
            hparams_dict = js.load(mdf)

    # read normalization file
    js_norm = os.path.join(norm_dir, "zscore_norm.json")
    print("Read normalization file for subsequent data transformation.")
    data_norm = ZScore(ds_dict["norm_dims"])
    data_norm.read_norm_from_file(js_norm)

    # load checkpointed model
    print(f"Load model '{parser_args.exp_name}' from {model_dir}")
    t0_load = timer()
    trained_model = keras.models.load_model(model_dir, compile=False)
    print(f"Model was loaded successfully. Model loading time: {timer() - t0_load:.3f}s")

    # get filename of test data and read netCDF
    fdata_test = get_dataset_filename(parser_args.data_dir, parser_args.dataset, "test",
                                      ds_dict.get("laugmented", False))

    print(f"Start opening datafile {fdata_test}...")
    t0_read = timer()
    with xr.open_dataset(fdata_test) as ds_test:
        ds_test = data_norm.normalize(ds_test)
    da_test = HandleDataClass.reshape_ds(ds_test)

    da_test_in, da_test_tar = HandleDataClass.split_in_tar(da_test)
    tar_varname = da_test_tar['variables'].values[0]
    _ = ds_test[tar_varname].astype("float32", copy=False)
    print(f"Variable {tar_varname} serves as ground truth data.")

    if hparams_dict["z_branch"]:
        print(f"Add high-resolved target topography to input features.")
        da_test_in = xr.concat([da_test_in, da_test_tar.isel({"variables": -1})], dim="variables")

    da_test_in = da_test_in.squeeze().values

    # start inference
    print(f"Loading and preparation of test dataset finished. Data preparation time: {timer() - t0_read:.2f}s")
    t0_train = timer()
    print("Start inference on trained model...")
    _ = trained_model.predict(da_test_in, batch_size=32, verbose=2)
    tend = timer()
    print(f"Inference finished. Inference time: {tend - t0_train:.3f}s")

    print(f"Total runtime: {tend - t0:.2f}s")
    # some statistics on memory usage
    print_gpu_usage("Final GPU memory: ")
    print_cpu_usage("Final CPU memory: ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", "-data_dir", dest="data_dir", type=str, required=True,
                        help="Directory where test dataset (netCDF-file) is stored.")
    parser.add_argument("--model_base_directory", "-model_base_dir", dest="model_base_dir", type=str, required=True,
                        help="Base directory where trained models are saved.")
    parser.add_argument("--experiment_name", "-exp_name", dest="exp_name", type=str, required=True,
                        help="Name of the experiment/trained model to postprocess.")
    parser.add_argument("--downscaling_dataset", "-dataset", dest="dataset", type=str, required=True,
                        help="Name of dataset to be used for downscaling model.")
    parser.add_argument("--evaluate_last", "-last", dest="last", default=False, action="store_true",
                        help="Flag for evaluating last instead of best checkpointed model")
    parser.add_argument("--model_type", "-model_type", dest="model_type", default=None,
                        help="Name of model architecture. Only required if custom model architecture is not" +
                             "implemented in get_model_info-function (see postprocess.py)")
    parser.add_argument("--job_id", "-id", dest="id", type=int, required=True, help="Job-id from Slurm.")

    args = parser.parse_args()
    main(args)
