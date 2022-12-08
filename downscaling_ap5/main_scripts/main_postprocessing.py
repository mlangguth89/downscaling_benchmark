# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference on trained downscaling models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2022-12-08"

import os
import logging
import datetime as dt
from timeit import default_timer as timer
import json as js
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from handle_data_unet import *
from handle_data_class import  *
from statistical_evaluation import Scores
from plotting import *


def main(parser_args):

    t0 = timer()

    data_dir = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/"
    model_base_dir = "/p/home/jusers/langguth1/juwels/downscaling_maelstrom/downscaling_jsc_repo/downscaling_ap5/trained_models"
    # model_base_dir = "/p/scratch/deepacf/deeprain/ji4/Downsacling/results_ap5/unet_exp0909_booster_epoch30/"
    # name of the model to be postprocessed
    model_name = "wgan_era5_to_crea6_epochs40_supervision_ztar2in_noes2"
    # model_name = "unet_era5_to_crea6_test"
    lztar = True
    # lztar = False
    last = False

    # construct model directory path and infer model type
    model_base = os.path.join(parser_args.model_base_dir, parser_args.exp_name)
    if "wgan" in parser_args.exp_name:
        add_str = "_last" if parser_args.last else ""
        add_path = ".."
        model_dir = os.path.join(model_base, f"{parser_args.exp_name}_generator{add_str}")
        model_type = "wgan"
    elif "unet" in parser_args.exp_name:
        add_str = ""
        add_path = ""
        model_dir = model_base
        model_type = "unet"
    else:
        if not parser_args.model:
            raise ValueError(f"Could not infer model type from experiment name" +
                             "nor found parsed model type --model_type/model.")
        add_str = "_last" if parser_args.last else ""
        add_path = ""
        model_dir = model_base
        model_type = parser_args.model

    # get datafile and read netCDF
    fdata_test = get_dataset_filename(parser_args.data_dir, parser_args.dataset, "test",
                                      ds_dict.get("laugmented", False))

    logging.INFO(f"Start opening datafile {fdata_test}...")
    ds_test = xr.open_dataset(fdata_test)

    # prepare training and validation data
    t0_preproc = timer()

    da_test = HandleDataClass.reshape_ds(ds_test.astype("float32", copy=False)),

    data_norm = ZScore(ds_dict["norm_dims"])
    da_test = data_norm.normalize(da_test)
    data_norm.save_norm_to_file(os.path.join(model_dir, "norm.json"))


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
    parser.add_argument("--configuration_model", "-conf_md", dest="conf_md", type=argparse.FileType("r"), required=True,
                        help="JSON-file to configure model to be trained.")
    parser.add_argument("--evaluate_last", "-last", dest="last", default=False, action="store_true",
                        help="Flag for evaluating last instead of best checkpointed model")
    parser.add_argument("--job_id", "-id", dest="id", type=int, required=True, help="Job-id from Slurm.")

    args = parser.parse_args()
    main(args)