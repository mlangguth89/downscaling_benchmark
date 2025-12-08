# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference on trained downscaling models.
"""

__author__ = "Sebastian Lehner, Michael Langguth"
__email__ = "sebastian.lehner@geosphere.at, m.langguth@fz-juelich.de"
__date__ = "2025-10-03"
__update__ = "2025-10-03"

import os
from typing import Any
import logging
import argparse
from timeit import default_timer as timer
import json as js
#import datetime as dt
import gc
import xarray as xr
import cartopy.crs as ccrs
from inference import results_from_inference, results_from_inference_lightning,  run_feature_importance, run_feature_importance_lightning
from other_utils import config_logger
#from other_utils import free_mem

# get logger
logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
logger.setLevel(logging.DEBUG)

def main_lightning(parser_args):

    ### Preparation ###
    t0 = timer()
    # set up basic output directory
    plt_basedir = os.path.join(parser_args.output_base_dir, parser_args.exp_name)

    # load configuration for postprocessing
    conf_inference = js.load(parser_args.conf_inference)

    # get some variables for convenience
    varname = conf_inference["varname"]
    unit = conf_inference["unit"]

    # get data from inference or from data file
    if parser_args.mode == "inference":
        last_or_epoch = parser_args.epoch

        if parser_args.ckpt != "":
            last_or_epoch = parser_args.ckpt

        plt_dir = os.path.join(plt_basedir, f"epoch_{last_or_epoch}")
        if parser_args.ens_mem is not None:
            plt_dir = plt_dir.replace(f"epoch_{last_or_epoch}", f"epoch_{last_or_epoch}_ens{parser_args.ens_mem}")


        # create output-directory and initialze logger    
        os.makedirs(plt_dir, exist_ok=True)
        log_file = os.path.join(plt_dir, f"postprocessing_{parser_args.exp_name}.log")
        logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
        logger = config_logger(logger, log_file) 

        # get results from inference
        ds_out, test_info = results_from_inference_lightning(parser_args.model_base_dir, parser_args.exp_name, parser_args.data_dir, plt_dir,
                                                    varname, parser_args.model_type, last_or_epoch, parser_args.dataset, parser_args.ens_mem)
    # run feature importance analysis if specified
    if conf_inference.get("do_feature_importance", False) and parser_args.mode == "inference":
        # To-DO: make executable
        logger.info("Start feature importance analysis...")
        t0_fi = timer()
        
        plt_dir_importance = os.path.join(plt_dir, "feature_importance")

        # load test dataset
        ds_test = xr.open_dataset(test_info["file"])
        conf_fi = conf_inference["config_feature_importance"]

        # To-Do: allow for multiple target variables, e.g. for downscaling of wind components
        varname_tar = list(test_info["all_predictands"].keys())[0]
        # Note: The feature_importance method cannot use the prepare_dataset-method, since single predictors get randomized.
        #       The data pipeline options for the make_tf_dataset_allmem-method must therefore be constructed manually.
        data_loader_opts = {"stream_mode": test_info["stream_mode"], "batch_size": 32, "predictands": test_info["all_predictands"], 
                            "predictors": test_info["predictors"], "static_predictors": test_info["static_predictors"], 
                            "lrepeat": False, "drop_remainder": False,"lshuffle": False}
                             
        all_predictors = list(test_info["predictors"].keys()) + list(test_info["static_predictors"].keys()) if test_info["static_predictors"] is not None else list(test_info["predictors"].keys())

        _ = run_feature_importance_lightning(ds_test, conf_fi.get("predictors", all_predictors), varname_tar, test_info["trained_model"], 
                                   test_info["data_norm"], conf_fi["score_name"], data_loader_opts, plt_dir_importance, conf_fi.get("patch_size", (8, 8)))
        
        logger.info(f"Feature importance analysis finished in {timer() - t0_fi:.2f}s.")

def main(parser_args):

    ### Preparation ###
    t0 = timer()
    # set up basic output directory
    plt_basedir = os.path.join(parser_args.output_base_dir, parser_args.exp_name)

    # load configuration for postprocessing
    conf_inference = js.load(parser_args.conf_inference)

    # get some variables for convenience
    varname = conf_inference["varname"]
    unit = conf_inference["unit"]

    # get data from inference or from data file
    if parser_args.mode == "inference":
        if parser_args.last:
            last_or_epoch = "last"
        elif parser_args.epoch:
            last_or_epoch = parser_args.epoch
        else:
            last_or_epoch = "best"

        plt_dir = os.path.join(plt_basedir, f"epoch_{last_or_epoch}")
        if parser_args.ens_mem is not None:
            plt_dir = plt_dir.replace(f"epoch_{last_or_epoch}", f"epoch_{last_or_epoch}_ens{parser_args.ens_mem}")


        # create output-directory and initialze logger    
        os.makedirs(plt_dir, exist_ok=True)
        log_file = os.path.join(plt_dir, f"postprocessing_{parser_args.exp_name}.log")
        logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
        logger = config_logger(logger, log_file) 

        # get results from inference
        ds_out, test_info = results_from_inference(parser_args.model_base_dir, parser_args.exp_name, parser_args.data_dir, plt_dir,
                                                    varname, parser_args.model_type, last_or_epoch, parser_args.dataset, parser_args.ens_mem)
    # run feature importance analysis if specified
    if conf_inference.get("do_feature_importance", False) and parser_args.mode == "inference":
        # To-DO: make executable
        logger.info("Start feature importance analysis...")
        t0_fi = timer()

        plt_dir = os.path.join(plt_basedir, f"epoch_{last_or_epoch}")
        if parser_args.ens_mem is not None:
            plt_dir = plt_dir.replace(f"epoch_{last_or_epoch}", f"epoch_{last_or_epoch}_ens{parser_args.ens_mem}")

        # create output-directory and initialze logger    
        os.makedirs(plt_dir, exist_ok=True)
        plt_dir_importance = os.path.join(plt_dir, "feature_importance")

        # load test dataset
        ds_test = xr.open_dataset(test_info["file"])
        conf_fi = conf_inference["config_feature_importance"]

        # To-Do: allow for multiple target variables, e.g. for downscaling of wind components
        varname_tar = list(test_info["all_predictands"].keys())[0]
        # Note: The feature_importance method cannot use the prepare_dataset-method, since single predictors get randomized.
        #       The data pipeline options for the make_tf_dataset_allmem-method must therefore be constructed manually.
        data_loader_opts = {"stream_mode": test_info["stream_mode"], "batch_size": 32, "predictands": test_info["all_predictands"], 
                            "predictors": test_info["predictors"], "static_predictors": test_info["static_predictors"], 
                            "lrepeat": False, "drop_remainder": False,"lshuffle": False}
                             
        all_predictors = test_info["predictors"] | test_info["static_predictors"] if test_info["static_predictors"] is not None else test_info["predictors"]

        _ = run_feature_importance(ds_test, conf_fi.get("predictors", all_predictors), varname_tar, test_info["trained_model"], 
                                   test_info["data_norm"], conf_fi["score_name"], data_loader_opts, plt_dir_importance, conf_fi.get("patch_size", (4, 4)), parser_args.model_type, varname)
        
        logger.info(f"Feature importance analysis finished in {timer() - t0_fi:.2f}s.")

if __name__ == "__main__":
    
    def ens_mem_type(val: Any):
        """
        Check if parsed value is either None, a 'mean'-string or parseable as an integer.
        """
        if val is None or val == "mean":
            return val
        try:
            return int(val)
        except:
            raise argparse.ArgumentTypeError(
                f"Invalid value: {val}. Expected None, 'mean', or an integer.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_directory", "-output_base_dir", dest="output_base_dir", type=str, required=True,
                        help="Directory where results in form of plots are stored.")
    parser.add_argument("--configuration_inference", "--conf_inference", dest="conf_inference", type=argparse.FileType("r"), required=True,
                        help="JSON-file to configure inference.")
    parser.add_argument("--experiment_name", "-exp_name", dest="exp_name", type=str, required=True,
                                  help="Name of the experiment/trained model to postprocess.")
    
    # parsing arguments depending on evaluation mode (either from inference of trained model or provided results)
    subparsers = parser.add_subparsers(dest="mode", help="Provide mode")

    parser_inference = subparsers.add_parser("inference", help="Perform inference on trained model.")
    parser_inference.add_argument("--data_directory", "-data_dir", dest="data_dir", type=str, required=True,
                                  help="Directory where test dataset (netCDF-file) is stored.")
    parser_inference.add_argument("--model_base_directory", "-model_base_dir", dest="model_base_dir", type=str, required=True,
                                  help="Base directory where trained models are saved.")
    parser_inference.add_argument("--downscaling_dataset", "-dataset", dest="dataset", type=str, required=True,
                                  help="Name of dataset to be used for downscaling model.")
    parser_inference.add_argument("--model_type", "-model_type", dest="model_type", default=None,
                                  help="Name of model architecture. Only required if custom model architecture is not" +
                                       "implemented in get_model_info-function (see postprocess.py)")
    parser_inference.add_argument("--ensemble_member", "-ens_mem", dest="ens_mem", default=None, type=ens_mem_type,
                                  help="Ensemble member to evaluate. Only required for models with ensemble output during inference.")
    
    group = parser_inference.add_mutually_exclusive_group()
    group.add_argument("--evaluate_last", "-last", dest="last", default=False, action="store_true",
                       help="Flag for evaluating last instead of best checkpointed model")
    group.add_argument("--epoch", "-epoch", dest="epoch", type=int,
                       help="Epoch number to evaluate a specific checkpointed model")
    group.add_argument("--ckpt", "-ckpt", dest="ckpt",
                       help="ckpt to evaluate a specific checkpointed model")
    args = parser.parse_args()
    #main(args)
    main_lightning(args)

