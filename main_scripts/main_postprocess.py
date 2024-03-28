# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference on trained downscaling models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2024-03-04"

import os, sys, glob
import logging
import argparse
from timeit import default_timer as timer
import json as js
from datetime import datetime as dt
#import datetime as dt
import gc
import numpy as np
import xarray as xr
import matplotlib as mpl
import cartopy.crs as ccrs
from handle_data_unet import *
from postprocess import results_from_inference, results_from_file, TemporalEvaluation, SpatialEvaluation, run_feature_importance, run_spectral_analysis
from other_utils import config_logger
#from other_utils import free_mem

# get logger
logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
logger.setLevel(logging.DEBUG)

def main(parser_args):

    ### Preparation ###
    t0 = timer()
    plt_dir = os.path.join(parser_args.output_base_dir, parser_args.exp_name)

    # load configuration for postprocessing
    conf_postprocess = js.load(parser_args.conf_postprocess)    

    # get some variables for convenience
    varname = conf_postprocess["varname"]
    unit = conf_postprocess["unit"]
    model_type = parser_args.model_type

    # create output-directory and initialze logger
    os.makedirs(plt_dir, exist_ok=True)
    
    log_file = os.path.join(plt_dir, f"postprocessing_{parser_args.exp_name}.log")
    logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
    logger = config_logger(logger, log_file)    

    if parser_args.mode == "inference":
        ds_test, test_info = results_from_inference(parser_args.model_base_dir, parser_args.exp_name, parser_args.data_dir, parser_args.output_base_dir,
                                                    varname, model_type, parser_args.last, parser_args.dataset)
        model_info = test_info["model_info"]
    elif parser_args.mode == "provided_results":
        ds_test, model_info = results_from_file(parser_args, plt_dir)  

    if conf_postprocess.get("do_evaluation_time", False):
        logger.info("Start temporal evaluation...")
        t0_tplot = timer()

        temp_eval = TemporalEvaluation(varname, plt_dir, model_info, eval_dict=conf_postprocess.get("config_evaluation_time", None))
        temp_eval(ds_test[f"{varname}_fcst"], ds_test[f"{varname}_ref"])
        
        logger.info(f"Temporal evalutaion finished in {timer() - t0_tplot:.2f}s.")
        
    if conf_postprocess.get("do_evaluation_spatial", False):
        logger.info("Start spatial evaluation...")
        t0_splot = timer()

        spat_eval = SpatialEvaluation(varname, plt_dir, model_info, proj=ccrs.RotatedPole(pole_longitude=-162.0, pole_latitude=39.25), 
                                      eval_dict=conf_postprocess.get("config_evaluation_spatial", None))
        spat_eval(ds_test[f"{varname}_fcst"], ds_test[f"{varname}_ref"])

        logger.info(f"Spatial evalutaion finished in {timer() - t0_splot:.2f}s.")

    # run spectral analysis
    if conf_postprocess.get("do_spectral_analysis", False):
        logger.info("Start spectral analysis...")
        t0_spec = timer()

        run_spectral_analysis(ds_test, [f"{varname}_tar", f"{varname}_ref"], plt_dir, [model_info["model_longname"], "COSMO-REA6"], varname, unit)

        logger.info(f"Spectral analysis finished in {timer() - t0_spec:.2f}s.")

    if conf_postprocess.get("do_feature_importance", False) and parser_args.mode == "inference":
        # To-DO: make executable
        logger.info("Start feature importance analysis...")
        t0_fi = timer()
        
        conf_fi = conf_postprocess["config_feature_importance"]
        ds_dict = test_info["ds_dict"]

        # To-Do: allow for multiple target variables, e.g. for wind downscaling
        varname_tar = test_info["all_predictands"][0]
        data_loader_opts = {"batch_size": 32, "varnames_tar": ds_dict.get("predictands", None), "predictors": ds_dict.get("predictors", None),
                            "var_tar2in": ds_dict.get("var_tar2in", None), "lrepeat": False, "drop_remainder": False,"lshuffle": False, 
                            "named_targets": test_info["hparams_dict"].get("named_targets", None)}
                             
        _ = run_feature_importance(ds_test, conf_fi.get("predictors", test_info["all_predictors"]), varname_tar, test_info["model"], 
                                   test_info["data_norm"], conf_fi["score_name"], data_loader_opts, plt_dir, conf_fi.get("patch_size", (8, 8)))
        
        logger.info(f"Feature importance analysis finished in {timer() - t0_fi:.2f}s.")
    
    # clean-up to reduce memory footprint
    del ds_test
    gc.collect()
    #free_mem([da_test])

    logger.info(f"Postprocessing of experiment '{parser_args.exp_name}' finished. " +
                f"Elapsed total time: {timer() - t0:.1f}s.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_directory", "-output_base_dir", dest="output_base_dir", type=str, required=True,
                        help="Directory where results in form of plots are stored.")
    parser.add_argument("--configuration_postprocess", "--conf_postprocess", dest="conf_postprocess", type=argparse.FileType("r"), required=True,
                        help="JSON-file to configure postprocessing.")
    # parsing arguments depending on evaluation mode (either from inference of trained model or provided results)
    subparsers = parser.add_subparsers(dest="mode", help="Provide mode")

    parser_inference = subparsers.add_parser("inference", help="Perform inference on trained model.")
    parser_inference.add_argument("--data_directory", "-data_dir", dest="data_dir", type=str, required=True,
                                  help="Directory where test dataset (netCDF-file) is stored.")
    parser_inference.add_argument("--model_base_directory", "-model_base_dir", dest="model_base_dir", type=str, required=True,
                                  help="Base directory where trained models are saved.")
    parser_inference.add_argument("--experiment_name", "-exp_name", dest="exp_name", type=str, required=True,
                                  help="Name of the experiment/trained model to postprocess.")
    parser_inference.add_argument("--downscaling_dataset", "-dataset", dest="dataset", type=str, required=True,
                                  help="Name of dataset to be used for downscaling model.")
    parser_inference.add_argument("--evaluate_last", "-last", dest="last", default=False, action="store_true",
                                  help="Flag for evaluating last instead of best checkpointed model")
    parser_inference.add_argument("--model_type", "-model_type", dest="model_type", default=None,
                                  help="Name of model architecture. Only required if custom model architecture is not" +
                                   "implemented in get_model_info-function (see postprocess.py)")

    parser_results = subparsers.add_parser("provided_results", help="Evaluate provided results.")
    parser_results.add_argument("--results_netcdf", "-results_nc", dest="results_nc", type=str, required=True,
                            help="NetCDF-file containing results to be evaluated.")
    
    args = parser.parse_args()
    main(args)
