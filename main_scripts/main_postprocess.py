# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference on trained downscaling models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2025-01-24"

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
from postprocess import results_from_inference, results_from_file, TemporalEvaluation, SpatialEvaluation, run_cond_quantile_analysis, \
                        run_feature_importance, run_spectral_analysis, run_marginal_analysis, run_comparison_plots
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

    # create output-directory and initialze logger
    os.makedirs(plt_dir, exist_ok=True)
    
    log_file = os.path.join(plt_dir, f"postprocessing_{parser_args.exp_name}.log")
    logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
    logger = config_logger(logger, log_file)   

    # get data from inference or from data file
    if parser_args.mode == "inference":
        if parser_args.last:
            last_or_epoch = "last"
        elif parser_args.epoch:
            last_or_epoch = parser_args.epoch
        else:
            last_or_epoch = "best"

        ds_out, test_info = results_from_inference(parser_args.model_base_dir, parser_args.exp_name, parser_args.data_dir, parser_args.output_base_dir,
                                                    varname, parser_args.model_type, last_or_epoch, parser_args.dataset, parser_args.ens_mem)
        model_info = test_info["model_info"]
    elif parser_args.mode == "provided_results":
        ds_out, model_info = results_from_file(parser_args.results_nc, varname, parser_args.model_name)  

    # run temporal evaluation if specified
    if conf_postprocess.get("do_evaluation_time", False):
        logger.info("Start temporal evaluation...")
        t0_tplot = timer()

        temp_eval = TemporalEvaluation(varname, plt_dir, model_info, eval_dict=conf_postprocess.get("config_evaluation_time", None))
        temp_eval(ds_out[f"{varname}_fcst"], ds_out[f"{varname}_ref"])
        
        logger.info(f"Temporal evalutaion finished in {timer() - t0_tplot:.2f}s.")
        
    # run spatial evaluation if specified
    if conf_postprocess.get("do_evaluation_spatial", False):
        logger.info("Start spatial evaluation...")
        t0_splot = timer()

        spat_eval = SpatialEvaluation(varname, plt_dir, model_info, proj=ccrs.RotatedPole(pole_longitude=-162.0, pole_latitude=39.25), 
                                      eval_dict=conf_postprocess.get("config_evaluation_spatial", None))
        spat_eval(ds_out[f"{varname}_fcst"], ds_out[f"{varname}_ref"])

        logger.info(f"Spatial evalutaion finished in {timer() - t0_splot:.2f}s.")

    # run spectral analysis if specified
    if conf_postprocess.get("do_spectral_analysis", False):
        logger.info("Start spectral analysis...")
        t0_spec = timer()

        plt_dir_spec = os.path.join(plt_dir, "spectral_analysis")

        run_spectral_analysis(ds_out, [f"{varname}_fcst", f"{varname}_ref"], plt_dir_spec, [model_info["model_longname"], "COSMO-REA6"], varname, unit, 
                              **conf_postprocess.get("config_spectral_analysis", {}))

        logger.info(f"Spectral analysis finished in {timer() - t0_spec:.2f}s.")

    # run analysis of marginal distributions if specified
    if conf_postprocess.get("do_marginal_analysis", False):
        logger.info("Start analysis of marginal distribution...")
        t0_marg = timer()

        plt_dir_marg = os.path.join(plt_dir, "marginal_analysis")
        run_marginal_analysis(ds_out[f"{varname}_fcst"], ds_out[f"{varname}_ref"], plt_dir_marg, [model_info["model_longname"], "COSMO-REA6"], varname, unit,
                              **conf_postprocess.get("config_marginal_analysis", {}))
        
        logger.info(f"Marginal distribution analysis finished in {timer() - t0_marg:.2f}s.")

    # create comparison plots if specified
    if conf_postprocess.get("do_comparison_plots", False):
        logger.info("Start creating comparison plots...")
        t0_cplot = timer()

        plt_dir_comp = os.path.join(plt_dir, "comparison_plots")

        # access configuration for comparison plots for convenience and set arguments for run_comparison_plots
        conf_cp = conf_postprocess["config_comparison_plots"]
        nsamples = conf_cp.pop("nsamples", 200)
        score_name = conf_cp.pop("score_name", "rmse")
        offset = conf_cp.pop("offset", 0.)

        # Note that conf_cp is parsed to the plot_comparison_maps in run_comparison_plots
        conf_cp["titles"] = [f"{varname.capitalize()} COSMO-REA6", f"{varname.capitalize()} {model_info['model_longname']}"]
        conf_cp["vars2plt"] = [f"{varname}_ref", f"{varname}_fcst"] 

        run_comparison_plots(ds_out, plt_dir_comp, score_name, model_info["model_type"], nsamples, offset, **conf_cp)

        logger.info(f"Comparison plots finished in {timer() - t0_cplot:.2f}s.")

    # create conditional quantile plots if specified
    if conf_postprocess.get("do_cond_quantile_analysis", False):
        logger.info("Start conditional quantile plots...")
        t0_cq = timer()
    
        labels = [f"{varname.capitalize()} {model_info['model_longname']}", f"{varname.capitalize()} COSMO-REA6"]
        run_cond_quantile_analysis(ds_out[f"{varname}_fcst"], ds_out[f"{varname}_ref"], plt_dir, labels, unit, 
                                          **conf_postprocess.get("config_cond_quantile_analysis", {}))      

        logger.info(f"Conditional quantile plots finished in {timer() - t0_cq:.2f}s.")  

    # run feature importance analysis if specified
    if conf_postprocess.get("do_feature_importance", False) and parser_args.mode == "inference":
        # To-DO: make executable
        logger.info("Start feature importance analysis...")
        t0_fi = timer()
        
        # load test dataset
        ds_test = xr.open_dataset(test_info["file"])
        conf_fi = conf_postprocess["config_feature_importance"]

        # To-Do: allow for multiple target variables, e.g. for downscaling of wind components
        varname_tar = test_info["all_predictands"][0]
        # Note: The feature_importance method cannot use the prepare_dataset-method, since single predictors get randomized.
        #       The data pipeline options for the make_tf_dataset_allmem-method must therefore be constructed manually.
        data_loader_opts = {"stream_mode": test_info["stream_mode"], "batch_size": 32, "predictands": test_info["all_predictands"], 
                            "predictors": test_info["predictors"], "static_predictors": test_info["static_predictors"], 
                            "lrepeat": False, "drop_remainder": False,"lshuffle": False}
                             
        all_predictors = test_info["predictors"] + test_info["static_predictors"] if test_info["static_predictors"] is not None else test_info["predictors"]

        _ = run_feature_importance(ds_test, conf_fi.get("predictors", all_predictors), varname_tar, test_info["trained_model"], 
                                   test_info["data_norm"], conf_fi["score_name"], data_loader_opts, plt_dir, conf_fi.get("patch_size", (8, 8)))
        
        logger.info(f"Feature importance analysis finished in {timer() - t0_fi:.2f}s.")
    
    # clean-up to reduce memory footprint
    del ds_out
    gc.collect()
    #free_mem([da_test])

    logger.info(f"Postprocessing of experiment '{parser_args.exp_name}' finished. " +
                f"Elapsed total time: {timer() - t0:.1f}s.")



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
                f"Invalid value: {value}. Expected None, 'mean', or an integer.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_directory", "-output_base_dir", dest="output_base_dir", type=str, required=True,
                        help="Directory where results in form of plots are stored.")
    parser.add_argument("--configuration_postprocess", "--conf_postprocess", dest="conf_postprocess", type=argparse.FileType("r"), required=True,
                        help="JSON-file to configure postprocessing.")
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

    parser_results = subparsers.add_parser("provided_results", help="Evaluate provided results.")
    parser_results.add_argument("--results_netcdf", "-results_nc", dest="results_nc", type=str, required=True,
                            help="NetCDF-file containing results to be evaluated.")
    parser_results.add_argument("--model_name", "-model_name", dest="model_name", type=str, required=True,
                                help="Name of the model for which results are provided.")
    
    args = parser.parse_args()
    main(args)
