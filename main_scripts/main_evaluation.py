# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference on trained downscaling models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2025-04-10"

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
from evaluation import results_from_file, TemporalEvaluation, SpatialEvaluation, run_cond_quantile_analysis, \
                        run_spectral_analysis, run_marginal_analysis, run_comparison_plots, run_aggregate_scores
from evaluation_utils import config_logger

# get logger
logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
logger.setLevel(logging.DEBUG)

def main(parser_args):

    ### Preparation ###
    t0 = timer()
    # set up basic output directory
    plt_basedir = os.path.join(parser_args.output_base_dir, parser_args.exp_name)

    # load configuration for evaluation
    conf_postprocess = js.load(parser_args.conf_evaluation)

    # get some variables for convenience
    varname = conf_postprocess["varname"]
    unit = conf_postprocess["unit"]

    # get data from inference or from data file
    if parser_args.mode == "inference":
        raise NotImplementedError("The inference mode of the postprocessing script is deprecated, use the dedicated inference script.")
    elif parser_args.mode == "provided_results":
        if parser_args.derive_output_dir:
            plt_dir = os.path.dirname(parser_args.results_nc)
        else:
            plt_dir = plt_basedir

        # create output-directory and initialze logger
        os.makedirs(plt_dir, exist_ok=True)
        log_file = os.path.join(plt_dir, f"postprocessing_{parser_args.exp_name}.log")
        logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
        logger = config_logger(logger, log_file)

        # get results from file
        ds_out, model_info = results_from_file(parser_args.results_nc, varname, parser_args.model_name)

    plt_dir = os.path.join(plt_dir, "plots")
    os.makedirs(plt_dir, exist_ok=True)

    # run temporal evaluation if specified
    if conf_postprocess.get("do_evaluation_time", False):
        logger.info("Start temporal evaluation...")
        t0_tplot = timer()

        plt_dir_temporal = os.path.join(plt_dir, "temporal_evaluation")

        temp_eval = TemporalEvaluation(varname, plt_dir_temporal, model_info, eval_dict=conf_postprocess.get("config_evaluation_time", None))
        temp_eval(ds_out[f"{varname}_fcst"], ds_out[f"{varname}_ref"])

        logger.info(f"Temporal evalutaion finished in {timer() - t0_tplot:.2f}s.")

    # run spatial evaluation if specified
    if conf_postprocess.get("do_evaluation_spatial", False):
        logger.info("Start spatial evaluation...")
        t0_splot = timer()

        plt_dir_spatial = os.path.join(plt_dir, "spatial_evaluation")

        spat_eval = SpatialEvaluation(varname, plt_dir_spatial, model_info, proj=ccrs.RotatedPole(pole_longitude=-162.0, pole_latitude=39.25),
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

        plt_dir_condquant = os.path.join(plt_dir, "conditional_quantile_plots")

        labels = [f"{varname.capitalize()} {model_info['model_longname']}", f"{varname.capitalize()} COSMO-REA6"]
        run_cond_quantile_analysis(ds_out[f"{varname}_fcst"], ds_out[f"{varname}_ref"], plt_dir_condquant, labels, unit,
                                          **conf_postprocess.get("config_cond_quantile_analysis", {}))

        logger.info(f"Conditional quantile plots finished in {timer() - t0_cq:.2f}s.")


    # aggregate scores
    if conf_postprocess.get("do_aggregate_scores", False):
        logger.info("Start scores aggregation...")
        t0_cq = timer()

        metric_dir_score_card = os.path.join(plt_dir, "aggregate_scores")
        metric_dir_score_card = metric_dir_score_card.replace("/plots/", "/metric_files/")

        # creating an instance of the TemporalEvaluation class here is needed to access the evaluation_dict properly
        temp_eval = TemporalEvaluation(varname, os.path.join(plt_dir, "temporal_evaluation"), model_info, eval_dict=conf_postprocess.get("config_evaluation_time", None))
        metric_list=[*temp_eval.evaluation_dict]
        # expand fss by thresholds which are part of the metric name
        if "fss" in metric_list:
            fss_thres = temp_eval.evaluation_dict["fss"]["thres"]
            metric_list.remove("fss")
            for thres in fss_thres:
                metric_list.append(f"fss_thres_{thres}")
        run_aggregate_scores(
            model=model_info['model_longname'],
            varname=varname,
            metric_dir=metric_dir_score_card,
            metric_list=metric_list,
            **conf_postprocess.get("config_aggregate_scores", None),
        )

        logger.info(f"Aggregate scores finished {timer() - t0_cq:.2f}s.")

    # clean-up to reduce memory footprint
    del ds_out
    gc.collect()
    #free_mem([da_test])

    logger.info(f"Postprocessing of experiment '{parser_args.exp_name}' finished. " +
                f"Elapsed total time: {timer() - t0:.1f}s.")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_directory", "-output_base_dir", dest="output_base_dir", type=str, required=True,
                        help="Directory where results in form of plots are stored.")
    parser.add_argument("--configuration_evaluation", "--conf_evaluation", dest="conf_evaluation", type=argparse.FileType("r"), required=True,
                        help="JSON-file to configure evaluation.")
    parser.add_argument("--experiment_name", "-exp_name", dest="exp_name", type=str, required=True,
                                  help="Name of the experiment/trained model to postprocess.")

    # parsing arguments depending on evaluation mode (either from inference of trained model or provided results)
    subparsers = parser.add_subparsers(dest="mode", help="Provide mode")

    parser_results = subparsers.add_parser("provided_results", help="Evaluate provided results.")
    parser_results.add_argument("--results_netcdf", "-results_nc", dest="results_nc", type=str, required=True,
                            help="NetCDF-file containing results to be evaluated.")
    parser_results.add_argument("--model_name", "-model_name", dest="model_name", type=str, required=True,
                                help="Name of the model for which results are provided.")
    parser_results.add_argument("--derive_outdir_from_results", "-derive_output_dir", dest="derive_output_dir", default=False, action="store_true",
                                help="Flag to derive output directory from the path of the netCDF-file providing the downscaling results. " +
                                     "Overwrites the output_base_directory argument.")

    args = parser.parse_args()
    main(args)
