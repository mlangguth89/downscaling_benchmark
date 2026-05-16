# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference on trained downscaling models.
"""

__author__ = "Sebastian Lehner"
__email__ = "sebastian.lehner@geosphere.at"
__date__ = "2026-05-14"
__update__ = "2026-05-15"

import argparse

# import datetime as dt
import gc
import json as js
import logging
import os
from timeit import default_timer as timer
from typing import Any

import cartopy.crs as ccrs
import xarray as xr
from meta_evaluation import (
    load_aggregate_scores,
    load_metric_time_series,
    load_spectral_analysis,
    plot_multimodel_metric_line,
    plot_multimodel_power_spectra,
    visualise_scorecard,
)

# get logger
logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
logger.setLevel(logging.DEBUG)


def main(parser_args):

    ### Preparation ###
    t0 = timer()
    # set up basic output directory
    out_basedir = os.path.join(parser_args.output_base_dir)
    plt_basedir = os.path.join(parser_args.output_base_dir, "meta_evaluation")
    os.makedirs(plt_basedir, exist_ok=True)

    # load configuration for evaluation
    conf_postprocess = js.load(parser_args.conf_evaluation)

    # get some variables for convenience
    varname = conf_postprocess["varname"]
    unit = conf_postprocess["unit"]

    ## multimodel time plots
    if conf_postprocess.get("do_time_analysis", False):
        logger.info("Start temporal evaluation...")
        t0_tplot = timer()

        for metric in conf_postprocess["config_time_analysis"]["metrics"]:
            for time_agg in ["year", "DJF", "MAM", "JJA", "SON"]:
                xda = load_metric_time_series(
                    config=conf_postprocess,
                    basedir=out_basedir,
                    metric=metric,
                    fname=f"eval_{metric}_{time_agg}.nc",
                )
                plot_multimodel_metric_line(
                    xda,
                    metric={metric: unit},
                    time_period=time_agg,
                    plt_fname=f"{plt_basedir}/temporal_evaluation_{metric}_{time_agg}.png",
                    x_coord="hour",
                    value_range=conf_postprocess["config_time_analysis"][
                        "value_ranges"
                    ][metric],
                    title=f"{metric} {varname.upper()} ({time_agg})",
                )

        logger.info(f"Temporal evalutaion finished in {timer() - t0_tplot:.2f}s.")

    ## multimodel spectral analysis plots
    if conf_postprocess.get("do_spectral_analysis", False):
        logger.info("Start spectral analysis...")
        t0_spec = timer()

        for time_agg in ["year", "DJF", "MAM", "JJA", "SON"]:
            if time_agg == "year":
                fname_suffix = "all"
            else:
                fname_suffix = time_agg
            xda = load_spectral_analysis(
                config=conf_postprocess,
                basedir=out_basedir,
                varname=varname,
                fname=f"{varname}_power_spectrum_{fname_suffix}.nc",
            )
            plot_multimodel_power_spectra(
                da_ps=xda,
                var_info={varname: unit},
                labels=list(xda.model.values),
                plt_fname=f"{plt_basedir}/spectral_analysis_{varname}_{time_agg}.png",
                x_coord="wavenumber",
                title=f"Power spectrum of {varname.upper()} ({time_agg})",
            )

        logger.info(f"Spectral analysis finished in {timer() - t0_spec:.2f}s.")

    ## multimodel scorecards
    if conf_postprocess.get("do_aggregate_scores", False):
        logger.info("Start aggregate score evaluation...")
        t0_agg = timer()

        scores = load_aggregate_scores(config=conf_postprocess, basedir=out_basedir)
        visualise_scorecard(
            scores=scores,
            ref_model=conf_postprocess["reference"].upper(),
            variable="t2m",
            savepath=f"{plt_basedir}/scorecard_t2m.png",
        )

        logger.info(f"Aggregate score evaluation finished in {timer() - t0_agg:.2f}s.")

    # clean-up to reduce memory footprint
    gc.collect()

    logger.info(
        "Meta-evaluation finished. " + f"Elapsed total time: {timer() - t0:.1f}s."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_base_directory",
        "-output_base_dir",
        dest="output_base_dir",
        type=str,
        required=True,
        help="Directory where results in form of plots are stored.",
    )
    parser.add_argument(
        "--configuration_evaluation",
        "--conf_evaluation",
        dest="conf_evaluation",
        type=argparse.FileType("r"),
        required=True,
        help="JSON-file to configure evaluation.",
    )

    args = parser.parse_args()
    main(args)
