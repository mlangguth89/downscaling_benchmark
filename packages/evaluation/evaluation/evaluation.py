# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

"""
Contains all methods and classes used in main_evaluation.py.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2025-05-16"

import gc
import glob
import json as js
import logging
import multiprocessing as mp
import os
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, List, Union

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
from abstract_metric_evaluation_class import AbstractMetricEvaluation
from evaluation_utils import (
    bootstrap_grouped_hourly,
    calculate_cond_quantiles,
    check_str_in_list,
    convert_to_xarray,
    finditem,
    get_spectrum_exps,
    to_list,
)
from plotting import (
    get_season_t2m_levels,
    plot_comparison_maps,
    plot_cond_quantile,
    plot_histograms,
    plot_metric_line,
    plot_power_spectra,
    plot_score_map,
)
from scores_class import Scores

FSS_available = True
# basic data types
da_or_ds = Union[xr.DataArray, xr.Dataset]
list_or_str = Union[List[str], str]

# auxiliary variable for logger
logger_module_name = f"main_evaluation.{__name__}"
module_logger = logging.getLogger(logger_module_name)


def results_from_file(nc_file, varname, model_name):
    """
    Read downscaling results from netCDF-file.
    :param nc_file: Path to netCDF-file with downscaling results
    :param varname: Name of variable that was downscaled
    :param model_name: Name of model that was used for downscaling
    """
    # get local logger
    func_logger = logging.getLogger(
        f"{logger_module_name}.{results_from_file.__name__}"
    )

    func_logger.info(f"Read results from netCDF-file '{nc_file}'...")

    ds_out = xr.open_dataset(nc_file)

    # check if dataset contains required variables
    req_varnames = [f"{varname}_ref", f"{varname}_fcst"]
    for req_var in req_varnames:
        if req_var not in ds_out.variables:
            raise ValueError(f"Variable '{req_var}' not found in dataset '{nc_file}'")

    model_info = {
        "model_type": model_name.replace(" ", "_").lower(),
        "model_longname": model_name,
    }

    return ds_out, model_info


def run_evaluation_time(
    score_engine, score_name: str, score_unit: str, plot_dir: str, **kwargs
):
    """
    Create line plots of desired evaluation metric. Evaluation metric must have a time-dimension
    :param score_engine: Score engine object to comput evaluation metric
    :param score_name: Name of evaluation metric (must be implemented into score_engine)
    :param score_unit: Unit of evaluation metric
    :param plot_dir: Directory to save plot files
    """
    # get local logger
    func_logger = logging.getLogger(
        f"{logger_module_name}.{run_evaluation_time.__name__}"
    )

    # remove relative suffix from score_name, because suffix is handled further down below
    # and via the relative kwarg and not the suffix in the score_name
    score_name = (
        score_name.replace("_relative", "") if "_relative" in score_name else score_name
    )

    # create output-directories if necessary
    metric_dir = plot_dir.replace("/plots/", "/metric_files/")
    plot_dir = os.path.join(plot_dir, score_name)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    # get possible keyword arguments
    model_type = kwargs.pop("model_type", "sha_wgan")
    model_name = kwargs.pop("model_longname", "Model")
    quantiles = kwargs.pop("quantiles", (0.001, 0.99))

    # ad-hoc fix to remove unnecessary keyword arguments
    for key in ["model_longname", "nsubmodels", "model_dir", "hparams_dict"]:
        _ = kwargs.pop(key, None)
    # keyword arguments for configuring bootstrapping
    nboots = kwargs.pop("nboots", 1000)
    block_length = kwargs.pop("block_length", 5)

    # build score specific kwargs
    # fss, rmse, bias, mse
    score_kwargs = {
        key: kwargs.pop(key, None) for key in ["relative", "window", "thres"]
    }

    func_logger.debug(kwargs)
    if score_kwargs["relative"]:
        score_suffix = "_relative"
        score_unit = "1"
    else:
        score_suffix = ""

    score_all = score_engine(score_name, **score_kwargs)

    # include threshold value in naming style for fss
    if score_name == "fss":
        score_suffix = f"_thres_{score_kwargs['thres']}"

    score_name = f"{score_name}{score_suffix}"
    func_logger.debug(score_name)

    func_logger.info(f"Start evaluation in terms of {score_name}")

    func_logger.info(
        f"Globally averaged {score_name}: {score_all.mean().values:.4f} {score_unit}, "
        + f"standard deviation: {score_all.std().values:.4f}"
    )

    score_hourly_all = score_all.groupby("time.hour")
    score_hourly_mean = score_hourly_all.mean()

    score_hourly_mean_b = bootstrap_grouped_hourly(
        score_hourly_all, score_hourly_mean, block_length, nboots
    )

    fname_base = f"downscaling_{model_type}_{score_name.lower()}"
    fname = os.path.join(plot_dir, f"{fname_base}.png")

    # create plots
    kwargs["title"] = "year"
    plot_metric_line(
        score_hourly_mean,
        score_hourly_mean_b.quantile(quantiles[0], dim="iboot"),
        score_hourly_mean_b.quantile(quantiles[1], dim="iboot"),
        model_name,
        {score_name.upper(): score_unit},
        fname,
        **kwargs,
    )

    # save scores to netCDF
    fname_nc = os.path.join(metric_dir, f"eval_{score_name}_year.nc")

    func_logger.debug(f"Save hourly averaged {score_name} to {fname_nc}...")
    ds = xr.Dataset(
        {
            f"{score_name}": score_all,
            f"{score_name}_mean": score_hourly_mean,
            f"{score_name}_mean_boot": score_hourly_mean_b,
        }
    )
    ds.to_netcdf(fname_nc)

    # seasonal evaluation
    func_logger.debug("Run seasonal evaluation...")
    score_seas = score_all.groupby("time.season")

    for sea, score_sea in score_seas:
        fname = os.path.join(plot_dir, f"{fname_base}_{sea}.png")
        score_sea_hh = score_sea.groupby("time.hour")
        score_sea_hh_mean = score_sea_hh.mean()
        score_sea_hh_mean_b = bootstrap_grouped_hourly(
            score_sea_hh, score_sea_hh_mean, block_length, nboots
        )

        func_logger.info(
            f"Averaged {score_name} for {sea}: {score_sea.mean().values:.4f} {score_unit}, "
            + f"standard deviation: {score_sea.std().values:.4f}"
        )

        kwargs["title"] = sea
        plot_metric_line(
            score_sea_hh_mean,
            score_sea_hh_mean_b.quantile(quantiles[0], dim="iboot"),
            score_sea_hh_mean_b.quantile(quantiles[1], dim="iboot"),
            model_name,
            {score_name.upper(): score_unit},
            fname,
            **kwargs,
        )

        # save scores to netCDF
        fname_nc = os.path.join(metric_dir, f"eval_{score_name}_{sea}.nc")
        func_logger.debug(
            f"Save hourly averaged {score_name} for season {sea} to {fname_nc}..."
        )
        ds_sea = xr.Dataset(
            {
                f"{score_name}": score_sea,
                f"{score_name}_mean": score_sea_hh_mean,
                f"{score_name}_mean_boot": score_sea_hh_mean_b,
            }
        )
        ds_sea.to_netcdf(fname_nc)

    return score_all


def run_evaluation_spatial(
    score_engine,
    score_name: str,
    score_unit: str,
    plot_dir: str,
    dims=["rlat", "rlon"],
    **plt_kwargs,
):
    """
    Create map plots of desired evaluation metric. Evaluation metric must be given in rotated coordinates.
    :param score_engine: Score engine object to comput evaluation metric
    :param score_unit: Unit of evaluation metric
    :param plot_dir: Directory to save plot files
    :param dims: Spatial dimension names
    """
    # get local logger
    func_logger = logging.getLogger(
        f"{logger_module_name}.{run_evaluation_spatial.__name__}"
    )

    # remove relative suffix from score_name, because suffix is handled further down below
    # and via the relative kwarg and not the suffix in the score_name
    score_name = (
        score_name.replace("_relative", "") if "_relative" in score_name else score_name
    )

    metric_dir = plot_dir.replace("/plots/", "/metric_files/")
    plot_dir = os.path.join(plot_dir, f"{score_name}_spatial")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    model_type = plt_kwargs.pop("model_type", "sha_wgan")
    model_name = plt_kwargs.pop("model_longname", "Model")

    # ad-hoc fix to remove unnecessary keyword arguments
    for key in ["model_longname", "nsubmodels", "model_dir", "hparams_dict"]:
        _ = plt_kwargs.pop(key, None)

    # build score specific kwargs
    # fss, rmse, bias, mse
    score_kwargs = {
        key: plt_kwargs.pop(key, None) for key in ["relative", "window", "thres"]
    }
    score_all = score_engine(score_name, **score_kwargs)

    if score_kwargs["relative"]:
        score_suffix = "_relative"
        score_unit = "1"
    else:
        score_suffix = ""

    score_name = f"{score_name}{score_suffix}"

    fname_base = f"downscaling_{model_type}_{score_name.lower()}{score_suffix}"
    score_mean = score_all.mean(dim="time")
    fname = os.path.join(plot_dir, f"{fname_base}_avg_map.png")
    plot_score_map(
        score_mean,
        fname,
        model_name=model_name,
        dims=dims,
        title=f"{score_name.upper()} (avg.)",
        metric={score_name.upper(): score_unit},
        **plt_kwargs,
    )

    # save scores to netCDF
    fname_nc = os.path.join(metric_dir, f"eval_{score_name}_year.nc")
    func_logger.debug(f"Save spatial score {score_name} to {fname_nc}...")
    ds = xr.Dataset({f"{score_name}": score_all, f"{score_name}_mean": score_mean})
    ds.to_netcdf(fname_nc)

    score_mean_sea = score_all.groupby("time.season").mean(dim=["time"])
    for sea in score_mean_sea["season"]:
        fname_nc = os.path.join(metric_dir, f"eval_{score_name}_{str(sea.values)}.nc")
        score_mean_sea_iter = score_mean_sea.sel({"season": sea})
        func_logger.debug(f"Save spatial score season {score_name} to {fname_nc}...")
        ds = xr.Dataset({f"{score_name}_mean": score_mean_sea_iter})
        ds.to_netcdf(fname_nc)

    score_hourly_mean = score_all.groupby("time.hour").mean(dim=["time"])
    hours_in_data = score_hourly_mean.hour.values
    for hh in hours_in_data:
        func_logger.debug(f"Evaluation for {hh:02d} UTC")
        fname = os.path.join(plot_dir, f"{fname_base}_{hh:02d}_map.png")
        plot_score_map(
            score_hourly_mean.sel({"hour": hh}),
            fname,
            model_name=model_name,
            dims=dims,
            title=f"{score_name.upper()} {hh:02d} UTC",
            metric={score_name.upper(): score_unit},
            **plt_kwargs,
        )

    for hh in hours_in_data:
        score_now = (
            score_all.isel({"time": score_all.time.dt.hour == hh})
            .groupby("time.season")
            .mean(dim="time")
        )
        for sea in score_now["season"]:
            func_logger.debug(
                f"Evaluation for season '{str(sea.values)}' at {hh:02d} UTC"
            )
            fname = os.path.join(
                plot_dir, f"{fname_base}_{sea.values}_{hh:02d}_map.png"
            )
            plot_score_map(
                score_now.sel({"season": sea}),
                fname,
                model_name=model_name,
                dims=dims,
                title=f"{score_name} {sea.values} {hh:02d} UTC",
                metric={score_name.upper(): score_unit},
                **plt_kwargs,
            )

    return True


def run_cond_quantile_analysis(
    data_fcst, data_ref, plot_dir, varname_lables, unit, **plt_kwargs: dict
):
    """
    Create conditional quantile plots for given variables.
    :param data_fcst: xarray.DataArray with forecast data
    :param data_ref: xarray.DataArray with reference data
    :param plot_dir: Directory to save plot files
    :param varname_lables: List of variable names
    :param unit: Unit of variable
    :param plt_kwargs: Dictionary with configuration options
                Valid keys are:
                - factorization: Factorization of conditional quantile plots, i.e. "calibration_refinement" (default) or "likelihood-base_rate"
                - quantiles: Quantiles for dashed lines in plot
                - figsize: tuple with dimensions of figure, default: (12, 6)
                - fs_title: font size of title, default: 16)
                - fs_axis_label: font size of axis labels, default: fs_title-2
                - title: title of plot, default: ""
    """
    os.makedirs(plot_dir, exist_ok=True)
    # get local logger
    func_logger = logging.getLogger(
        f"{logger_module_name}.{run_cond_quantile_analysis.__name__}"
    )

    factorization = plt_kwargs.pop("factorization", "calibration_refinement")
    quantiles = plt_kwargs.pop("quantiles", [0.05, 0.5, 0.95])

    # conditional quantile analysis on all data
    quantile_panel_all, marginal_all = calculate_cond_quantiles(
        data_fcst,
        data_ref,
        varname_lables,
        unit,
        factorization=factorization,
        quantiles=quantiles,
    )

    # create plot
    plt_fname = os.path.join(
        plot_dir, f"conditional_quantile_plot_{factorization}_all.png"
    )
    plt_kwargs["title"] = "year"
    plot_cond_quantile(quantile_panel_all, marginal_all, plt_fname, **plt_kwargs)

    # conditional quantile analysis for each season
    data_fcst_seas, data_ref_seas = (
        data_fcst.groupby("time.season"),
        data_ref.groupby("time.season"),
    )

    for sea, data_fcst_sea in data_fcst_seas:
        func_logger.info(f"Start conditional quantile analysis for season '{sea}'...")
        data_ref_sea = data_ref_seas[sea]

        quantile_panel_sea, marginal_sea = calculate_cond_quantiles(
            data_fcst_sea,
            data_ref_sea,
            varname_lables,
            unit,
            factorization=factorization,
            quantiles=quantiles,
        )

        plt_fname = os.path.join(
            plot_dir, f"conditional_quantile_plot_{factorization}_{sea}.png"
        )
        plt_kwargs["title"] = sea
        plot_cond_quantile(quantile_panel_sea, marginal_sea, plt_fname, **plt_kwargs)


def run_marginal_analysis(
    data_fcst: xr.DataArray,
    data_ref: xr.DataArray,
    plot_dir: str,
    labels: List[str],
    varname: str,
    unit: str,
    **plt_kwargs: dict,
):
    """
    Perform marginal analysis for forecast (downscaled) and reference data
    which includes calculation of the Interquartile Distance (IQD) score and plotting both histograms.
    :param data_fcst: xarray.DataArray with forecast (downscaled) data
    :param data_ref: xarray.DataArray with reference (ground truth) data
    :param plot_dir: Directory to save plot files
    :param labels: List of labels for forecast and reference data
    :param varname: Physical name of variable
    :param unit: Physical unit of variable
    :param plt_kwargs: Additional keyword arguments for plotting
    :return: None
    """
    func_logger = logging.getLogger(
        f"{logger_module_name}.{run_marginal_analysis.__name__}"
    )

    # calculate IQD-score
    func_logger.info(f"Start marginal analysis for {varname}...")

    score_engine = Scores(
        data_fcst, data_ref, []
    )  # no avergaing possible for IQD, i.e. pass empty list
    iqd = score_engine("iqd")

    func_logger.info(f"IQD for all {varname} data: {iqd: .2e}")

    # create output-directories if necessary
    os.makedirs(plot_dir, exist_ok=True)

    # create histogram plot for all data
    plt_fname = os.path.join(plot_dir, f"histogram_{varname}_all.png")
    plt_kwargs["title"] = "year"
    plot_histograms(
        data_fcst,
        data_ref,
        plt_fname,
        labels,
        iqd,
        xlabel=f"{varname} [{unit}]",
        **plt_kwargs,
    )

    # conditional quantile analysis for each season
    data_fcst_seas, data_ref_seas = (
        data_fcst.groupby("time.season"),
        data_ref.groupby("time.season"),
    )

    for sea, data_fcst_sea in data_fcst_seas:
        func_logger.info(f"Start marginal analysis for season '{sea}'...")
        data_ref_sea = data_ref_seas[sea]

        score_engine = Scores(
            data_fcst_sea, data_ref_sea, []
        )  # no avergaing possible for IQD, i.e. pass empty list
        iqd_sea = score_engine("iqd")

        func_logger.info(f"IQD for {varname} data from season {sea}: {iqd_sea: .2e}")

        plt_fname = os.path.join(plot_dir, f"histogram_{varname}_{sea}.png")
        plt_kwargs["title"] = sea
        plot_histograms(
            data_fcst_sea,
            data_ref_sea,
            plt_fname,
            labels,
            iqd_sea,
            xlabel=f"{varname} [{unit}]",
            **plt_kwargs,
        )


def run_spectral_analysis(
    ds: xr.Dataset,
    data_vars: List[str],
    plot_dir: str,
    labels: List[str],
    varname: str,
    var_unit: str,
    lonlat_dims: list_or_str = ["rlon", "rlat"],
    lcutoff: bool = True,
    re: float = 6371.0,
    **plt_kwargs,
):
    """
    Run spectral analysis, create power spectrum plot and save results into a netCDF-file.
    Spectral analysis is done for all samples and for each season separately, i.e. assimung that the input data provides samples over a complete year.
    The dataset can provide multiple experiments for spectral analysis, but varname and var_unit must be the same for all experiments.
    Example: Spectral analysis for 2m temperature from downscaling and reference (ground truth) data.
    :param ds: xarray.Dataset with input data
    :param data_vars: List of variable names from ds for spectral analysis
    :param plot_dir: Directory to save plot files
    :param labels: List of labels for each variable
    :param varname: Physical name of quantity
    :param var_unit: Physical unit of quantity
    :param lonlat_dims: Name of longitude and latitude dimensions
    :param lcutoff: Flag to apply low-pass filter in spectral analysis
    :param re: Earth radius used for wavenumber calculation in spectral analysis
    :param plt_kwargs: Additional keyword arguments for plotting that are parsed to the plot_power_spectra-method
    """
    func_logger = logging.getLogger(
        f"{logger_module_name}.{run_spectral_analysis.__name__}"
    )

    metric_dir = plot_dir.replace("/plots/", "/metric_files/")
    os.makedirs(metric_dir, exist_ok=True)

    # check if number of vairables for spectral analysis and labels are equal
    ds_vars, labels = to_list(data_vars), to_list(labels)
    nexps = len(ds_vars)
    assert nexps == len(labels), (
        f"Number of variables ({nexps}) and labels ({len(labels)}) must be equal."
    )
    assert all([var in ds.variables for var in ds_vars]), (
        f"Some variables from {', '.join(ds_vars)} are not in dataset."
    )

    # compute wave numbers based on size of input data
    nlon, nlat = ds[lonlat_dims[0]].size, ds[lonlat_dims[1]].size

    dims = ["wavenumber"]
    coord_dict = {
        "wavenumber": np.arange(0, np.amin(np.array([int(nlon / 2), int(nlat / 2)])))
    }
    var_unit_spec = f"{var_unit}**2 m"

    info = {
        "lonlat_dims": lonlat_dims,
        "dims": dims,
        "coord_dict": coord_dict,
        "varname": varname,
        "var_unit": var_unit_spec,
    }

    # get power spectrum for complete dataset
    func_logger.info(f"Start spectral analysis for all data...")

    ds_ps = get_spectrum_exps(ds, ds_vars, info, lcutoff=lcutoff, re=re)

    # create plot
    os.makedirs(plot_dir, exist_ok=True)
    colors = plt_kwargs.pop("colors", ["navy", "green"])

    plt_fname = os.path.join(plot_dir, f"{varname}_power_spectrum_all.png")
    plt_kwargs["title"] = "Power spectrum YEAR"
    plot_power_spectra(
        ds_ps,
        {varname: var_unit_spec},
        labels,
        plt_fname,
        colors=colors,
        x_coord="wavenumber",
        **plt_kwargs,
    )

    # save power spectrum to netCDF
    fname_nc = os.path.join(metric_dir, f"{varname}_power_spectrum_all.nc")

    func_logger.debug(f"Save power spectrum to {fname_nc}...")
    ds_ps.to_netcdf(fname_nc)

    # get power spectrum for each season
    ds_seas = ds.groupby("time.season")

    for sea, ds_sea in ds_seas:
        func_logger.info(f"Start spectral analysis for season '{sea}'...")

        ds_ps_sea = get_spectrum_exps(ds_sea, ds_vars, info, lcutoff=lcutoff, re=re)

        plt_fname = os.path.join(plot_dir, f"{varname}_power_spectrum_{sea}.png")
        plt_kwargs["title"] = f"Power spectrum {sea}"
        plot_power_spectra(
            ds_ps_sea,
            {varname: var_unit_spec},
            labels,
            plt_fname,
            colors=colors,
            x_coord="wavenumber",
            **plt_kwargs,
        )

        # save power spectrum to netCDF
        fname_nc = os.path.join(metric_dir, f"{varname}_power_spectrum_{sea}.nc")

        func_logger.debug(f"Save power spectrum to {fname_nc}...")
        ds_ps_sea.to_netcdf(fname_nc)


def run_comparison_plots(
    ds,
    plot_dir,
    score_name,
    model_type,
    nsamples=200,
    offset=0.0,
    seasonal_levels: bool = True,
    **kwargs,
):
    """
    Run comparison plots for a given number of samples. The samples will be picked based on the performance
    of the downscaling model in terms of the provided score.
    For instance, for nsamples=100 and score_name='rmse', samples with an RMSE corresponding to
    the 0th, 1st, 2nd, ... 100th percentile will be selected.
    :param ds: xarray.Dataset providing the downscaled and reference/ground truth data
    :param plot_dir: Directory to save plot files
    :param score_name: Name of score to determine which samples to plot
    :param model_type: Type of model
    :param nsamples: Number of samples to plot
    :param offset: value to offset data (e.g. -273.15 for temperature)
    :param seasonal_levels: Flag to use seasonal levels (for 2m temperature only!)
    :param kwargs: Additional keyword arguments for plotting that are parsed to the plot_comparison_maps-method
    """
    # get local logger
    func_logger = logging.getLogger(
        f"{logger_module_name}.{run_comparison_plots.__name__}"
    )

    # create output-directories if necessary
    os.makedirs(plot_dir, exist_ok=True)

    # get score data
    metric_dir = plot_dir.replace("/plots/", "/metric_files/").replace(
        "/comparison_plots", "/temporal_evaluation"
    )
    score_file = os.path.join(metric_dir, f"eval_{score_name}_year.nc")
    if not os.path.exists(score_file):
        raise FileNotFoundError(
            f"File {score_file} not found. Run run_evaluation_time-method for score '{score_name}' first."
        )

    func_logger.debug(f"Read {score_file} to determine which samples to plot...")
    ds_score = xr.open_dataset(score_file)

    # get sample indices to plot
    sorted_score = ds_score[f"{score_name}"].sortby(ds_score[f"{score_name}"])
    indices = np.linspace(0, len(sorted_score), nsamples, dtype="int", endpoint=False)
    times2plt = sorted_score["time"].isel({"time": indices})

    # auxiliary variables
    varname = kwargs.get("vars2plt")[0].replace("_ref", "").replace("_fcst", "")

    # run parallelized plotting
    nworkers = min(int(mp.cpu_count() / 2), nsamples, 96)
    pool = Pool(processes=nworkers)
    func_logger.info(
        f"Start parallelized plotting of comparison plots for {nsamples} samples over {nworkers} workers..."
    )

    def errorhandler(exc):
        print("Exception:", exc)

    for i, t in enumerate(times2plt):
        kwargs_now = kwargs.copy()
        date_str = (pd.to_datetime(t.values)).strftime("%Y%m%dT%H00")
        quantile_now = f"{(i + 1) / nsamples:.3f}".replace(".", "p")
        fname = os.path.join(
            plot_dir,
            f"{model_type}_{varname}_{date_str}_{score_name}_q{quantile_now}.png",
        )
        if varname == "t2m" and seasonal_levels:
            kwargs_now["levels"] = get_season_t2m_levels(t.values)
        kwargs_now["suptitle"] = (
            f"{date_str} {score_name.upper()} q{quantile_now.replace('p', '.')}"
        )
        pool.apply_async(
            plot_comparison_maps,
            (ds.sel({"time": t}) + offset, fname),
            kwargs_now,
            error_callback=errorhandler,
        )

    pool.close()
    pool.join()


def run_aggregate_scores(model, varname, metric_dir, metric_list, **kwargs):
    # get local logger
    func_logger = logging.getLogger(
        f"{logger_module_name}.{run_aggregate_scores.__name__}"
    )

    # create output-directories if necessary
    os.makedirs(metric_dir, exist_ok=True)

    # load scores from precalculated metrics
    scores_dict = {
        "model": [],
        "time": [],
        "varname": [],
        "score_name": [],
        "value": [],
    }
    temporal_eval_metric_dir = metric_dir.replace(
        "aggregate_scores", "temporal_evaluation"
    )
    for metric_iter in metric_list:
        for time_agg in ["year", "MAM", "JJA", "SON", "DJF"]:
            try:
                score_iter = xr.open_dataset(
                    os.path.join(
                        temporal_eval_metric_dir, f"eval_{metric_iter}_{time_agg}.nc"
                    )
                )[f"{metric_iter}_mean"]
                score_iter_mean = np.nanmean(score_iter)
            except FileNotFoundError:
                # set the aggregated value to nan if file is not calculated and log a warning
                func_logger.warning(
                    f"Metric file {os.path.join(temporal_eval_metric_dir, f'eval_{metric_iter}_{time_agg}.nc')} "
                    f" for {model = } {time_agg = } {varname = } {metric_iter = } does"
                    " not exist. Filling with NaN."
                )
                score_iter_mean = np.nan

            scores_dict["model"].append(model)
            scores_dict["time"].append(time_agg.upper())
            scores_dict["varname"].append(varname)
            scores_dict["score_name"].append(metric_iter)
            scores_dict["value"].append(score_iter_mean)

    scores_df = pd.DataFrame(scores_dict)
    fname_csv = os.path.join(metric_dir, "scores.csv")
    func_logger.debug(f"Saving scores to {fname_csv}...")
    scores_df.to_csv(fname_csv)


class TemporalEvaluation(AbstractMetricEvaluation):
    """
    Class for temporal evaluation of downscaling results.
    """

    def __init__(
        self,
        varname: str,
        plt_dir: str,
        model_info: dict,
        avg_dims: List[str] = ["rlat", "rlon"],
        eval_dict: Dict = None,
        proj=ccrs.PlateCarree(),
    ):
        super().__init__(varname, plt_dir, model_info, avg_dims, eval_dict)

    def __call__(self, data_fcst: xr.DataArray, data_ref: xr.DataArray, **plt_kwargs):

        # get score engine
        score_engine = Scores(data_fcst, data_ref, self.avg_dims)

        # add varname to plotting kwargs
        plt_kwargs["varname"] = self.varname

        # run evaluation for each metric
        for metric, metric_config in self.evaluation_dict.items():
            # fss needs a dedicated loop for multiple threshold inputs
            if metric == "fss":
                if isinstance(metric_config["thres"], list):
                    thres_list = metric_config.pop("thres")
                    for thres in thres_list:
                        _ = run_evaluation_time(
                            score_engine,
                            metric,
                            plot_dir=self.plt_dir,
                            thres=thres,
                            **metric_config,
                            **self.model_info,
                            **plt_kwargs,
                        )
                    continue
            _ = run_evaluation_time(
                score_engine,
                metric,
                plot_dir=self.plt_dir,
                **metric_config,
                **self.model_info,
                **plt_kwargs,
            )

    def get_default_config(self, eval_dict):
        """
        Get default configuration for known variables.
        If the variable for evaluation is unknown, eval_dict cannot be None.
        :param eval_dict: Custom configuration dictionary. Can be None for known variables.
        """
        if self.varname == "t2m":
            eval_dict = {
                "rmse": {
                    "score_unit": "K",
                    "value_range": (0.0, 3.0),
                    "ref_line": None,
                    "relative": False,
                },
                "bias": {
                    "score_unit": "K",
                    "value_range": (-1.0, 1.0),
                    "ref_line": 0,
                    "relative": False,
                },
                "grad_amplitude": {
                    "score_unit": "1",
                    "value_range": (0.7, 1.2),
                    "ref_line": 1.0,
                },
                "me_std": {
                    "score_unit": "K",
                    "value_range": (0.1, 0.4),
                    "ref_line": None,
                },
                "ralsd": {
                    "score_unit": "dB",
                    "value_range": (0.0, 6.0),
                    "ref_line": None,
                },
            }
        elif self.varname == "ws100m":
            eval_dict = {
                "rmse": {
                    "score_unit": "m/s",
                    "value_range": (0.0, 3.0),
                    "ref_line": None,
                    "relative": False,
                },
                "bias": {
                    "score_unit": "m/s",
                    "value_range": (-1.0, 1.0),
                    "ref_line": 0,
                    "relative": False,
                },
                "grad_amplitude": {
                    "score_unit": "1",
                    "value_range": (0.5, 1.3),
                    "ref_line": 1.0,
                },
                "me_std": {
                    "score_unit": "m/s",
                    "value_range": (0.1, 0.5),
                    "ref_line": None,
                },
                "ralsd": {
                    "score_unit": "dB",
                    "value_range": (0.0, 10.0),
                    "ref_line": None,
                },
            }
        elif self.varname == "glob_rad":
            eval_dict = {
                "rmse": {
                    "score_unit": "W/m^2",
                    "value_range": (25.0, 250.0),
                    "ref_line": None,
                    "relative": False,
                },
                "bias": {
                    "score_unit": "W/m^2",
                    "value_range": (-30.0, 50.0),
                    "ref_line": 0,
                    "relative": False,
                },
                "rmse_relative": {
                    "score_unit": "1",
                    "value_range": (0.0, 1.0),
                    "ref_line": None,
                    "relative": True,
                },
                "bias_relative": {
                    "score_unit": "1",
                    "value_range": (-0.5, 0.5),
                    "ref_line": 0,
                    "relative": True,
                },
                "grad_amplitude": {
                    "score_unit": "1",
                    "value_range": (0.2, 1.4),
                    "ref_line": 1.0,
                },
                "me_std": {
                    "score_unit": "W/m^2",
                    "value_range": (0.0, 80.0),
                    "ref_line": None,
                },
                "ralsd": {
                    "score_unit": "dB",
                    "value_range": (0.0, 25.0),
                    "ref_line": None,
                },
            }
            if FSS_available:
                eval_dict["fss"] = {
                    "score_unit": "1",
                    "value_range": (0, 1.2),
                    "ref_line": 0.5,
                    "window": (4, 4),
                    "thres": [50, 100, 300, 500],
                }
        else:
            if eval_dict is None:
                raise ValueError(
                    f"No default configuration available for variable {self.varname}. "
                    + "Parse custom eval_dict."
                )

        return eval_dict

    def required_config_keys(self):
        return ["score_unit", "value_range", "ref_line"]


class SpatialEvaluation(AbstractMetricEvaluation):
    """
    Class for spatial evaluation of downscaling results.
    """

    def __init__(
        self,
        varname: str,
        plt_dir: str,
        model_info: dict,
        proj,
        spatial_dims=["rlat", "rlon"],
        avg_dims: List[str] = [],
        eval_dict: Dict = None,
    ):
        super().__init__(varname, plt_dir, model_info, avg_dims, eval_dict)

        self.spatial_dims = spatial_dims
        self.proj = proj

    def __call__(self, data_fcst: xr.DataArray, data_ref: xr.DataArray, **plt_kwargs):
        # get score engine
        score_engine = Scores(data_fcst, data_ref, self.avg_dims)
        # run evaluation for each metric
        for metric, metric_config in self.evaluation_dict.items():
            _ = run_evaluation_spatial(
                score_engine,
                metric,
                plot_dir=self.plt_dir,
                dims=self.spatial_dims,
                projection=self.proj,
                **self.model_info,
                **metric_config,
                **plt_kwargs,
            )

    def get_default_config(self, eval_dict):
        """
        Get default configuration for known variables.
        If the variable for evaluation is unknown, eval_dict cannot be None.
        :param eval_dict: Custom configuration dictionary. Can be None for known variables.
        """
        lvl_bias = np.arange(-2, 2.1, 0.1)
        lvl_rmse = np.arange(0.0, 3.1, 0.2)
        if self.varname == "t2m":
            eval_dict = {
                "rmse": {
                    "score_unit": "K",
                    "levels": lvl_rmse,
                    "cmap_name": "afmhot_r",
                    "relative": False,
                },
                "bias": {
                    "score_unit": "K",
                    "levels": lvl_bias,
                    "cmap_name": "seismic",
                    "relative": False,
                },
            }
        elif self.varname == "ws100m":
            eval_dict = {
                "rmse": {
                    "score_unit": "m/s",
                    "levels": lvl_rmse,
                    "cmap_name": "afmhot_r",
                    "relative": False,
                },
                "bias": {
                    "score_unit": "m/s",
                    "levels": lvl_bias,
                    "cmap_name": "seismic",
                    "relative": False,
                },
            }
        elif self.varname == "glob_rad":
            eval_dict = {
                "rmse": {
                    "score_unit": "W/m^2",
                    "levels": np.arange(0, 71, 7),
                    "cmap_name": "afmhot_r",
                    "relative": False,
                },
                "bias": {
                    "score_unit": "W/m^2",
                    "levels": np.arange(-30, 31, 6),
                    "cmap_name": "seismic",
                    "relative": False,
                },
                "rmse_relative": {
                    "score_unit": "1",
                    "levels": lvl_rmse,
                    "cmap_name": "afmhot_r",
                    "relative": True,
                },
                "bias_relative": {
                    "score_unit": "1",
                    "levels": lvl_bias,
                    "cmap_name": "seismic",
                    "relative": True,
                },
            }
        else:
            if eval_dict is None:
                raise ValueError(
                    f"No default configuration available for variable {self.varname}. "
                    + "Parse custom eval_dict."
                )
        if self.varname == "t2m":
            eval_dict = {
                "rmse": {
                    "score_unit": "K",
                    "levels": lvl_rmse,
                    "cmap_name": "afmhot_r",
                    "relative": False,
                },
                "bias": {
                    "score_unit": "K",
                    "levels": lvl_bias,
                    "cmap_name": "seismic",
                    "relative": False,
                },
            }
        elif self.varname == "ws100m":
            eval_dict = {
                "rmse": {
                    "score_unit": "m/s",
                    "levels": lvl_rmse,
                    "cmap_name": "afmhot_r",
                    "relative": False,
                },
                "bias": {
                    "score_unit": "m/s",
                    "levels": lvl_bias,
                    "cmap_name": "seismic",
                    "relative": False,
                },
            }
        elif self.varname == "glob_rad":
            eval_dict = {
                "rmse": {
                    "score_unit": "W/m^2",
                    "levels": lvl_rmse,
                    "cmap_name": "afmhot_r",
                    "relative": False,
                },
                "bias": {
                    "score_unit": "W/m^2",
                    "levels": lvl_bias,
                    "cmap_name": "seismic",
                    "relative": False,
                },
                "rmse_relative": {
                    "score_unit": "1",
                    "levels": lvl_rmse,
                    "cmap_name": "afmhot_r",
                    "relative": True,
                },
                "bias_relative": {
                    "score_unit": "1",
                    "levels": lvl_bias,
                    "cmap_name": "seismic",
                    "relative": True,
                },
            }
        else:
            if eval_dict is None:
                raise ValueError(
                    f"No default configuration available for variable {self.varname}. "
                    + "Parse custom eval_dict."
                )

        return eval_dict

    def required_config_keys(self):
        return ["levels", "cmap_name"]
