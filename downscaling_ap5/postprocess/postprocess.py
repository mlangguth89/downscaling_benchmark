# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Auxiliary methods for postprocessing.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2022-12-08"

import os
import logging
import xarray as xr
from cartopy import crs
from plotting import create_line_plot, create_map_score

# auxiliary variable for logger
logger_module_name = f"main_postprocess.{__name__}"
module_logger = logging.getLogger(logger_module_name)


def get_model_info(model_base, output_base: str, exp_name: str, bool_last: bool = False, model_type: str = None):

    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{get_model_info.__name__}")

    model_name = os.path.basename(model_base)

    if "wgan" in exp_name:
        func_logger.debug(f"WGAN-modeltype detected.")
        add_str = "_last" if bool_last else ""
        model_dir, plt_dir = os.path.join(model_base, f"{exp_name}_generator{add_str}"), \
                             os.path.join(output_base, model_name)
        norm_dir = model_base
        model_type = "wgan"
    elif "unet" in exp_name or "deepru" in exp_name:
        func_logger.debug(f"U-Net-modeltype detected.")
        model_dir, plt_dir = model_base, os.path.join(output_base, model_name)
        norm_dir = model_dir
        model_type = "unet" if "unet" in exp_name else "deepru"
    else:
        func_logger.debug(f"Model type could not be inferred from experiment name. Try my best by defaulting...")
        if not model_type:
            raise ValueError(f"Could not infer model type from experiment name" +
                             "nor found parsed model type --model_type/model.")
        model_dir, plt_dir = model_base, os.path.join(output_base, model_name)
        norm_dir = model_dir
        model_type = model_type

    return model_dir, plt_dir, norm_dir, model_type


def run_evaluation_time(score_engine, score_name: str, score_unit: str, plot_dir: str, **plt_kwargs):
    """
    Create line plots of desired evaluation metric. Evaluation metric must have a time-dimension
    :param score_engine: Score engine object to comput evaluation metric
    :param score_name: Name of evaluation metric (must be implemented into score_engine)
    :param score_unit: Unit of evaluation metric
    :param plot_dir: Directory to save plot files
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{run_evaluation_time.__name__}")

    os.makedirs(plot_dir, exist_ok=True)
    model_type = plt_kwargs.get("model_type", "wgan")

    func_logger.info(f"Start evaluation in terms of {score_name}")
    score_all = score_engine(score_name)

    func_logger.info(f"Globally averaged {score_name}: {score_all.mean().values:.4f} {score_unit}, " +
                     f"standard deviation: {score_all.std().values:.4f}")

    score_hourly_all = score_all.groupby("time.hour")
    score_hourly_mean, score_hourly_std = score_hourly_all.mean(), score_hourly_all.std()
    for hh in range(24):
        func_logger.debug(f"Evaluation for {hh:02d} UTC")
        if hh == 0:
            tmp = score_all.isel({"time": score_all.time.dt.hour == hh}).groupby("time.season")
            score_hourly_mean_sea, score_hourly_std_sea = tmp.mean().copy(), tmp.std().copy()
        else:
            tmp = score_all.isel({"time": score_all.time.dt.hour == hh}).groupby("time.season")
            score_hourly_mean_sea, score_hourly_std_sea = xr.concat([score_hourly_mean_sea, tmp.mean()], dim="hour"), \
                                                          xr.concat([score_hourly_std_sea, tmp.std()], dim="hour")

    # create plots
    create_line_plot(score_hourly_mean, score_hourly_std, model_type.upper(),
                     {score_name.upper(): score_unit},
                     os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}.png"), **plt_kwargs)

    for sea in score_hourly_mean_sea["season"]:
        func_logger.debug(f"Evaluation for season '{sea}'...")
        create_line_plot(score_hourly_mean_sea.sel({"season": sea}),
                         score_hourly_std_sea.sel({"season": sea}),
                         model_type.upper(), {score_name.upper(): score_unit},
                         os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}_{sea.values}.png"),
                         **plt_kwargs)
    return True


def run_evaluation_spatial(score_engine, score_name: str, plot_dir: str, **plt_kwargs):
    """
    Create map plots of desired evaluation metric. Evaluation metric must be given in rotated coordinates.
    To-Do: Add flexibility regarding underlying coordinate data (i.e. projection).
    :param score_engine: Score engine object to comput evaluation metric
    :param score_name: Name of evaluation metric (must be implemented into score_engine)
    :param plot_dir: Directory to save plot files
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{run_evaluation_spatial.__name__}")

    os.makedirs(plot_dir, exist_ok=True)

    model_type = plt_kwargs.get("model_type", "wgan")
    score_all = score_engine(score_name)
    cosmo_prj = crs.RotatedPole(pole_longitude=-162.0, pole_latitude=39.25)

    score_mean = score_all.mean(dim="time")
    fname = os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}_avg_map.png")
    create_map_score(score_mean, fname, score_dims=["rlat", "rlon"],
                     title=f"{score_name.upper()} (avg.)", projection=cosmo_prj, **plt_kwargs)

    score_hourly_mean = score_all.groupby("time.hour").mean(dim=["time"])
    for hh in range(24):
        func_logger.debug(f"Evaluation for {hh:02d} UTC")
        fname = os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}_{hh:02d}_map.png")
        create_map_score(score_hourly_mean.sel({"hour": hh}), fname,
                         score_dims=["rlat", "rlon"], title=f"{score_name.upper()} {hh:02d} UTC",
                         projection=cosmo_prj, **plt_kwargs)

    for hh in range(24):
        score_now = score_all.isel({"time": score_all.time.dt.hour == hh}).groupby("time.season").mean(dim="time")
        for sea in score_now["season"]:
            func_logger.debug(f"Evaluation for season '{str(sea)}' at {hh:02d} UTC")
            fname = os.path.join(plot_dir,
                                 f"downscaling_{model_type}_{score_name.lower()}_{sea.values}_{hh:02d}_map.png")
            create_map_score(score_now.sel({"season": sea}), fname, score_dims=["rlat", "rlon"],
                             title=f"{score_name} {sea.values} {hh:02d} UTC", projection=cosmo_prj, **plt_kwargs)

    return True
