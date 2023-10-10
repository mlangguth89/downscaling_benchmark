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

def convert_to_xarray(model_output, norm, varname, coords, dims, z_branch=False):
    """
    Converts numpy-array of model output to xarray.DataArray and performs denormalization.
    :param model_output: numpy-array of model output
    :param norm: normalization object
    :param varname: name of variable
    :param coords: coordinates of target data
    :param dims: dimensions of target data
    :param z_branch: flag for z-branch
    :return: xarray.DataArray of model output with denormalized data
    """
    if z_branch:
        # slice data to get first channel only
        if isinstance(model_output, list): model_output = model_output[0]
        y_pred = xr.DataArray(model_output[..., 0].squeeze(), coords=coords, dims=dims, name=varname)
    else:
        # no slicing required
        y_pred = xr.DataArray(model_output.squeeze(), coords=coords, dims=dims, name=varname)

    # perform denormalization
    y_pred = norm.denormalize(y_pred, varname=varname)

    return y_pred


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

    # create output-directories if necessary 
    metric_dir = os.path.join(plot_dir, "metric_files")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    
    model_type = plt_kwargs.get("model_type", "wgan")

    func_logger.info(f"Start evaluation in terms of {score_name}")
    score_all = score_engine(score_name)
    score_all = score_all.drop_vars("variables")

    func_logger.info(f"Globally averaged {score_name}: {score_all.mean().values:.4f} {score_unit}, " +
                     f"standard deviation: {score_all.std().values:.4f}")  
    
    score_hourly_all = score_all.groupby("time.hour")
    score_hourly_mean, score_hourly_std = score_hourly_all.mean(), score_hourly_all.std()

    # create plots
    create_line_plot(score_hourly_mean, score_hourly_std, model_type.upper(),
                     {score_name.upper(): score_unit},
                     os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}.png"), **plt_kwargs)

    scores_to_csv(score_hourly_mean, score_hourly_std, score_name, fname=os.path.join(metric_dir, f"eval_{score_name}_year.csv"))

    score_seas = score_all.groupby("time.season")
    for sea, score_sea in score_seas:
        score_sea_hh = score_sea.groupby("time.hour")
        score_sea_hh_mean, score_sea_hh_std = score_sea_hh.mean(), score_sea_hh.std()
        func_logger.debug(f"Evaluation for season '{sea}'...")
        create_line_plot(score_sea_hh_mean,
                         score_sea_hh.std(),
                         model_type.upper(), {score_name.upper(): score_unit},
                         os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}_{sea}.png"),
                         **plt_kwargs)
        
        scores_to_csv(score_sea_hh_mean, score_sea_hh_std, score_name, 
                      fname=os.path.join(metric_dir, f"eval_{score_name}_{sea}.csv"))
    return True


def run_evaluation_spatial(score_engine, score_name: str, plot_dir: str, 
                           dims = ["rlat", "rlon"], **plt_kwargs):
    """
    Create map plots of desired evaluation metric. Evaluation metric must be given in rotated coordinates.
    :param score_engine: Score engine object to comput evaluation metric
    :param plot_dir: Directory to save plot files
    :param dims: Spatial dimension names
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{run_evaluation_time.__name__}")
    
    os.makedirs(plot_dir, exist_ok=True)

    model_type = plt_kwargs.get("model_type", "wgan")
    score_all = score_engine(score_name)
    score_all = score_all.drop_vars("variables")

    score_mean = score_all.mean(dim="time")
    fname = os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}_avg_map.png")
    create_map_score(score_mean, fname, dims=dims,
                     title=f"{score_name.upper()} (avg.)", **plt_kwargs)

    score_hourly_mean = score_all.groupby("time.hour").mean(dim=["time"])
    for hh in range(24):
        func_logger.debug(f"Evaluation for {hh:02d} UTC")
        fname = os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}_{hh:02d}_map.png")
        create_map_score(score_hourly_mean.sel({"hour": hh}), fname,
                         dims=dims, title=f"{score_name.upper()} {hh:02d} UTC",
                         **plt_kwargs)

    for hh in range(24):
        score_now = score_all.isel({"time": score_all.time.dt.hour == hh}).groupby("time.season").mean(dim="time")
        for sea in score_now["season"]:
            func_logger.debug(f"Evaluation for season '{str(sea.values)}' at {hh:02d} UTC")
            fname = os.path.join(plot_dir,
                                 f"downscaling_{model_type}_{score_name.lower()}_{sea.values}_{hh:02d}_map.png")
            create_map_score(score_now.sel({"season": sea}), fname, dims=dims,
                             title=f"{score_name} {sea.values} {hh:02d} UTC", **plt_kwargs)

    return True


def scores_to_csv(score_mean, score_std, score_name, fname="scores.csv"):
    """
    Save scores to csv file
    :param score_mean: Hourly mean of score
    :param score_std: Hourly standard deviation of score
    :param score_name: Name of score
    :param fname: Filename of csv file
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{scores_to_csv.__name__}")
    
    df_mean = score_mean.to_dataframe(name=f"{score_name}_mean")
    df_std = score_std.to_dataframe(name=f"{score_name}_std")
    df = df_mean.join(df_std)

    func_logger.info(f"Save values of {score_name} to {fname}...")
    df.to_csv(fname)
