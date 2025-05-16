# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

"""
Contains all methods and classes used in main_postrprocess.py.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2024-09-19"

import os
import glob
from typing import Union, List, Dict
from pathlib import Path
import json as js
from timeit import default_timer as timer
import logging
import gc
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from all_normalizations import GeneralNormalizer
try:
    from handle_data_class import prepare_dataset, make_tf_dataset_allmem
    from model_engine import ModelEngine
except (ModuleNotFoundError, NameError):
    # skip import when running postprocessing with different env
    print(
        "Warning: inference is not possible within the postprocessing environment, because of the missing Tensorflow package")
    pass
from abstract_metric_evaluation_class import AbstractMetricEvaluation
from scores_class import Scores
from evaluation_utils import bootstrap_grouped_hourly, sample_permut_xyt, get_spectrum_exps, calculate_cond_quantiles
from plotting import plot_metric_line, plot_score_map, create_box_plot, plot_power_spectra, plot_cond_quantile, \
                     plot_comparison_maps, plot_histograms, get_season_t2m_levels
from other_utils import convert_to_xarray, check_str_in_list, finditem, to_list

# basic data types
da_or_ds = Union[xr.DataArray, xr.Dataset]
list_or_str = Union[List[str], str]

# auxiliary variable for logger
logger_module_name = f"main_postprocess.{__name__}"
module_logger = logging.getLogger(logger_module_name)

def results_from_inference(model_base_dir: Union[Path, str], exp_name: str, data_dir: Union[Path, str], out_dir: Union[Path, str],
                           varname: str, model_type: str, last_or_epoch: Union[str, int], dataset: str, ens_member: Union[str, int] = None):
    """
    Run inference on trained model, convert output to xarray.DataArray and save results to disc.
    :param model_base_dir: Base directory where trained models are stored
    :param exp_name: Experiment name
    :param data_dir: Directory where testdata is stored
    :param out_dir: Output directory where results are stored in a netCDF-file
    :param varname: Name of variable that was downscaled
    :param model_type: Type of model (if None, model type is inferred from experiment name)
    :param last_or_epoch: Flag to either use last or best checkpointed model or the checkpointed model from a specific epoch  
    :param dataset: Name of dataset
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{results_from_inference.__name__}")

    # construct model directory path and infer model type
    model_base = Path(model_base_dir).joinpath(exp_name)

    # get trained model for inference
    trained_model, model_info = get_trained_model(model_base, exp_name, last_or_epoch, model_type)

    # read configuration files
    ds_config_pattern = f"config_ds_{dataset}.json"
    ds_config_file = glob.glob(os.path.join(model_base, ds_config_pattern))
    if not ds_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for dataset '{ds_config_pattern}' " +
                                f"under '{model_base}'")
    else:
        with open(ds_config_file[0]) as dsf:
            func_logger.info(f"Read dataset configuration file '{ds_config_file[0]}'.")
            ds_dict = js.load(dsf)
            func_logger.debug(ds_dict)

    #logger.info(f"Start postprocessing at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    func_logger.info(f"Start postprocessing at...")

    ### Run inference on trained model
    # get normalization object and preprare test dataset
    t0_preproc = timer()
    func_logger.info(f"Start preparing test dataset...")

    # prepare normalization
    js_norm = Path(model_base).joinpath("norm.json")
    func_logger.debug("Read normalization file for subsequent data transformation.")
    # get normalization methods for all variables of interest
    predictands = ds_dict["predictands"]
    if finditem(model_info["hparams_dict"], "z_branch", False):
        predictands = {**predictands, **ds_dict["varname_z"]}

    norm_config = {**ds_dict["predictors"], **ds_dict.get("var_tar2in", {}), 
                    **ds_dict.get("static_predictors", {}), **predictands}
    
    # Initialize normalization object and read normalization parameters from file
    data_norm = GeneralNormalizer(norm_config, ds_dict["norm_dims"])
    data_norm.read_norms_from_file(js_norm)

    #ds_dict["batch_size"] = 36
    
    # get dataset pipeline for inference    
    tfds_test, test_info = prepare_dataset(data_dir, dataset, ds_dict, model_info["hparams_dict"], "test", norm_obj=data_norm, 
                                           shuffle=False, lrepeat=False, drop_remainder=False) 
    
    # add further information to test_info (for later processing)
    test_info["ds_dict"] = ds_dict
    test_info["trained_model"] = trained_model
    test_info["model_info"] = model_info

    # get ground truth data
    # To-Do: Enable handling of multiple target variables (e.g. wind vectors)
    tar_varname = list(test_info["all_predictands"].keys())[0]
    func_logger.info(f"Variable {tar_varname} serves as ground truth data.")

    # get ground truth data
    ds_test = xr.open_dataset(test_info["file"])
    # rename coordinates and dimensions of target data for consistency
    dims_new = [dim.replace("_tar", "") for dim in ds_test[tar_varname].dims]
    ds_test = ds_test.rename({old: new for old, new in zip(ds_test[tar_varname].dims, dims_new) if old != new})
    coords, dims = ds_test[tar_varname].squeeze().coords, ds_test[tar_varname].squeeze().dims

    # start inference
    func_logger.info(f"Preparation of test dataset finished after {timer() - t0_preproc:.2f}s. " +
                      "Start inference on trained model...")
    t0_infer = timer()
    y_pred = trained_model.predict(tfds_test, verbose=2)

    func_logger.info(f"Inference on test dataset finished. Start denormalization of output data...")
    
    # clean-up to reduce memory footprint
    del tfds_test
    gc.collect()

    if isinstance(y_pred, list): y_pred = y_pred[0]

    ### Post-process results from test dataset
    # average over ensemble members or select specific member
    if np.ndim(y_pred) == 5:
        ens_out = True
        if ens_member == "mean":
            y_pred = np.mean(y_pred, axis=-1)
        else:
            assert isinstance(ens_member, int), f"Invalid value '{ens_member}' for ens_member. Must be 'mean' or integer."
            y_pred = y_pred[..., ens_member]
    else:
        ens_out = False

    # convert to xarray
    y_pred = convert_to_xarray(y_pred, data_norm, tar_varname, coords, dims, finditem(model_info["hparams_dict"], "z_branch", False))

    # for the global radiance downscaling task, we need to rescale the data
    if tar_varname == "global_rad_pp_ratio":
        func_logger.info("Re-scale global_rad_pp_ration to global_rad_pp.")
        y_pred = y_pred * ds_test["tisr_tar"]
        tar_varname = "global_rad_pp"

    # write inference data to netCDf
    ncfile_out = Path(out_dir).joinpath(f"downscaled_{varname}_{model_info['model_type']}.nc")
    func_logger.info(f"Write inference data to netCDF-file '{str(ncfile_out)}'")

    ds_out = xr.Dataset({f"{varname}_ref": ds_test[tar_varname].squeeze().astype("float32"), f"{varname}_fcst": y_pred}, 
                        coords=coords) 
    # add attributes such as model_type and from which model the data was generated and used ds_dict
    # This is also relevant for later processing (e,g. when doing feature importance analysis)
    ds_out.attrs["model_path"] = str(model_info["model_dir"])

    # add ensemble member information for probabilistic models
    if ens_out:
        ds_out.attrs["ensemble_output"] = ens_member if ens_member == "mean" else f"member {ens_member}"
        
    ds_out.to_netcdf(str(ncfile_out))

    func_logger.info(f"Output data on test dataset successfully processed in {timer()-t0_infer:.2f}s. Start evaluation...")

    return ds_out, test_info

def results_from_file(nc_file, varname, model_name):
    """
    Read downscaling results from netCDF-file.
    :param nc_file: Path to netCDF-file with downscaling results
    :param varname: Name of variable that was downscaled
    :param model_name: Name of model that was used for downscaling
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{results_from_file.__name__}")

    func_logger.info(f"Read results from netCDF-file '{nc_file}'...")

    ds_out = xr.open_dataset(nc_file)

    # check if dataset contains required variables
    req_varnames = [f"{varname}_ref", f"{varname}_fcst"]
    for req_var in req_varnames:
        if req_var not in ds_out.variables:
            raise ValueError(f"Variable '{req_var}' not found in dataset '{nc_file}'")

    model_info = {"model_type": model_name.replace(" ", "_").lower(), "model_longname": model_name}

    return ds_out, model_info

def get_trained_model(model_base: Union[Path, str], exp_name: str, last_or_epoch: Union[str, int], model_type: str = None):
    """
    Get trained model from model base directory and output base directory
    :param model_base: Base directory of model
    :param exp_name: Experiment name
    :param last_or_epoch: Flag to either use last or best checkpointed model or the checkpointed model from a specific epoch  
    :param model_type: Model type
    :return: Trained model for inference and model information as dictionary
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{get_trained_model.__name__}")

    if isinstance(last_or_epoch, str):
        assert last_or_epoch in ["best", "last"], f"Invalid value '{last_or_epoch}' for last_or_epoch. Must be 'best' or 'last'."
        add_str = f"_{last_or_epoch}"
    elif isinstance(last_or_epoch, int):
        add_str = f"_epoch{last_or_epoch:05d}"
    else:
        raise ValueError(f"Invalid type '{type(last_or_epoch)}' for last_or_epoch. Must be str or int.")

    model_dir = Path(model_base).joinpath(f"{exp_name}{add_str}")

    def modelinfo_from_expname(expname: str):
        model_type = None

        for known_model in ModelEngine.known_models:
            if known_model in expname:
                model_type = known_model

        if not model_type: raise ValueError(f"Could not infer known model from experiment name '{expname}'")
        
        model_instance = ModelEngine(model_type)
        nsubmodels = len(model_instance.model) - 1
        
        return (model_instance, model_type, model_instance.model_longname, nsubmodels)

    if model_type:
        func_logger.debug(f"Get model info from parsed model type '{model_type}'")

        model_instance = ModelEngine(model_type)
        model_longname = model_instance.model_longname
        nsubmodels = len(model_instance.model) - 1 
        model_info = {"model_type": model_type, "model_longname": model_instance.model_longname,
                      "nsubmodels": len(model_instance.model) - 1}
    else:
        func_logger.debug(f"Try to infer model info from parsed experiment name '{exp_name}'")

        model_instance, model_type, model_longname, nsubmodels = modelinfo_from_expname(exp_name)

    # read configuration files
    md_config_pattern = f"config_{model_type}.json"
    md_config_file = glob.glob(str(model_base.joinpath(md_config_pattern)))

    if not md_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for model '{md_config_pattern}' " +
                                f"under '{model_base}'")
    else:
        with open(md_config_file[0]) as mdf:
            func_logger.info(f"Read model configuration file '{md_config_file[0]}'.")
            hparams_dict = js.load(mdf)
            func_logger.debug(hparams_dict)
    
    #hparams_dict["batch_size"] = 36

    model_info = {"model_dir": model_dir, "model_type": model_type, "model_longname": model_longname,
                  "nsubmodels": nsubmodels, "hparams_dict": hparams_dict}

    # initialize model with dummy-values for shape_in and varnames_tar as they are obtained when loading saved model
    # Note: shape_in = None triggers dummy-values of shape_in in model classes
    vars_tar_dummy = ["dummy1", "dummy2"] if finditem(model_info["hparams_dict"], "z_branch", False) else "dummy"
    trained_model = model_instance(None, vars_tar_dummy, hparams_dict, model_base, exp_name)

    # ...and load checkpointed model
    func_logger.info(f"Load model '{exp_name}' from {model_dir}")
    trained_model = trained_model.load_inference_model(model_dir)
    func_logger.info(f"Model was loaded successfully.")

    return trained_model, model_info      
        

def run_evaluation_time(score_engine, score_name: str, score_unit: str, plot_dir: str, **kwargs):
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
    
    # get possible keyword arguments
    model_type = kwargs.pop("model_type", "sha_wgan")
    model_name = kwargs.pop("model_name", "Sha WGAN")
    quantiles = kwargs.pop("quantiles", (.001, .99))

    # ad-hoc fix to remove unnecessary keyword arguments
    for key in ["model_longname", "nsubmodels", "model_dir", "hparams_dict"]:
        _ = kwargs.pop(key, None)
    # keyword arguments for configuring bootstrapping
    nboots = kwargs.pop("nboots", 1000)
    block_length = kwargs.pop("block_length", 5)

    # build score specific kwargs
    # fss, rmse, bias, mse
    score_kwargs = {key: kwargs.pop(key, None) for key in ["relative", "window", "thres"]}
    
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

    func_logger.info(f"Globally averaged {score_name}: {score_all.mean().values:.4f} {score_unit}, " +
                     f"standard deviation: {score_all.std().values:.4f}")  
    
    score_hourly_all = score_all.groupby("time.hour")
    score_hourly_mean = score_hourly_all.mean()

    score_hourly_mean_b = bootstrap_grouped_hourly(score_hourly_all, score_hourly_mean, block_length, nboots)   
    
    fname_base = f"downscaling_{model_type}_{score_name.lower()}"
    fname = os.path.join(plot_dir, f"{fname_base}.png")

    
    # create plots
    plot_metric_line(score_hourly_mean, score_hourly_mean_b.quantile(quantiles[0], dim="iboot"), score_hourly_mean_b.quantile(quantiles[1], dim="iboot"),
                     model_name, {score_name.upper(): score_unit},
                     fname, **kwargs)

    # save scores to netCDF
    fname_nc = os.path.join(metric_dir, f'eval_{score_name}_year.nc')

    func_logger.debug(f"Save hourly averaged {score_name} to {fname_nc}...")
    ds = xr.Dataset({f"{score_name}": score_all, f"{score_name}_mean": score_hourly_mean, f"{score_name}_mean_boot": score_hourly_mean_b})
    ds.to_netcdf(fname_nc)

    # seasonal evaluation
    func_logger.debug("Run seasonal evaluation...")
    score_seas = score_all.groupby("time.season")

    for sea, score_sea in score_seas:
        fname = os.path.join(plot_dir, f"{fname_base}_{sea}.png")
        score_sea_hh = score_sea.groupby("time.hour")
        score_sea_hh_mean = score_sea_hh.mean()
        score_sea_hh_mean_b = bootstrap_grouped_hourly(score_sea_hh, score_sea_hh_mean, block_length, nboots)  

        func_logger.info(f"Averaged {score_name} for {sea}: {score_sea.mean().values:.4f} {score_unit}, " +
                         f"standard deviation: {score_sea.std().values:.4f}")  
        
        plot_metric_line(score_sea_hh_mean, score_sea_hh_mean_b.quantile(quantiles[0], dim="iboot"), score_sea_hh_mean_b.quantile(quantiles[1], dim="iboot"),
                         model_name, {score_name.upper(): score_unit},
                         fname, **kwargs)
        
        # save scores to netCDF
        fname_nc = os.path.join(metric_dir, f'eval_{score_name}_{sea}.nc')
        func_logger.debug(f"Save hourly averaged {score_name} for season {sea} to {fname_nc}...")
        ds_sea = xr.Dataset({f"{score_name}": score_sea, f"{score_name}_mean": score_sea_hh_mean, f"{score_name}_mean_boot": score_sea_hh_mean_b})
        ds_sea.to_netcdf(fname_nc)

    return score_all


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

    model_type = plt_kwargs.pop("model_type", "sha_wgan")
    # ad-hoc fix to remove unnecessary keyword arguments
    for key in ["model_longname", "nsubmodels", "model_dir", "hparams_dict"]:
        _ = plt_kwargs.pop(key, None)
    
    # build score specific kwargs
    # fss, rmse, bias, mse
    score_kwargs = {key: plt_kwargs.pop(key, None) for key in ["relative", "window", "thres"]}
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
    plot_score_map(score_mean, fname, dims=dims,
                     title=f"{score_name.upper()} (avg.)", **plt_kwargs)

    score_hourly_mean = score_all.groupby("time.hour").mean(dim=["time"])
    for hh in range(24):
        func_logger.debug(f"Evaluation for {hh:02d} UTC")
        fname = os.path.join(plot_dir, f"{fname_base}_{hh:02d}_map.png")
        plot_score_map(score_hourly_mean.sel({"hour": hh}), fname,
                       dims=dims, title=f"{score_name.upper()} {hh:02d} UTC", **plt_kwargs)

    for hh in range(24):
        score_now = score_all.isel({"time": score_all.time.dt.hour == hh}).groupby("time.season").mean(dim="time")
        for sea in score_now["season"]:
            func_logger.debug(f"Evaluation for season '{str(sea.values)}' at {hh:02d} UTC")
            fname = os.path.join(plot_dir,
                                 f"{fname_base}_{sea.values}_{hh:02d}_map.png")
            plot_score_map(score_now.sel({"season": sea}), fname, dims=dims,
                           title=f"{score_name} {sea.values} {hh:02d} UTC", **plt_kwargs)

    return True

def run_cond_quantile_analysis(data_fcst, data_ref, plt_dir, varname_lables, unit, **opts: dict):
    """
    Create conditional quantile plots for given variables.
    :param data_fcst: xarray.DataArray with forecast data
    :param data_ref: xarray.DataArray with reference data
    :param plt_dir: Directory to save plot files
    :param varname_lables: List of variable names
    :param unit: Unit of variable
    :param opt: Dictionary with configuration options
                Valid keys are: 
                - "factorization": Factorization of conditional quantile plots, i.e. "calibration_refinement" (default) or "likelihood-base_rate"
                - "quantiles": Quantiles for dashed lines in plot
                - "figsize": tuple with dimensions of figure, default: (12, 6)
                - "fs_title": font size of title, default: 16)
                - "fs_axis_label": font size of axis labels, default: fs_title-2
                - "plt_title": title of plot, default: ""
    """

    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{run_cond_quantile_analysis.__name__}")

    factorization = opts.pop("factorization", "calibration_refinement")  
    quantiles = opts.pop("quantiles", [0.05, 0.5, 0.95])

    # conditional quantile analysis on all data
    quantile_panel_all, marginal_all = calculate_cond_quantiles(data_fcst, data_ref, varname_lables, unit, factorization=factorization, quantiles=quantiles)

    # create plot
    plt_fname = os.path.join(plt_dir, f"conditional_quantile_plot_{factorization}_all.png")
    plot_cond_quantile(quantile_panel_all, marginal_all, plt_fname, **opts)

    # conditional quantile analysis for each season
    data_fcst_seas, data_ref_seas = data_fcst.groupby("time.season"), data_ref.groupby("time.season")

    for sea, data_fcst_sea in data_fcst_seas:
        func_logger.info(f"Start conditional quantile analysis for season '{sea}'...")
        data_ref_sea = data_ref_seas[sea]

        quantile_panel_sea, marginal_sea = calculate_cond_quantiles(data_fcst_sea, data_ref_sea, varname_lables, unit, factorization=factorization, quantiles=quantiles)

        plt_fname = os.path.join(plt_dir, f"conditional_quantile_plot_{factorization}_{sea}.png")
        plot_cond_quantile(quantile_panel_sea, marginal_sea, plt_fname, **opts) 

def run_marginal_analysis(data_fcst: xr.DataArray, data_ref: xr.DataArray, plt_dir: str, labels: List[str], varname: str, unit: str, **opts: dict):
    """
    Perform marginal analysis for forecast (downscaled) and reference data
    which includes calculation of the Interquartile Distance (IQD) score and plotting both histograms.
    :param data_fcst: xarray.DataArray with forecast (downscaled) data
    :param data_ref: xarray.DataArray with reference (ground truth) data
    :param plt_dir: Directory to save plot files
    :param labels: List of labels for forecast and reference data
    :param varname: Physical name of variable
    :param unit: Physical unit of variable
    :param opts: Additional keyword arguments for plotting
    :return: None
    """ 
    func_logger = logging.getLogger(f"{logger_module_name}.{run_marginal_analysis.__name__}")

    # calculate IQD-score
    func_logger.info(f"Start marginal analysis for {varname}...")

    score_engine = Scores(data_fcst, data_ref, [])      # no avergaing possible for IQD, i.e. pass empty list
    iqd = score_engine("iqd")

    func_logger.info(f"IQD for all {varname} data: {iqd: .2e}")

    # create output-directories if necessary
    os.makedirs(plt_dir, exist_ok=True)

    # create histogram plot for all data
    plt_fname = os.path.join(plt_dir, f"histogram_{varname}_all.png")
    plot_histograms(data_fcst, data_ref, plt_fname, labels, iqd, xlabel=f"{varname} [{unit}]", **opts)

    # conditional quantile analysis for each season
    data_fcst_seas, data_ref_seas = data_fcst.groupby("time.season"), data_ref.groupby("time.season")

    for sea, data_fcst_sea in data_fcst_seas:
        func_logger.info(f"Start marginal analysis for season '{sea}'...")
        data_ref_sea = data_ref_seas[sea]

        score_engine = Scores(data_fcst_sea, data_ref_sea, [])      # no avergaing possible for IQD, i.e. pass empty list
        iqd_sea = score_engine("iqd")

        func_logger.info(f"IQD for {varname} data from season {sea}: {iqd_sea: .2e}")

        plt_fname = os.path.join(plt_dir, f"histogram_{varname}_{sea}.png")
        plot_histograms(data_fcst_sea, data_ref_sea, plt_fname, labels, iqd_sea, xlabel=f"{varname} [{unit}]", **opts)
                           

def run_spectral_analysis(ds: xr.Dataset, data_vars: List[str], plt_dir: str, labels: List[str], varname: str, var_unit: str,
                          lonlat_dims: list_or_str = ["rlon", "rlat"], lcutoff: bool= True, re: float = 6371., **plt_kwargs):
    """
    Run spectral analysis, create power spectrum plot and save results into a netCDF-file.
    Spectral analysis is done for all samples and for each season separately, i.e. assimung that the input data provides samples over a complete year.
    The dataset can provide multiple experiments for spectral analysis, but varname and var_unit must be the same for all experiments.
    Example: Spectral analysis for 2m temperature from downscaling and reference (ground truth) data.
    :param ds: xarray.Dataset with input data
    :param data_vars: List of variable names from ds for spectral analysis 
    :param plt_dir: Directory to save plot files
    :param labels: List of labels for each variable
    :param varname: Physical name of quantity
    :param var_unit: Physical unit of quantity
    :param lonlat_dims: Name of longitude and latitude dimensions
    :param lcutoff: Flag to apply low-pass filter in spectral analysis
    :param re: Earth radius used for wavenumber calculation in spectral analysis
    :param plt_kwargs: Additional keyword arguments for plotting that are parsed to the plot_power_spectra-method
    """
    func_logger = logging.getLogger(f"{logger_module_name}.{run_spectral_analysis.__name__}")

    # check if number of vairables for spectral analysis and labels are equal
    ds_vars, labels = to_list(data_vars), to_list(labels)
    nexps = len(ds_vars)    
    assert nexps== len(labels), f"Number of variables ({nexps}) and labels ({len(labels)}) must be equal."
    assert all([var in ds.variables for var in ds_vars]), f"Some variables from {', '.join(ds_vars)} are not in dataset."

    # compute wave numbers based on size of input data
    nlon, nlat = ds[lonlat_dims[0]].size, ds[lonlat_dims[1]].size

    dims = ["wavenumber"]
    coord_dict = {"wavenumber": np.arange(0, np.amin(np.array([int(nlon/2), int(nlat/2)])))}
    var_unit_spec = f"{var_unit}**2 m"

    info = {"lonlat_dims": lonlat_dims, "dims": dims, "coord_dict": coord_dict, "varname": varname, "var_unit": var_unit_spec}

    # get power spectrum for complete dataset
    func_logger.info(f"Start spectral analysis for all data...")

    ds_ps = get_spectrum_exps(ds, ds_vars, info, lcutoff=lcutoff, re=re) 

    # create plot
    os.makedirs(plt_dir, exist_ok=True)
    colors = plt_kwargs.pop("colors", ["navy", "green"])

    plt_fname = os.path.join(plt_dir, f"{varname}_power_spectrum_all.png")
    plot_power_spectra(ds_ps, {varname: var_unit_spec}, labels, plt_fname, colors= colors,
                       x_coord="wavenumber", **plt_kwargs)
    
    # save power spectrum to netCDF
    fname_nc = os.path.join(plt_dir, f'{varname}_power_spectrum_all.nc')

    func_logger.debug(f"Save power spectrum to {fname_nc}...")
    ds_ps.to_netcdf(fname_nc)

    # get power spectrum for each season
    ds_seas = ds.groupby("time.season")

    for sea, ds_sea in ds_seas:
        func_logger.info(f"Start spectral analysis for season '{sea}'...")

        ds_ps_sea = get_spectrum_exps(ds_sea, ds_vars, info, lcutoff=lcutoff, re=re)

        plt_fname = os.path.join(plt_dir, f"{varname}_power_spectrum_{sea}.png")
        plot_power_spectra(ds_ps_sea, {varname: var_unit_spec}, labels, plt_fname, colors= colors,
                           x_coord="wavenumber", **plt_kwargs)
        
        # save power spectrum to netCDF
        fname_nc = os.path.join(plt_dir, f'{varname}_power_spectrum_{sea}.nc')

        func_logger.debug(f"Save power spectrum to {fname_nc}...")
        ds_ps_sea.to_netcdf(fname_nc)
        

def run_feature_importance(ds: xr.Dataset, predictors: list_or_str, varname_tar: str, model, norm, score_name: str,
                           data_loader_opt: dict, plt_dir: str, patch_size = (6, 6)):
    """
    Run feature importance analysis and create box-plot of results
    :param ds: Unnormalized xr.Dataset with predictors and target variable
    :param predictors: List of predictor names for which feature importance analysis should be run
    :param varname_tar: Name of target variable
    :param model: Model object
    :param norm: Normalization object
    :param score_name: Name of score to compute feature importance
    :param data_loader_opt: Data loader options that will be parsed to the make_tf_dataset_allmem-method
    :param plt_dir: Directory to save plot files
    :param patch_size: Patch size for feature importance analysis
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{run_feature_importance.__name__}")
    
    # get feature importance scores
    func_logger.debug(f"Start feature importance analysis for {score_name}...")
    feature_scores = feature_importance(ds, predictors, varname_tar, model, norm, score_name, data_loader_opt, 
                                        patch_size=patch_size)
    
    # get reference score
    func_logger.debug(f"Retrieve reference score to finish feature importance analysis...")
    score_file = os.path.join(plt_dir, "metric_files", f"eval_{score_name}_year.nc")
    if not os.path.exists(score_file):
        raise FileNotFoundError(f"File {score_file} not found. Run run_evaluation_time-method for score '{score_name}' first.")
    ds_score = xr.open_dataset(score_file)
    ref_score = ds_score[f"{score_name}"] 

    rel_changes = feature_scores / ref_score
    max_rel_change = int(np.ceil(np.amax(rel_changes) + 1.))

    # plot feature importance scores in a box-plot with whiskers where each variable is a box
    plt_fname = os.path.join(plt_dir, f"feature_importance_{score_name}.png")

    func_logger.debug(f"Plot feature importance-analysis results into file '{plt_fname}'.")
    create_box_plot(rel_changes.T, plt_fname, **{"title": f"Feature Importance ({score_name.upper()})", "ref_line": 1., "widths": .3, 
                                                 "xlabel": "Predictors", "ylabel": f"Rel. change {score_name.upper()}", "labels": predictors, 
                                                 "yticks": range(1, max_rel_change), "colors": "b"})

    return feature_scores

def feature_importance(ds: xr.Dataset, predictors: list_or_str, varname_tar: str, model, norm, score_name: str,
                       data_loader_opt: dict, patch_size = (8, 8)):
    """
    Run featiure importance analysis based on permutation method (see signature of sample_permut_xyt-method)
    :param ds: The unnormalized (test-)dataset
    :param predictors: List of predictor variables for which feature importance analysis should be run
    :param varname_tar: Name of target variable
    :param model: Trained model for inference
    :param norm: Normalization object
    :param score_name: Name of metric-score to be calculated
    :param data_loader_opt: Dictionary providing options for the make_tf_dataset_allmem-method
    :param patch_size: Tuple for patch size during spatio-temporal permutation
    :return score_all: DataArray with scores for all predictor variables
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{feature_importance.__name__}")

    # sanity checks
    _ = check_str_in_list(list(ds.data_vars), predictors)
    #try:
    #    assert ds.dims[0] == "time", f"First dimension of the data must be a time-dimensional, but is {ds.dims[0]}."
    #except AssertionError as e:
    #    func_logger.error(e, stack_info=True, exc_info=True)
    #    raise e

    ntimes = len(ds["time"])

    # get ground truth data and underlying metadata
    ground_truth = ds[varname_tar].copy() 
    # normalize dataset
    ds = norm.normalize(ds)   

    # initialize score-array
    score_all = xr.DataArray(np.zeros((len(predictors), ntimes)), coords={"predictor": predictors, "time": ds["time"]},
                             dims=["predictor", "time"])

    for var in predictors:
        func_logger.info(f"Run sample importance analysis for {var}...")
        # get copy of sample array
        ds_copy = ds.copy(deep=True)
        # permute sample
        da_now = ds[var].copy()
        if "time" not in da_now.dims:
            da_now = da_now.expand_dims({"time": ds_copy["time"]}, axis=0)
        da_permut = sample_permut_xyt(da_now, patch_size=patch_size)
        ds_copy[var] = da_permut
        
        # get TF dataset
        func_logger.info(f"Set-up data pipeline with permuted sample for {var}...")
        stream_mode = data_loader_opt.pop("stream_mode")
        tfds_test = make_tf_dataset_allmem(stream_mode, ds_copy, **data_loader_opt)

        # predict
        func_logger.info(f"Run inference with permuted sample for {var}...")
        y_pred = model.predict(tfds_test, verbose=2)

        # convert to xarray
        y_pred = convert_to_xarray(y_pred, norm, varname_tar, ground_truth.coords, ground_truth.dims, True)

        # calculate score
        func_logger.info(f"Calculate score for permuted samples of {var}...")
        score_engine = Scores(y_pred, ground_truth, dims=ground_truth.dims[1::])
        score_all.loc[{"predictor": var}] = score_engine(score_name)

        #free_mem([da_copy, da_permut, tfds_test, y_pred, score_engine])

    return score_all

def run_comparison_plots(ds, plt_dir, score_name, model_type, nsamples = 200, offset = 0., seasonal_levels: bool = True, **kwargs):
    """
    Run comparison plots for a given number of samples. The samples will be picked based on the performance 
    of the downscaling model in terms of the provided score.
    For instance, for nsamples=100 and score_name='rmse', samples with an RMSE corresponding to
    the 0th, 1st, 2nd, ... 100th percentile will be selected.
    :param ds: xarray.Dataset providing the downscaled and reference/ground truth data
    :param plt_dir: Directory to save plot files
    :param score_name: Name of score to determine which samples to plot
    :param model_type: Type of model
    :param nsamples: Number of samples to plot
    :param offset: value to offset data (e.g. -273.15 for temperature)
    :param seasonal_levels: Flag to use seasonal levels (for 2m temperature only!)
    :param kwargs: Additional keyword arguments for plotting that are parsed to the plot_comparison_maps-method
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{run_comparison_plots.__name__}")

    # create output-directories if necessary
    os.makedirs(plt_dir, exist_ok=True)

    # get score data
    score_file = os.path.join(plt_dir, "..", "metric_files", f"eval_{score_name}_year.nc")
    if not os.path.exists(score_file):
        raise FileNotFoundError(f"File {score_file} not found. Run run_evaluation_time-method for score '{score_name}' first.")
    
    func_logger.debug(f"Read {score_file} to determine which samples to plot...")
    ds_score = xr.open_dataset(score_file)

    # get sample indices to plot
    sorted_score = ds_score[f"{score_name}"].sortby(ds_score[f"{score_name}"])
    indices = np.linspace(0, len(sorted_score), nsamples, dtype="int", endpoint=False)
    times2plt = sorted_score["time"].isel({"time": indices})

    # auxiliary variables
    varname = kwargs.get("vars2plt")[0].replace("_ref", "").replace("_fcst", "")

    # run parallelized plotting
    nworkers = min(int(mp.cpu_count()/2), nsamples, 96)
    pool = Pool(processes=nworkers)
    func_logger.info(f"Start parallelized plotting of comparison plots for {nsamples} samples over {nworkers} workers...")

    def errorhandler(exc):
        print('Exception:', exc)

    for i, t in enumerate(times2plt):
        kwargs_now = kwargs.copy()
        date_str = (pd.to_datetime(t.values)).strftime("%Y%m%dT%H00")
        quantile_now = f"{(i+1) / nsamples:.3f}".replace(".", "p")
        fname = os.path.join(plt_dir, f"{model_type}_{varname}_{date_str}_{score_name}_q{quantile_now}.png")
        if varname == "t2m" and seasonal_levels:
           kwargs_now["levels"] = get_season_t2m_levels(t.values)
        pool.apply_async(plot_comparison_maps, (ds.sel({"time": t}) + offset, fname), kwargs_now, error_callback=errorhandler)
        
    pool.close()
    pool.join()

class TemporalEvaluation(AbstractMetricEvaluation):
    """
    Class for temporal evaluation of downscaling results.
    """
    def __init__(self, varname: str, plt_dir: str, model_info: dict, avg_dims: List[str] = ["rlat", "rlon"], eval_dict: Dict = None, 
                 proj = ccrs.PlateCarree()):
        super().__init__(varname, plt_dir, model_info, avg_dims, eval_dict)

    def __call__(self, data_fcst: xr.DataArray, data_ref: xr.DataArray, **plt_kwargs):
        
        # get score engine
        score_engine = Scores(data_fcst, data_ref, self.avg_dims)

        # run evaluation for each metric
        for metric, metric_config in self.evaluation_dict.items():
            # fss needs a dedicated loop for multiple threshold inputs
            if metric == "fss":
                if isinstance(metric_config["thres"], list):
                    thres_list = metric_config.pop("thres")
                    for thres in thres_list:
                        _ = run_evaluation_time(score_engine, metric, plot_dir=self.plt_dir, thres=thres, **metric_config, **self.model_info, **plt_kwargs)
                    continue
            _ = run_evaluation_time(score_engine, metric, plot_dir=self.plt_dir, **metric_config, **self.model_info, **plt_kwargs)

    def get_default_config(self, eval_dict):
        """
        Get default configuration for known variables.
        If the variable for evaluation is unknown, eval_dict cannot be None.
        :param eval_dict: Custom configuration dictionary. Can be None for known variables.
        """
        if self.varname == "t2m":
            eval_dict = {"rmse": {"score_unit": "K", "value_range": (0., 3.), "ref_line": None, "relative": False}, 
                         "bias": {"score_unit": "K", "value_range": (-1., 1.), "ref_line": 0, "relative": False},
                         "grad_amplitude": {"score_unit": "1", "value_range": (0.7, 1.1), "ref_line": 1.},
                         "me_std": {"score_unit": "K", "value_range": (0.1, 0.3), "ref_line": None},
                         "ralsd": {"score_unit": "dB", "value_range": (0., 5.), "ref_line": None},
                        }
        elif self.varname == "wind":
            eval_dict = {"rmse": {"score_unit": "m/s", "value_range": (0., 3.), "ref_line": None, "relative": False}, 
                         "bias": {"score_unit": "m/s", "value_range": (-1., 1.), "ref_line": 0, "relative": False},
                         "grad_amplitude": {"score_unit": "1", "value_range": (0.7, 1.1), "ref_line": 1.},
                         "me_std": {"score_unit": "m/s", "value_range": (0.1, 0.3), "ref_line": None},
                        }
        elif self.varname == "irradiance":
            eval_dict = {"rmse": {"score_unit": "W/m^2", "value_range": (0., 3.), "ref_line": None, "relative": False}, 
                         "bias": {"score_unit": "W/m^2", "value_range": (-1., 1.), "ref_line": 0, "relative": False},
                         "grad_amplitude": {"score_unit": "1", "value_range": (0.7, 1.1), "ref_line": 1.},
                         "me_std": {"score_unit": "W/m^2", "value_range": (0.1, 0.3), "ref_line": None},
                         "fss": {"score_unit": "1", "value_range": (0, 1.), "ref_line": 0.5, "window": (4, 4), "thres": [50, 100, 300, 500]},
                        }
        else:
            if eval_dict is None:
                raise ValueError(f"No default configuration available for variable {self.varname}. " + \
                                 "Parse custom eval_dict.")

        return eval_dict
    
    def required_config_keys(self):
        return ["score_unit", "value_range", "ref_line"]
    

class SpatialEvaluation(AbstractMetricEvaluation):
    """
    Class for spatial evaluation of downscaling results.
    """
    def __init__(self, varname: str, plt_dir: str, model_info: dict, proj, spatial_dims = ["rlat", "rlon"], avg_dims: List[str] = [], eval_dict: Dict = None):
        super().__init__(varname, plt_dir, model_info, avg_dims, eval_dict)

        self.spatial_dims = spatial_dims
        self.proj = proj
        
    def __call__(self, data_fcst: xr.DataArray, data_ref: xr.DataArray, **plt_kwargs):
        
        # get score engine
        score_engine = Scores(data_fcst, data_ref, self.avg_dims)

        # run evaluation for each metric
        for metric, metric_config in self.evaluation_dict.items():
            _ = run_evaluation_spatial(score_engine, metric, plot_dir=os.path.join(self.plt_dir, f"{metric}_spatial"), 
                                       dims=self.spatial_dims, projection=self.proj, **self.model_info, 
                                       **metric_config, **plt_kwargs)

    def get_default_config(self, eval_dict):
        """
        Get default configuration for known variables.
        If the variable for evaluation is unknown, eval_dict cannot be None.
        :param eval_dict: Custom configuration dictionary. Can be None for known variables.
        """
        if self.varname in ["t2m", "wind", "global_rad"]:
            lvl_bias = np.arange(-2, 2.1, .1)
            lvl_rmse =  np.arange(0., 3.1, 0.2)
            eval_dict = {"rmse": {"levels": lvl_rmse, "cmap_name": "afmhot_r", "relative": False}, 
                         "bias": {"levels": lvl_bias, "cmap_name": "seismic", "relative": False}}
        else:
            if eval_dict is None:
                raise ValueError(f"No default configuration available for variable {self.varname}. " + \
                                 "Parse custom eval_dict.")

        return eval_dict
    
    def required_config_keys(self):
        return ["levels", "cmap_name"]
