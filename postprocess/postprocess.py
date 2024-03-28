# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Contains all methods and classes used in main_postrprocess.py.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2024-03-28"

import os
import glob
from typing import Union, List, Dict
import json as js
from timeit import default_timer as timer
import logging
import gc
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow.keras as keras
import matplotlib as mpl
import cartopy.crs as ccrs
from handle_data_class import prepare_dataset
from all_normalizations import ZScore
from model_engine import ModelEngine
from abstract_metric_evaluation_class import AbstractMetricEvaluation
from scores_class import Scores
from evaluation_utils import feature_importance, get_spectrum
from plotting import create_line_plot, create_map_score, create_box_plot, create_ps_plot
from other_utils import convert_to_xarray, finditem, to_list

# basic data types
da_or_ds = Union[xr.DataArray, xr.Dataset]
list_or_str = Union[List[str], str]

# auxiliary variable for logger
logger_module_name = f"main_postprocess.{__name__}"
module_logger = logging.getLogger(logger_module_name)

def results_from_inference(model_base_dir, exp_name, data_dir, out_dir, varname, model_type, last, dataset):

    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{results_from_inference.__name__}")

    # construct model directory path and infer model type
    model_base = os.path.join(model_base_dir, exp_name)

    model_dir, plt_dir, norm_dir, model_info = get_model_info(model_base, out_dir, exp_name, last, model_type)

    #logger.info(f"Start postprocessing at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    func_logger.info(f"Start postprocessing at...")

    # read configuration files
    md_config_pattern, ds_config_pattern = f"config_{model_info['model_type']}.json", f"config_ds_{dataset}.json"
    md_config_file, ds_config_file = glob.glob(os.path.join(model_base, md_config_pattern)), \
                                     glob.glob(os.path.join(model_base, ds_config_pattern))
    if not ds_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for dataset '{ds_config_pattern}' " +
                                f"under '{model_base}'")
    else:
        with open(ds_config_file[0]) as dsf:
            func_logger.info(f"Read dataset configuration file '{ds_config_file[0]}'.")
            ds_dict = js.load(dsf)
            func_logger.debug(ds_dict)

    if not md_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for model '{md_config_pattern}' " +
                                f"under '{model_base}'")
    else:
        with open(md_config_file[0]) as mdf:
            func_logger.info(f"Read model configuration file '{md_config_file[0]}'.")
            hparams_dict = js.load(mdf)
            func_logger.debug(hparams_dict)

    ### Run inference on trained model
    # Load checkpointed model
    func_logger.info(f"Load model '{exp_name}' from {model_dir}")
    trained_model = keras.models.load_model(model_dir, compile=False)
    func_logger.info(f"Model was loaded successfully.")

    # get normalization object and preprare test dataset
    t0_preproc = timer()
    func_logger.info(f"Start preparing test dataset...")

    # prepare normalization
    js_norm = os.path.join(norm_dir, "norm.json")
    func_logger.debug("Read normalization file for subsequent data transformation.")
    data_norm = ZScore(ds_dict["norm_dims"])
    data_norm.read_norm_from_file(js_norm)
    
    # get dataset pipeline for inference    
    tfds_test, test_info = prepare_dataset(data_dir, dataset, ds_dict, hparams_dict, "test", norm_obj=data_norm, 
                                           shuffle=False, lrepeat=False, drop_remainder=False) 
    
    # add further information to test_info (for later processing)
    test_info["ds_dict"] = ds_dict
    test_info["hparams_dict"] = hparams_dict
    test_info["trained_model"] = trained_model
    test_info["model_info"] = model_info

    # get ground truth data
    # To-Do: Enable handling of multiple target variables (e.g. wind vectors)
    tar_varname = test_info["all_predictands"][0]
    func_logger.info(f"Variable {tar_varname} serves as ground truth data.")

    # get ground truth data
    ds_test = xr.open_dataset(test_info["file"])
    coords, dims = ds_test[tar_varname].squeeze().coords, ds_test[tar_varname].squeeze().dims

    # start inference
    func_logger.info(f"Preparation of test dataset finished after {timer() - t0_preproc:.2f}s. " +
                      "Start inference on trained model...")
    t0_train = timer()
    y_pred = trained_model.predict(tfds_test, verbose=2)

    func_logger.info(f"Inference on test dataset finished. Start denormalization of output data...")
    
    # clean-up to reduce memory footprint
    del tfds_test
    gc.collect()
    #free_mem([tfds_test])

    ### Post-process results from test dataset
    # convert to xarray
    y_pred = convert_to_xarray(y_pred, data_norm, tar_varname, coords, dims, finditem(hparams_dict, "z_branch", False))

    # write inference data to netCDf
    ncfile_out = os.path.join(plt_dir, f"downscaled_{varname}_{model_info['model_type']}.nc")
    func_logger.info(f"Write inference data to netCDF-file '{ncfile_out}'")

    ds_out = xr.Dataset({f"{varname}_ref": ds_test[tar_varname].squeeze().astype("float32"), f"{varname}_fcst": y_pred}, 
                        coords=coords) 
    # add attributes such as model_type and from which model the data was generated and used ds_dict
    # This is also relevant for later processing (e,g. when doing feature importance analysis)
    ds_out.attrs["model_path"] = model_dir
    ds_out.to_netcdf(ncfile_out)

    func_logger.info(f"Output data on test dataset successfully processed in {timer()-t0_train:.2f}s. Start evaluation...")

    return ds_out, test_info

def results_from_file(nc_file, varname, model_name):

    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{results_from_file.__name__}")

    func_logger.info(f"Read results from netCDF-file '{nc_file}'...")

    ds_out = xr.open_dataset(nc_file)

    # check if dataset contains required variables
    req_varnames = [f"{varname}_ref", f"{varname}_fcst"]
    for req_var in req_varnames:
        if req_var not in ds_out.variables:
            raise ValueError(f"Variable '{req_var}' not found in dataset '{nc_file}'")

    model_info = {"model_type": model_name, "model_longname": model_name.replace(" ", "_").lower()}

    return ds_out, model_info

def get_model_info(model_base, output_base: str, exp_name: str, bool_last: bool = False, model_type: str = None):

    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{get_model_info.__name__}")

    model_name = os.path.basename(model_base)
    norm_dir = model_base

    add_str = "_last" if bool_last else "_best"

    def modelinfo_from_expname(expname):
        found_model = None

        for known_model in ModelEngine.known_models:
            if known_model in expname:
                found_model = known_model

        if not found_model: raise ValueError(f"Could not infer known model from experiment name '{expname}'")
        
        model_dummy = ModelEngine(found_model)

        nsubmodels = len(model_dummy.model) - 1
        
        return (found_model, model_dummy.model_longname, nsubmodels)

    if model_type:
        func_logger.debug(f"Get model info from parsed model type '{model_type}'")

        model_dummy = ModelEngine(model_type)
        model_info = {"model_type": model_type, "model_longname": model_dummy.model_longname,
                      "nsubmodels": len(model_dummy.model) - 1}
    else:
        func_logger.debug(f"Try to infer model info from parsed experiment name '{exp_name}'")

        model_type, model_longname, nsubmodels = modelinfo_from_expname(exp_name)
        model_info = {"model_type": model_type, "model_longname": model_longname,
                      "nsubmodels": nsubmodels}
        
    if nsubmodels == 1:
        model_dir, plt_dir = os.path.join(model_base, f"{exp_name}{add_str}"), os.path.join(output_base, model_name)
    else: 
        model_dir, plt_dir = os.path.join(model_base, f"{exp_name}{add_str}", f"{exp_name}_generator{add_str}"), \
                             os.path.join(output_base, model_name)

    return model_dir, plt_dir, norm_dir, model_info      
        

def run_feature_importance(ds: xr.DataArray, predictors: list_or_str, varname_tar: str, model, norm, score_name: str,
                           ref_score: float, data_loader_opt: dict, plt_dir: str, patch_size = (6, 6)):
    """
    Run feature importance analysis and create box-plot of results
    :param ds: Unnormalized xr.Dataset with predictors and target variable
    :param predictors: List of predictor names
    :param varname_tar: Name of target variable
    :param model: Model object
    :param norm: Normalization object
    :param score_name: Name of score to compute feature importance
    :param ref_score: Reference score to normalize feature importance scores
    :param data_loader_opt: Data loader options
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
    score_file = os.path.join(plt_dir, "metric_files", f"eval_{score_name}_year.csv")
    if not os.path.exists(score_file):
        raise FileNotFoundError(f"File {score_file} not found. Run run_evaluation_time for score '{score_name}' first.")
    score_data = pd.read_csv(score_file)
    ref_score = score_data[f"{score_name}_mean"].mean()

    rel_changes = feature_scores / ref_score
    max_rel_change = int(np.ceil(np.amax(rel_changes) + 1.))

    # plot feature importance scores in a box-plot with whiskers where each variable is a box
    plt_fname = os.path.join(plt_dir, f"feature_importance_{score_name}.png")

    func_logger.debug(f"Plot feature importance-analysis results into file '{plt_fname}'.")
    create_box_plot(rel_changes.T, plt_fname, **{"title": f"Feature Importance ({score_name.upper()})", "ref_line": 1., "widths": .3, 
                                                 "xlabel": "Predictors", "ylabel": f"Rel. change {score_name.upper()}", "labels": predictors, 
                                                 "yticks": range(1, max_rel_change), "colors": "b"})

    return feature_scores


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
    
    model_type = plt_kwargs.get("model_type", "sha_wgan")
    model_name = plt_kwargs.get("model_name", "Sha WGAN")

    func_logger.info(f"Start evaluation in terms of {score_name}")
    score_all = score_engine(score_name)

    func_logger.info(f"Globally averaged {score_name}: {score_all.mean().values:.4f} {score_unit}, " +
                     f"standard deviation: {score_all.std().values:.4f}")  
    
    score_hourly_all = score_all.groupby("time.hour")
    score_hourly_mean, score_hourly_std = score_hourly_all.mean(), score_hourly_all.std()

    # create plots
    create_line_plot(score_hourly_mean, score_hourly_std, model_name,
                     {score_name.upper(): score_unit},
                     os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}.png"), **plt_kwargs)

    func_logger.debug(f"Save hourly averaged {score_name} to {os.path.join(metric_dir, f'eval_{score_name}_year.csv')}...")
    scores_to_csv(score_hourly_mean, score_hourly_std, score_name, fname=os.path.join(metric_dir, f"eval_{score_name}_year.´csv"))

    # seasonal evaluation
    func_logger.debug("Run seasonal evaluation...")
    score_seas = score_all.groupby("time.season")

    for sea, score_sea in score_seas:
        score_sea_hh = score_sea.groupby("time.hour")
        score_sea_hh_mean, score_sea_hh_std = score_sea_hh.mean(), score_sea_hh.std()
        func_logger.info(f"Averaged {score_name} for {sea}: {score_sea.mean().values:.4f} {score_unit}, " +
                         f"standard deviation: {score_sea.std().values:.4f}")  
        
        create_line_plot(score_sea_hh_mean, score_sea_hh.std(),
                         model_name, {score_name.upper(): score_unit},
                         os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}_{sea}.png"),
                         **plt_kwargs)
        
        func_logger.debug(f"Save hourly averaged {score_name} to {os.path.join(metric_dir, f'eval_{score_name}_{sea}.csv')}...")
        scores_to_csv(score_sea_hh_mean, score_sea_hh_std, score_name, 
                      fname=os.path.join(metric_dir, f"eval_{score_name}_{sea}.csv"))
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

    model_type = plt_kwargs.get("model_type", "sha_wgan")
    model_name = plt_kwargs.get("model_name", "Sha WGAN")

    score_all = score_engine(score_name)
    #score_all = score_all.drop_vars("variables")

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
   

def run_spectral_analysis(ds: xr.Dataset, ds_vars: List[str], plt_dir: str, labels: List[str], varname: str, var_unit: str,
                          lonlat_dims: list_or_str = ["rlon", "rlat"], lcutoff: bool= True, re: float = 6371.):
    """
    Run spectral analysis for chosen variables and create power spectrum plot.
    :param ds: xarray.Dataset with input data
    :param ds_vars: List of variable names for spectral analysis
    :param plt_dir: Directory to save plot files
    :param labels: List of labels for each variable
    :param varname: Name of variable
    :param var_unit: Unit of variable
    :param lonlat_dims: Name of longitude and latitude dimensions
    :param lcutoff: Flag to apply low-pass filter
    :param re: Earth radius
    """
    func_logger = logging.getLogger(f"{logger_module_name}.{run_spectral_analysis.__name__}")

    # check if number of vairables for spectral analysis and labels are equal
    ds_vars, labels = to_list(ds_vars), to_list(labels)
    nexps = len(ds_vars)    
    assert nexps== len(labels), f"Number of variables ({nexps}) and labels ({len(labels)}) must be equal."
    assert all([var in ds.variables for var in ds_vars]), f"Some variables from {', '.join(ds_vars)} are not in dataset."

    # initialize dictionary for power spectrum
    ps_dict = {}

    # compute wave numbers based on size of input data
    nlon, nlat = ds[lonlat_dims[0]].size, ds[lonlat_dims[1]].size

    dims = ["wavenumber"]
    coord_dict = {"wavenumber": np.arange(0, np.amin(np.array([int(nlon/2), int(nlat/2)])))}

    for i, exp in enumerate(ds_vars):
        func_logger.info(f"Start spectral analysis for experiment {exp} ({i+1}/{nexps})...")

        # run spectral analysis
        ps_exp = get_spectrum(ds[exp], lonlat_dims = lonlat_dims, lcutoff= lcutoff, re=re)
        # average over all time steps and create xarray.DataArray
        da_ps_exp = xr.DataArray(ps_exp.mean(axis=0), dims=dims, coords=coord_dict, name=exp)

        # remove wavenumber 0 and append dictionary    
        ps_dict[exp] = da_ps_exp[1::]

    # create xarray.Dataset 
    ds_ps = xr.Dataset(ps_dict)

    # create plot   
    plt_fname = os.path.join(plt_dir, f"{varname}_power_spectrum.png")
    create_ps_plot(ds_ps, {varname: f"{var_unit}**2 m"}, labels, plt_fname, colors= ["navy", "green"],
                   x_coord="wavenumber")

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
            _ = run_evaluation_time(score_engine, metric, plot_dir=self.plt_dir, **metric_config, **self.model_info, **plt_kwargs)

    def get_default_config(self, eval_dict):
        """
        Get default configuration for known variables.
        If the variable for evaluation is unknown, eval_dict cannot be None.
        :param eval_dict: Custom configuration dictionary. Can be None for known variables.
        """
        if self.varname == "t2m":
            eval_dict = {"rmse": {"score_unit": "K", "value_range": (0., 3.), "ref_line": None}, 
                         "bias": {"score_unit": "K", "value_range": (-1., 1.), "ref_line": 0},
                         "grad_amplitude": {"score_unit": "1", "value_range": (0.7, 1.1), "ref_line": 1.},
                         "me_std": {"score_unit": "K", "value_range": (0.1, 0.3), "ref_line": None}}
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
        if self.varname == "t2m":
            lvl_bias = np.arange(-2, 2.1, .1)
            lvl_rmse =  np.arange(0., 3.1, 0.2)
            eval_dict = {"rmse": {"levels": lvl_rmse, "cmap": mpl.cm.afmhot_r(np.linspace(0., 1., len(lvl_rmse)))}, 
                         "bias": {"levels": lvl_bias, "cmap": mpl.cm.seismic(np.linspace(0., 1., len(lvl_bias)))}}
            
        else:
            if eval_dict is None:
                raise ValueError(f"No default configuration available for variable {self.varname}. " + \
                                 "Parse custom eval_dict.")

        return eval_dict
    
    def required_config_keys(self):
        return ["levels", "cmap"]
