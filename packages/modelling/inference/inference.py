# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

"""
Contains all methods and classes used in main_evaluation.py.
"""

__author__ = "Sebastian Lehner, Michael Langguth"
__email__ = "sebastian.lehner@geosphere.at, m.langguth@fz-juelich.de"
__date__ = "2025-10-03"
__update__ = "2025-10-03"

from typing import Union, List
import glob
import os
import json as js
from pathlib import Path
from timeit import default_timer as timer
import logging
import gc
import numpy as np
import xarray as xr
from all_normalizations import GeneralNormalizer
from handle_data_class import prepare_dataset, make_tf_dataset_allmem
from model_engine import ModelEngine
from other_utils import finditem, convert_to_xarray, check_str_in_list, sample_permut_xyt
from inference_plotting import create_box_plot
from inference_scores_class import InferenceScores

# auxiliary variable for logger
logger_module_name = f"main_inference.{__name__}"
module_logger = logging.getLogger(logger_module_name)

list_or_str = Union[List[str], str]

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

    # hacky fix for Harris WGAN batch size
    if model_type == "harris_wgan":
        func_logger.info("Adjust batch size for Harris WGAN model.")
        ds_dict["batch_size"] = 36
    
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
    if tar_varname == "glob_rad_pp_ratio_tar":
        func_logger.info("Re-scale global_rad_pp_ration to global_rad_pp.")
        y_pred = y_pred * ds_test["tisr_tar"]
        tar_varname = "glob_rad_pp_tar"

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

    func_logger.info(f"Inference completed. Output data on test dataset successfully processed in {timer()-t0_infer:.2f}s.")

    return ds_out, test_info


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
    
    # hacky fix for Harris WGAN batch size
    if model_type == "harris_wgan":
        func_logger.info("Adjust batch size for Harris WGAN model.")
        hparams_dict["batch_size"] = 36

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

def run_feature_importance(ds: xr.Dataset, predictors: list_or_str, varname_tar: str, model, norm, score_name: str,
                           data_loader_opt: dict, plot_dir: str, patch_size = (6, 6), model_type: str = None):
    """
    Run feature importance analysis and create box-plot of results
    :param ds: Unnormalized xr.Dataset with predictors and target variable
    :param predictors: List of predictor names for which feature importance analysis should be run
    :param varname_tar: Name of target variable
    :param model: Model object
    :param norm: Normalization object
    :param score_name: Name of score to compute feature importance
    :param data_loader_opt: Data loader options that will be parsed to the make_tf_dataset_allmem-method
    :param plot_dir: Directory to save plot files
    :param patch_size: Patch size for feature importance analysis
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{run_feature_importance.__name__}")
    
    # sanity check predictor type
    if isinstance(predictors, dict):
        predictors = list(predictors.keys())

    # get feature importance scores
    func_logger.debug(f"Start feature importance analysis for {score_name}...")
    feature_scores = feature_importance(ds, predictors, varname_tar, model, norm, score_name, data_loader_opt, 
                                        patch_size=patch_size)
    
    # get reference score
    func_logger.debug(f"Retrieve reference score to finish feature importance analysis...")
    score_file = os.path.join(plot_dir.replace("/plots/", "/metric_files/"), f"eval_{score_name}_year.nc")
    if not os.path.exists(score_file):
        raise FileNotFoundError(f"File {score_file} not found. Run run_evaluation_time-method for score '{score_name}' first.")
    ds_score = xr.open_dataset(score_file)
    ref_score = ds_score[f"{score_name}"] 

    rel_changes = feature_scores / ref_score
    max_rel_change = int(np.ceil(np.amax(rel_changes) + 1.))

    # plot feature importance scores in a box-plot with whiskers where each variable is a box
    plt_fname = os.path.join(plot_dir, f"feature_importance_{score_name}.png")

    func_logger.debug(f"Plot feature importance-analysis results into file '{plt_fname}'.")
    create_box_plot(rel_changes.T, plt_fname, **{"title": f"Feature Importance ({score_name.upper()})", "ref_line": 1., "widths": .3, 
                                                 "xlabel": "Predictors", "ylabel": f"Rel. change {score_name.upper()}", "labels": predictors, 
                                                 "yticks": range(1, max_rel_change), "colors": "b"})

    return feature_scores


def feature_importance(ds: xr.Dataset, predictors: list_or_str, varname_tar: str, model, norm, score_name: str,
                       data_loader_opt: dict, patch_size = (8, 8), model_type: str = None):
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

    # hacky fix for Harris WGAN batch size
    if model_type == "harris_wgan":
        func_logger.info("Adjust batch size for Harris WGAN model.")
        data_loader_opt["batch_size"] = 36

    stream_mode = data_loader_opt.pop("stream_mode")
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
        tfds_test = make_tf_dataset_allmem(stream_mode, ds_copy, **data_loader_opt)

        # predict
        func_logger.info(f"Run inference with permuted sample for {var}...")
        y_pred = model.predict(tfds_test, verbose=2)

        # convert to xarray
        y_pred = convert_to_xarray(y_pred, norm, varname_tar, ground_truth.coords, ground_truth.dims, True)

        # calculate score
        func_logger.info(f"Calculate score for permuted samples of {var}...")
        score_engine = InferenceScores(y_pred, ground_truth, dims=ground_truth.dims[1::])
        score_all.loc[{"predictor": var}] = score_engine(score_name)

        #free_mem([da_copy, da_permut, tfds_test, y_pred, score_engine])

    return score_all