# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference on trained downscaling models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2023-08-21"

import os, sys, glob
import logging
import argparse
from timeit import default_timer as timer
import json as js
from datetime import datetime as dt
import gc
import numpy as np
import xarray as xr
import tensorflow.keras as keras
import matplotlib as mpl
import cartopy.crs as ccrs
from handle_data_unet import *
from handle_data_class import HandleDataClass, get_dataset_filename
from all_normalizations import ZScore
from statistical_evaluation import Scores
from postprocess import get_model_info, run_evaluation_time, run_evaluation_spatial, run_feature_importance
from model_utils import convert_to_xarray
#from other_utils import free_mem

# get logger
logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')


def main(parser_args):

    t0 = timer()
    # construct model directory path and infer model type
    model_base = os.path.join(parser_args.model_base_dir, parser_args.exp_name)

    model_dir, plt_dir, norm_dir, model_type = get_model_info(model_base, parser_args.output_base_dir,
                                                              parser_args.exp_name, parser_args.last,
                                                              parser_args.model_type)

    # create output-directory and set name of netCDF-file to store inference data
    os.makedirs(plt_dir, exist_ok=True)
    ncfile_out = os.path.join(plt_dir, "postprocessed_ds_test.nc")
    # create logger handlers
    logfile = os.path.join(plt_dir, f"postprocessing_{parser_args.exp_name}.log")
    if os.path.isfile(logfile): os.remove(logfile)
    fh = logging.FileHandler(logfile)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    fh.setLevel(logging.INFO)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh), logger.addHandler(ch)
    
    #logger.info(f"Start postprocessing at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Start postprocessing at...")

    # read configuration files
    md_config_pattern, ds_config_pattern = f"config_{model_type}.json", f"config_ds_{parser_args.dataset}.json"
    md_config_file, ds_config_file = glob.glob(os.path.join(model_base, md_config_pattern)), \
                                     glob.glob(os.path.join(model_base, ds_config_pattern))
    if not ds_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for dataset '{ds_config_pattern}' " +
                                f"under '{model_base}'")
    else:
        with open(ds_config_file[0]) as dsf:
            logger.info(f"Read dataset configuration file '{ds_config_file[0]}'.")
            ds_dict = js.load(dsf)
            logger.debug(ds_dict)

    if not md_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for model '{md_config_pattern}' " +
                                f"under '{model_base}'")
    else:
        with open(md_config_file[0]) as mdf:
            logger.info(f"Read model configuration file '{md_config_file[0]}'.")
            hparams_dict = js.load(mdf)
            logger.debug(hparams_dict)

    named_targets = hparams_dict.get("named_targets", False)

    # Load checkpointed model
    logger.info(f"Load model '{parser_args.exp_name}' from {model_dir}")
    trained_model = keras.models.load_model(model_dir, compile=False)
    logger.info(f"Model was loaded successfully.")

    # get datafile and read netCDF
    fdata_test = get_dataset_filename(parser_args.data_dir, parser_args.dataset, "test",
                                      ds_dict.get("laugmented", False))

    logger.info(f"Start opening datafile {fdata_test}...")
    # prepare normalization
    js_norm = os.path.join(norm_dir, "norm.json")
    logger.debug("Read normalization file for subsequent data transformation.")
    norm = ZScore(ds_dict["norm_dims"])
    norm.read_norm_from_file(js_norm)

    tar_varname = ds_dict["predictands"][0]
    logger.info(f"Variable {tar_varname} serves as ground truth data.")

    with xr.open_dataset(fdata_test) as ds_test:
        ground_truth = ds_test[tar_varname].astype("float32", copy=True)
        ds_test = norm.normalize(ds_test)

    # prepare training and validation data
    logger.info(f"Start preparing test dataset...")
    t0_preproc = timer()

    da_test = HandleDataClass.reshape_ds(ds_test).astype("float32", copy=True)
    
    # clean-up to reduce memory footprint
    del ds_test
    gc.collect()
    #free_mem([ds_test])

    tfds_opts = {"batch_size": ds_dict["batch_size"], "predictands": ds_dict["predictands"], "predictors": ds_dict.get("predictors", None),
                "lshuffle": False, "var_tar2in": ds_dict["var_tar2in"], "named_targets": named_targets, "lrepeat": False, "drop_remainder": False}    

    tfds_test = HandleDataClass.make_tf_dataset_allmem(da_test, **tfds_opts)
    
    predictors = ds_dict.get("predictors", None)
    if predictors is None:
        predictors = [var for var in list(da_test["variables"].values) if var.endswith("_in")]
        if ds_dict.get("var_tar2in", False): predictors.append(ds_dict["var_tar2in"])

    # start inference
    logger.info(f"Preparation of test dataset finished after {timer() - t0_preproc:.2f}s. " +
                 "Start inference on trained model...")
    t0_train = timer()
    y_pred = trained_model.predict(tfds_test, verbose=2)

    logger.info(f"Inference on test dataset finished. Start denormalization of output data...")
    
    # clean-up to reduce memory footprint
    del tfds_test
    gc.collect()
    #free_mem([tfds_test])

    # convert to xarray
    y_pred = convert_to_xarray(y_pred, norm, tar_varname, da_test.sel({"variables": tar_varname}).squeeze().coords,
                               da_test.sel({"variables": tar_varname}).squeeze().dims, hparams_dict["z_branch"])

    # write inference data to netCDf
    logger.info(f"Write inference data to netCDF-file '{ncfile_out}'")
    ground_truth.name, y_pred.name = f"{tar_varname}_ref", f"{tar_varname}_fcst"
    ds = xr.Dataset(xr.Dataset.merge(y_pred.to_dataset(), ground_truth.to_dataset()))
    ds.to_netcdf(ncfile_out)

    # start evaluation
    logger.info(f"Output data on test dataset successfully processed in {timer()-t0_train:.2f}s. Start evaluation...")

    # instantiate score engine for time evaluation (i.e. hourly time series of evalutaion metrics)
    score_engine = Scores(y_pred, ground_truth, ds_dict["norm_dims"][1:])

    logger.info("Start temporal evaluation...")
    t0_tplot = timer()
    rmse_all = run_evaluation_time(score_engine, "rmse", "K", plt_dir, value_range=(0., 3.), model_type=model_type)
    _ = run_evaluation_time(score_engine, "bias", "K", plt_dir, value_range=(-1., 1.), ref_line=0.,
                            model_type=model_type)
    _ = run_evaluation_time(score_engine, "grad_amplitude", "1", plt_dir, value_range=(0.7, 1.1),
                            ref_line=1., model_type=model_type)

    logger.info(f"Temporal evalutaion finished in {timer() - t0_tplot:.2f}s.")

    # run feature importance analysis for RMSE
    logger.info("Start feature importance analysis...")
    t0_fi = timer()

    rmse_ref = rmse_all.mean().values

    _ = run_feature_importance(da_test, predictors, tar_varname, trained_model, norm, "rmse", rmse_ref,
                               tfds_opts, plt_dir, patch_size=(6, 6), variable_dim="variables")
    
    logger.info(f"Feature importance analysis finished in {timer() - t0_fi:.2f}s.")
    
    # clean-up to reduce memory footprint
    del da_test
    gc.collect()
    #free_mem([da_test])

    # instantiate score engine with retained spatial dimensions
    score_engine = Scores(y_pred, ground_truth, [])

    # ad-hoc adaption to projection basaed on norm_dims
    if "rlat" in ds_dict["norm_dims"]:
        proj=ccrs.RotatedPole(pole_longitude=-162.0, pole_latitude=39.25)
    else:
        proj=ccrs.PlateCarree()

    logger.info("Start spatial evaluation...")
    lvl_rmse = np.arange(0., 3.1, 0.2)
    cmap_rmse = mpl.cm.afmhot_r(np.linspace(0., 1., len(lvl_rmse)))
    _ = run_evaluation_spatial(score_engine, "rmse", os.path.join(plt_dir, "rmse_spatial"), 
                               dims=ds_dict["norm_dims"][1::], cmap=cmap_rmse, levels=lvl_rmse,
                               projection=proj)

    lvl_bias = np.arange(-2., 2.1, 0.1)
    cmap_bias = mpl.cm.seismic(np.linspace(0., 1., len(lvl_bias)))
    _ = run_evaluation_spatial(score_engine, "bias", os.path.join(plt_dir, "bias_spatial"), 
                               dims=ds_dict["norm_dims"][1::], cmap=cmap_bias, levels=lvl_bias,
                               projection=proj)

    logger.info(f"Spatial evalutaion finished in {timer() - t0_tplot:.2f}s.")

    logger.info(f"Postprocessing of experiment '{parser_args.exp_name}' finished. " +
                f"Elapsed total time: {timer() - t0:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", "-data_dir", dest="data_dir", type=str, required=True,
                        help="Directory where test dataset (netCDF-file) is stored.")
    parser.add_argument("--output_base_directory", "-output_base_dir", dest="output_base_dir", type=str, required=True,
                        help="Directory where results in form of plots are stored.")
    parser.add_argument("--model_base_directory", "-model_base_dir", dest="model_base_dir", type=str, required=True,
                        help="Base directory where trained models are saved.")
    parser.add_argument("--experiment_name", "-exp_name", dest="exp_name", type=str, required=True,
                        help="Name of the experiment/trained model to postprocess.")
    parser.add_argument("--downscaling_dataset", "-dataset", dest="dataset", type=str, required=True,
                        help="Name of dataset to be used for downscaling model.")
    parser.add_argument("--evaluate_last", "-last", dest="last", default=False, action="store_true",
                        help="Flag for evaluating last instead of best checkpointed model")
    parser.add_argument("--model_type", "-model_type", dest="model_type", default=None,
                        help="Name of model architecture. Only required if custom model architecture is not" +
                             "implemented in get_model_info-function (see postprocess.py)")

    args = parser.parse_args()
    main(args)
