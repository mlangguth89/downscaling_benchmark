# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference on trained downscaling models for Destine-AQ performance analysis deliverable
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2023-10-27"

import os, sys, glob
import logging
import argparse
from timeit import default_timer as timer
import json as js
from datetime import datetime as dt
import gc
import pandas as pd
import xarray as xr
import tensorflow.keras as keras
import cartopy.crs as ccrs
import multiprocessing as mp
from multiprocessing.pool import Pool
import matplotlib as mpl
# auxiliary functions
from handle_data_unet import *
from handle_data_class import HandleDataClass
from all_normalizations import ZScore
from plotting import create_map_score
from postprocess import get_model_info 
from model_utils import convert_to_xarray

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

    # get datafile list
    fdata_test_list = glob.glob(os.path.join(parser_args.data_dir, "downscaling_{parser_args.dataset}*.nc"))
    nfiles = len(fdata_test_list)
    if nfiles == 0:
        raise FileNotFoundError(f"Could not find any datafile for dataset '{parser_args.dataset}' " +
                                f"under '{parser_args.data_dir}'")

    # prepare normalization
    js_norm = os.path.join(norm_dir, "norm.json")
    logger.debug("Read normalization file for subsequent data transformation.")
    norm = ZScore(ds_dict["norm_dims"])
    norm.read_norm_from_file(js_norm)

    tar_varname = ds_dict["predictands"][0]
    logger.info(f"Variable {tar_varname} serves as ground truth data.")

    # initialize lists for timing
    times_read = []
    times_preproc = []
    times_infer = []
    times_postproc = []
    times_plot = []

    for fdata_test in fdata_test_list:
        # read test data
        t0_read = timer()
        logger.info(f"Start reading data from file '{fdata_test}'")

        with xr.open_dataset(fdata_test) as ds_test:
            ds_test = norm.normalize(ds_test)
        
        # track reading time
        dt_read = timer() - t0_read 
        times_read.append(dt_read)
        logger.info(f"Data from file '{fdata_test}' successfully read in {dt_read:.2f}s.")            

        # prepare training and validation data
        logger.info(f"Start preparing test dataset...")
        t0_preproc = timer()

        da_test = HandleDataClass.reshape_ds(ds_test).astype("float32", copy=True)
    
        # clean-up to reduce memory footprint
        del ds_test
        gc.collect()

        # instantiate tf.data.Dataset object
        tfds_opts = {"batch_size": ds_dict["batch_size"], "predictands": ds_dict["predictands"], "predictors": ds_dict.get("predictors", None),
                    "lshuffle": False, "var_tar2in": ds_dict["var_tar2in"], "named_targets": named_targets, "lrepeat": False, "drop_remainder": False}    

        tfds_test = HandleDataClass.make_tf_dataset_allmem(da_test, **tfds_opts)
    
        predictors = ds_dict.get("predictors", None)
        if predictors is None:
            predictors = [var for var in list(da_test["variables"].values) if var.endswith("_in")]
            if ds_dict.get("var_tar2in", False): predictors.append(ds_dict["var_tar2in"])

        # track preparation time
        dt_preproc = timer() -t0_preproc
        times_preproc.append(dt_preproc)

        logger.info(f"Preparation of test dataset finished after {timer() - t0_preproc:.2f}s. " +
                    "Start inference on trained model...")

        # start inference
        t0_train = timer()
        y_pred = trained_model.predict(tfds_test, verbose=2)
    
        # clean-up to reduce memory footprint
        del tfds_test
        gc.collect()

        # track inference time
        dt_infer = timer() - t0_train
        times_infer.append(dt_infer)
        logger.info(f"Inference on test dataset finished after {dt_infer:.2f}s.")

        # start postprocessing
        logger.info(f"Start postprocessing of inference data...")
        t0_postproc = timer()
    
        # convert to xarray
        y_pred = convert_to_xarray(y_pred, norm, tar_varname, da_test.sel({"variables": tar_varname}).squeeze().coords,
                                   da_test.sel({"variables": tar_varname}).squeeze().dims, hparams_dict["z_branch"])

        # write inference data to netCDf
        logger.info(f"Write inference data to netCDF-file '{ncfile_out}'")
        y_pred.name = f"{tar_varname}_fcst"
        ds = y_pred.to_dataset()
        ds.to_netcdf(ncfile_out)

        # track postprocessing time
        dt_postproc = timer() - t0_postproc
        times_postproc.append(dt_postproc)
        logger.info(f"Writing postprocessed output data in {dt_postproc:.2f}s. Start producing maps of downscaled product...")

        # start creating plots
        t0_plot = timer()

        # set plot arguments
        proj=ccrs.RotatedPole(pole_longitude=-162.0, pole_latitude=39.25)
        lvl_t2m = np.arange(-34, 46, 2.)
        cmap_t2m = mpl.hsv_r(np.linspace(0., 1., len(lvl_t2m)))
        plt_kwargs = {"dims": ds_dict["norm_dims"][1::], "cmap": cmap_t2m, "levels": lvl_t2m,
                      "projection": proj}
        
        # Create a pool of processes
        nworkers = min(mp.cpu_count(), len(y_pred["time"]))
        pool = Pool(processes=nworkers)

        for it, t in enumerate(y_pred["time"]):
            date = pd.to_datetime(t)
            fname = os.path.join(plt_dir, f"{model_type}_downscaled_{tar_varname}_{date.strftime('%Y%m%d_%h00')}")
            pool.apply_async(create_map_score, (y_pred.isel({"time": it}), fname, title=f"{date.strftime('%Y-%m%-%d %h:00 UTC')}",
                                                **plt_kwargs))
            
        pool.close()
        pool.join()
        
        # track plotting time
        dt_plot = timer() - t0_plot
        times_plot.append(dt_plot)
        logger.info(f"Creatimg maps of downscaled product took {dt_plot:.2f}s.")

    # finalize postprocessing and save tracked times to JSON-file
    dt_all = timer() - t0
    logger.info(f"Postprocessing of experiment '{parser_args.exp_name}' finished. " +
                f"Elapsed total time: {dt_all:.1f}s.")

    time_dict = {"reading times": times_read, "preprocessing times": times_preproc, "inference times": times_infer,
                 "postprocessing times": times_postproc, "plotting times": times_plot, "total run time": dt_all}
    

    times_js = os.path.join(plt_dir, "tracked_times.json")
    logger.info(f"Save tracked times to {times_js}")
    with open(times_js) as jsf:
        jsf.dump(time_dict)
    


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
