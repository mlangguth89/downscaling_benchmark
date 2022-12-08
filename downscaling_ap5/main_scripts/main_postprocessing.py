# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference on trained downscaling models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2022-12-08"

import os, glob
import logging
import argparse
from timeit import default_timer as timer
import json as js
import numpy as np
import xarray as xr
import tensorflow.keras as keras
import matplotlib as mpl
from handle_data_unet import *
from handle_data_class import HandleDataClass, get_dataset_filename
from all_normalizations import ZScore
from statistical_evaluation import Scores
from postprocess import get_model_info, run_evaluation_time, run_evaluation_spatial

# get logger
logger = logging.getLogger("main_postprocessing_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname): %(message)')


def main(parser_args):

    t0 = timer()
    # construct model directory path and infer model type
    model_base = os.path.join(parser_args.model_base_dir, parser_args.exp_name)

    model_dir, plt_dir, norm_dir, model_type = get_model_info(model_base, parser_args.output_base_dir,
                                                              parser_args.exp_name, parser_args.last,
                                                              parser_args.model_type)
    # create logger handlers
    fh = logging.FileHandler(os.path.join(plt_dir, f"postprocessing_{parser_args.exp_name}.log"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh), logger.addHandler(ch)

    logger.info(f"Start postprocessing at {t0}")

    # read configuration files
    md_config_pattern, ds_config_pattern = f"config_{model_type}.json", f"config_ds_{parser_args.dataset}.json"
    md_config_file, ds_config_file = glob.glob(os.path.join(model_base, md_config_pattern)), \
                                     glob.glob(os.path.join(model_base, ds_config_pattern))
    if not ds_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for dataset '{ds_config_pattern}' " +
                                f"under '{model_dir}'")
    else:
        with ds_config_file[0] as dsf:
            logging.info(f"Read dataset configuration file '{ds_config_file[0]}'.")
            ds_dict = js.load(dsf)
            logging.debug(ds_dict)

    if not md_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for model '{md_config_pattern}' " +
                                f"under '{model_dir}'")
    else:
        with md_config_file[0] as mdf:
            logging.info(f"Read model configuration file '{md_config_file[0]}'.")
            hparams_dict = js.load(mdf)
            logging.debug(hparams_dict)

    # Load checkpointed model
    logging.info(f"Load model '{parser_args.exp_name}' from {model_dir}")
    trained_model = keras.models.load_model(model_dir, compile=False)
    logging.info(f"Model was loaded successfully.")

    # get datafile and read netCDF
    fdata_test = get_dataset_filename(parser_args.data_dir, parser_args.dataset, "test",
                                      ds_dict.get("laugmented", False))

    logging.info(f"Start opening datafile {fdata_test}...")
    ds_test = xr.open_dataset(fdata_test)

    # prepare training and validation data
    logging.info(f"Start preparing test dataset...")
    t0_preproc = timer()

    da_test = HandleDataClass.reshape_ds(ds_test.astype("float32", copy=False)),

    # perform normalization
    js_norm = os.path.join(norm_dir, "norm.json")
    logging.debug("Read normalization file for subsequent data transformation.")
    norm = ZScore(ds_dict["norm_dims"])
    norm.read_norm_from_file(js_norm)
    da_test = norm.normalize(da_test)

    da_test_in, da_test_tar = HandleDataClass.split_in_tar(da_test)
    ground_truth, tar_varname = da_test_tar.isel(variables=0), da_test_tar['variables'][0]
    logging.debug(f"Variable {tar_varname} serves as ground truth data.")

    if hparams_dict["z_branch"]:
        logging.info(f"Add high-resolved target topography to input features.")
        da_test_in = xr.concat([da_test_in, da_test_tar.isel({"variables": -1})], dim="variables")

    data_in = da_test_in.squeeze().values

    # start inference
    logging.info(f"Preparation of test dataset finished after {timer() - t0_preproc}s." +
                 "Start inference on trained model...")
    y_pred_trans = trained_model.predict(data_in, batch_size=32, verbose=2)

    logging.info(f"Inference on test dataset finished. Start denormalization of output data...")
    # get coordinates and dimensions from target data
    coords = da_test_tar.isel(variables=0).squeeze().coords
    dims = da_test_tar.isel(variables=0).squeeze().dims
    y_pred = xr.DataArray(y_pred_trans[0].squeeze(), coords=coords, dims=dims)
    # perform denormalization
    y_pred = norm.denormalize(y_pred.squeeze(), mu=norm.norm_stats["mu"].sel({"variables": tar_varname}),
                              sigma=norm.norm_stats["sigma"].sel({"variables": tar_varname}))
    y_pred = xr.DataArray(y_pred, coords=coords, dims=dims)

    # start evaluation
    logging.info(f"Output data on test dataset successfully processed in {t0 - timer()}s. Start evaluation...")

    # create plot directory if required
    os.makedirs(plt_dir, exist_ok=True)

    # instantiate score engine for time evaluation (i.e. hourly time series of evalutaion metrics)
    score_engine = Scores(y_pred, ground_truth, ds_dict["norm_dims"][1:])

    logging.info("Start temporal evaluation...")
    t0_tplot = timer()
    _ = run_evaluation_time(score_engine, "rmse", "K", plt_dir, value_range=(0., 3.), model_type=model_type)
    _ = run_evaluation_time(score_engine, "bias", "K", plt_dir, value_range=(-1., 1.), ref_line=0.,
                            model_type=model_type)
    _ = run_evaluation_time(score_engine, "grad_amplitude", "1", plt_dir, value_range=(0.7, 1.1),
                            ref_line=1., model_type=model_type)

    logging.info(f"Temporal evalutaion finished in {timer() - t0_tplot}s.")

    # instantiate score engine with retained spatial dimensions
    score_engine = Scores(y_pred, ground_truth, [])

    logging.info("Start spatial evaluation...")
    lvl_rmse = np.arange(0., 3.1, 0.2)
    cmap_rmse = mpl.cm.afmhot_r(np.linspace(0., 1., len(lvl_rmse)))
    _ = run_evaluation_spatial(score_engine, "rmse", "K", os.path.join(plt_dir, "rmse_spatial"), cmap=cmap_rmse,
                               levels=lvl_rmse)

    lvl_bias = np.arange(-2., 2.1, 0.1)
    cmap_bias = mpl.cm.seismic(np.linspace(0., 1., len(lvl_bias)))
    _ = run_evaluation_spatial(score_engine, "bias", "K", os.path.join(plt_dir, "bias_spatial"), cmap=cmap_bias,
                               levels=lvl_bias)

    logging.info(f"Temporal evalutaion finished in {timer() - t0_tplot}s.")

    logging.info(f"Postprocessing of experiment '{parser_args.exp_name}' finished." +
                 "Elapsed total time: {t0 - timer()}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", "-data_dir", dest="data_dir", type=str, required=True,
                        help="Directory where test dataset (netCDF-file) is stored.")
    parser.add_argument("--model_base_directory", "-model_base_dir", dest="model_base_dir", type=str, required=True,
                        help="Base directory where trained models are saved.")
    parser.add_argument("--experiment_name", "-exp_name", dest="exp_name", type=str, required=True,
                        help="Name of the experiment/trained model to postprocess.")
    parser.add_argument("--downscaling_dataset", "-dataset", dest="dataset", type=str, required=True,
                        help="Name of dataset to be used for downscaling model.")
    parser.add_argument("--evaluate_last", "-last", dest="last", default=False, action="store_true",
                        help="Flag for evaluating last instead of best checkpointed model")
    parser.add_argument("--model_type", "-model_type", dest="model_tyoe", default=None,
                        help="Name of model architecture. Only required if custom model architecture is not" +
                             "implemented in get_model_info-function (see postprocess.py)")

    args = parser.parse_args()
    main(args)