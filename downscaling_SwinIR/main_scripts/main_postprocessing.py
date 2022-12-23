# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference on trained downscaling models.
"""

__author__ = "Maxim Bragilovski"
__email__ = "maximbr@post.bgu.ac.il"
__date__ = "2022-12-17"
__update__ = "2022-12-17"

import os, sys, glob
sys.path.append('../')

import torch
import logging
import argparse
from timeit import default_timer as timer
import json as js
import numpy as np
import xarray as xr
# import tensorflow.keras as keras
import matplotlib as mpl
# from handle_data_unet import *
sys.path.append('../')
from postprocess.statistical_evaluation import Scores
from postprocess.postprocess import get_model_info, run_evaluation_time, run_evaluation_spatial
from postprocess.test_dataset import test_dataset
from models.network_unet import UNet as unet
from datetime import datetime as dt
from torch.utils.data import DataLoader


# get logger
logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(parser_args):
    t0 = timer()
    # construct model directory path and infer model type
    model_base = os.path.join(parser_args.model_base_dir, parser_args.exp_name)

    model_dir, plt_dir, norm_dir, model_type = get_model_info(model_base, parser_args.output_base_dir,
                                                              parser_args.exp_name, parser_args.last,
                                                              parser_args.model_type)
    # create logger handlers
    os.makedirs(plt_dir, exist_ok=True)
    logfile = os.path.join(plt_dir, f"postprocessing_{parser_args.exp_name}.log")
    if os.path.isfile(logfile): os.remove(logfile)
    fh = logging.FileHandler(logfile)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    fh.setLevel(logging.INFO)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh), logger.addHandler(ch)

    logger.info(f"Start postprocessing at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # read configuration files
    md_config_pattern, ds_config_pattern = f"config_{model_type}.json", f"config_ds_{parser_args.dataset}.json"
    md_config_file, ds_config_file = glob.glob(os.path.join(model_base, md_config_pattern)), \
                                     glob.glob(os.path.join(model_base, ds_config_pattern))
    if not ds_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for dataset '{ds_config_pattern}' " +
                                f"under '{model_dir}'")
    else:
        with open(ds_config_file[0]) as dsf:
            logger.info(f"Read dataset configuration file '{ds_config_file[0]}'.")
            ds_dict = js.load(dsf)
            logger.debug(ds_dict)

    if not md_config_file:
        raise FileNotFoundError(f"Could not find expected configuration file for model '{md_config_pattern}' " +
                                f"under '{model_dir}'")
    else:
        with open(md_config_file[0]) as mdf:
            logger.info(f"Read model configuration file '{md_config_file[0]}'.")
            hparams_dict = js.load(mdf)
            logger.debug(hparams_dict)

    # Load checkpointed model
    logger.info(f"Load model '{parser_args.exp_name}' from {model_dir}")
    model = unet(n_channels=9)
    model.to(torch.float64)
    model_path = glob.glob(os.path.join(model_dir, parser_args.model_name))[0]
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    model.to(device)
    logger.info(f"Model was loaded successfully.")

    # get datafile and read netCDF
    # fdata_test = get_dataset_filename(parser_args.data_dir, parser_args.dataset, "test",
    #                                   ds_dict.get("laugmented", False))
    fdata_test = parser_args.data_dir


    # logger.info(f"Start opening datafile {fdata_test}...")
    # ds_test = xr.open_dataset(fdata_test)

    # prepare training and validation data
    logger.info(f"Start preparing test dataset...")
    t0_preproc = timer()

    test = test_dataset(fdata_test=fdata_test, norm_dir=norm_dir, ds_dict=ds_dict, logger=logger)

    # start inference
    logger.info(f"Preparation of test dataset finished after {timer() - t0_preproc:.2f}s. " +
                "Start inference on trained model...")
    t0_train = timer()
    output = []
    train_dataloader = DataLoader(test, batch_size=32, shuffle=False)
    for i, train_data in enumerate(train_dataloader):
        print(i*32, len(test))
        batch_output = model(train_data[0])
        for tens in batch_output:
            output.append(tens)

    y_pred_trans = torch.stack(output)
    y_pred_trans = torch.permute(y_pred_trans, (0, 2, 3, 1))

    logger.info(f"Inference on test dataset finished. Start denormalization of output data...")
    # get coordinates and dimensions from target data
    coords = test.da_test_tar.isel(variables=0).squeeze().coords
    dims = test.da_test_tar.isel(variables=0).squeeze().dims
    check = y_pred_trans.detach().numpy().squeeze()

    y_pred = xr.DataArray(check, coords=coords, dims=dims)
    # perform denormalization
    y_pred = test.norm.denormalize(y_pred.squeeze(), mu=test.norm.norm_stats["mu"].sel({"variables": test.tar_varname}),
                              sigma=test.norm.norm_stats["sigma"].sel({"variables": test.tar_varname}))
    y_pred = xr.DataArray(y_pred, coords=coords, dims=dims)

    # start evaluation
    logger.info(f"Output data on test dataset successfully processed in {timer() - t0_train:.2f}s. Start evaluation...")

    # create plot directory if required
    os.makedirs(plt_dir, exist_ok=True)

    # instantiate score engine for time evaluation (i.e. hourly time series of evalutaion metrics)
    score_engine = Scores(y_pred, test.ground_truth, ds_dict["norm_dims"][1:])

    logger.info("Start temporal evaluation...")
    t0_tplot = timer()
    _ = run_evaluation_time(score_engine, "rmse", "K", plt_dir, value_range=(0., 3.), model_type=model_type)
    _ = run_evaluation_time(score_engine, "bias", "K", plt_dir, value_range=(-1., 1.), ref_line=0.,
                            model_type=model_type)
    _ = run_evaluation_time(score_engine, "grad_amplitude", "1", plt_dir, value_range=(0.7, 1.1),
                            ref_line=1., model_type=model_type)

    logger.info(f"Temporal evalutaion finished in {timer() - t0_tplot:.2f}s.")

    # instantiate score engine with retained spatial dimensions
    score_engine = Scores(y_pred, test.ground_truth, [])

    logger.info("Start spatial evaluation...")
    lvl_rmse = np.arange(0., 3.1, 0.2)
    cmap_rmse = mpl.cm.afmhot_r(np.linspace(0., 1., len(lvl_rmse)))
    _ = run_evaluation_spatial(score_engine, "rmse", os.path.join(plt_dir, "rmse_spatial"), cmap=cmap_rmse,
                               levels=lvl_rmse)

    lvl_bias = np.arange(-2., 2.1, 0.1)
    cmap_bias = mpl.cm.seismic(np.linspace(0., 1., len(lvl_bias)))
    _ = run_evaluation_spatial(score_engine, "bias", os.path.join(plt_dir, "bias_spatial"), cmap=cmap_bias,
                               levels=lvl_bias)

    logger.info(f"Spatial evalutaion finished in {timer() - t0_tplot:.2f}s.")

    logger.info(f"Postprocessing of experiment '{parser_args.exp_name}' finished. " +
                f"Elapsed total time: {timer() - t0:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", "-data_dir", dest="data_dir", type=str, required=False,
                        default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\downscaling_SwinIR\\model_base_dir\\wgan\\downscaling_tier2_train.nc",
                        help="Directory where test dataset (netCDF-file) is stored.")
    parser.add_argument("--model_name", "-model_name", dest="model_name", type=str, required=False,
                        default="generator_step735000.pth",
                        help="Directory where test dataset (netCDF-file) is stored.")
    parser.add_argument("--output_base_directory", "-output_base_dir", dest="output_base_dir", type=str, required=False,
                        default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\downscaling_ap5\\output\\",
                        help="Directory where results in form of plots are stored.")
    parser.add_argument("--model_base_directory", "-model_base_dir", dest="model_base_dir", type=str, required=False,
                        default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\downscaling_SwinIR\\model_base_dir\\",
                        help="Base directory where trained models are saved.")
    parser.add_argument("--experiment_name", "-exp_name", dest="exp_name", type=str, required=False,
                        default="wgan", # generator_step735000.pth
                        help="Name of the experiment/trained model to postprocess.")
    parser.add_argument("--downscaling_dataset", "-dataset", dest="dataset", type=str, required=False,
                        default="downscaling_tier2_train",
                        help="Name of dataset to be used for downscaling model.")
    parser.add_argument("--evaluate_last", "-last", dest="last", default=False, action="store_true",
                        help="Flag for evaluating last instead of best checkpointed model")
    parser.add_argument("--model_type", "-model_type", dest="model_type", default=None,
                        help="Name of model architecture. Only required if custom model architecture is not" +
                             "implemented in get_model_info-function (see postprocess.py)")

    args = parser.parse_args()
    main(args)
