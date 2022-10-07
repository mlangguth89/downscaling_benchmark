# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to train downscaling models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-10-06"
__update__ = "2022-10-07"

import os
import argparse
from datetime import datetime as dt
print("Start with importing packages at {0}".format(dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")))
import gc
import json as js
from timeit import default_timer as timer
import xarray as xr
import tensorflow.keras as keras
from model_utils import ModelEngine
from handle_data_class import HandleDataClass, get_dataset_filename
from all_normalizations import ZScore
from benchmark_utils import BenchmarkCSV, get_training_time_dict


# Open issues:
# * customized choice on predictors and predictands

def main(parser_args):
    # start timing
    t0 = timer()

    # initialize benchmarking object
    bm_obj = BenchmarkCSV(os.path.join(os.getcwd(), "benchmark_training_wgan.csv"))

    # Get some basic directories and flags
    datadir = parser_args.input_dir
    outdir = parser_args.output_dir
    job_id = parser_args.id
    dataset = parser_args.dataset_name.lower()

    # initialize checkpoint-directory path for saving the model
    model_savedir = os.path.join(outdir, parser_args.exp_name)

    # read configuration files
    try:
        with open(parser_args.conf_ds, "r") as dsf:
            ds_dict = js.load(dsf)
    except Exception as err:
        raise err

    try:
        with open(parser_args.conf_md, "r") as mdf:
            hparams_dict = js.load(mdf)
    except Exception as err:
        raise err

    # get model instance and path to data files
    model_instance = ModelEngine(parser_args.model_name)
    fdata_train, fdata_val = get_dataset_filename(datadir, dataset, "training"), \
                             get_dataset_filename(datadir, dataset, "validation")

    # initialize benchmarking object
    bm_obj = BenchmarkCSV(os.path.join(os.getcwd(), f"benchmark_training_{parser_args.model}.csv"))

    # Read data from disk and preprocess (normalization and conversion to TF dataset)
    print("Start reading data from disk...")
    t0_save = timer()
    ds_train, ds_val = xr.open_dataset(fdata_train), xr.open_dataset(fdata_val)

    benchmark_dict = {"loading data time": timer() - t0_save}

    # prepare training and validation data
    t0_preproc = timer()

    da_train, da_val = HandleDataClass.reshape_ds(ds_train), HandleDataClass.reshape_ds(ds_val)

    data_norm = ZScore(ds_dict["norm_dims"])
    da_train = data_norm.normalize(da_train)
    da_val = data_norm.normalize(da_val)
    data_norm.save_norm_to_file(os.path.join(model_savedir, "norm.json"))

    bs_train = ds_dict["batch_size"] * ds_dict["d_steps"] if "d_steps" in ds_dict else ds_dict["batch_size"]
    tfds_train = HandleDataClass.make_tf_dataset(da_train, bs_train, var_tar2in=ds_dict["var_tar2in"])
    tfds_val = HandleDataClass.make_tf_dataset(da_val, ds_dict["batch_size"], lshuffle=False,
                                               var_tar2in=ds_dict["var_tar2in"])

    nsamples = da_train.shape[0]
    shape_in = tfds_train.element_spec[0].shape[1:]

    # clean up to save some memory
    del ds_train
    del ds_val
    gc.collect()

    t0_compile = timer()
    benchmark_dict["preprocessing data time"] = t0_compile - t0_preproc

    # instantiate and compile model
    model = model_instance(hparams_dict, parser_args.exp_name, model_savedir)

    func_compile_opt = getattr(model, "get_compile_options", None)
    if callable(func_compile_opt):
        compile_opts = func_compile_opt()
    else:
        compile_opts = {}

    model.compile(shape_in, **compile_opts)

    # train model
    # define class for creating timer callback
    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.epoch_times = []

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = timer()

        def on_epoch_end(self, epoch, logs={}):
            self.epoch_times.append(timer() - self.epoch_time_start)

    time_tracker = TimeHistory()
    callback_list = [time_tracker]

    # retrieve customize fit-options if required
    func_fit_opt = getattr(model, "get_fit_options", None)
    if callable(func_fit_opt):
        fit_opts = func_fit_opt()
    else:
        fit_opts = {}

    print("Start training of WGAN...")
    history = model.fit(tfds_train, tfds_val, callbacks=callback_list, **fit_opts)

    # get some parameters from tracked training times and put to dictionary
    training_times = get_training_time_dict(time_tracker.epoch_times,
                                            model.nsamples * hparams_dict["train_epochs"])
    benchmark_dict = {**benchmark_dict, **training_times}

    print(f"Training of model '{parser_args.model_name}' training finished. Save model to '{model_savedir}'")

    # save trained model
    t0_save = timer()

    os.makedirs(model_savedir, exist_ok=True)
    model.save(model_savedir)

    tend = timer()
    benchmark_dict["saving model time"] = tend - t0_save

    # finally, track total runtime...
    benchmark_dict["total runtime"] = tend - t0
    benchmark_dict["job id"] = job_id
    # currently untracked variables
    benchmark_dict["#nodes"], benchmark_dict["#cpus"], benchmark_dict["#gpus"] = None, None, None
    benchmark_dict["#mpi tasks"], benchmark_dict["node id"], benchmark_dict["max. gpu power"] = None, None, None
    benchmark_dict["gpu energy consumption"] = None
    benchmark_dict["final training loss"] = -999.
    benchmark_dict["final validation loss"] = -999.
    # ... and save CSV-file with tracked data on disk
    bm_obj.populate_csv_from_dict(benchmark_dict)

    js_file = os.path.join(model_savedir, "benchmark_training_static.json")
    if not os.path.isfile(js_file):
        func_model_info = getattr(model, "get_model_info", None)
        if callable(func_model_info):
            model_info = func_model_info()
        else:
            model_info = {}
        stat_info = {"static_model_info": model_info,
                     "data_info": {"training data size": da_train.nbytes, "validation data size": da_val.nbytes,
                                   "nsamples": nsamples, "shape_samples": shape_in,
                                   "batch_size": hparams_dict["batch_size"]}}

        with open(js_file, "w") as jsf:
            js.dump(stat_info, jsf)

    print("Finished job at {0}".format(dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-in", dest="input_dir", type=str, required=True,
                        help="Directory where input netCDF-files are stored.")
    parser.add_argument("--output_dir", "-out", dest="output_dir", type=str, required=True,
                        help="Output directory where model is savded.")
    parser.add_argument("--downscaling_model", "-model", dest="model", type=str, default="U-Net",
                        help="Downscaling model to train.")
    parser.add_argument("--job_id", "-id", dest="id", type=int, required=True, help="Job-id from Slurm.")
    parser.add_argument("--experiment_name", "-exp_name", dest="exp_name", type=str, required=True,
                        help="Name for the current experiment.")

    args = parser.parse_args()
    main(args)