# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to train downscaling models
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-10-06"
__update__ = "2022-10-06"

import os
import argparse
from datetime import datetime as dt

print("Start with importing packages at {0}".format(dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")))
import gc
import json as js
from timeit import default_timer as timer
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.utils.layer_utils import count_params
from model_utils import get_model
from handle_data_class import get_dataset_filename
from benchmark_utils import BenchmarkCSV, get_training_time_dict


# Open issues:
# * customized input file (= data engine)
# * customized choice on predictors and predictands
# * harmonize and merge with main_train_unet.py

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

    # get model object or tuple of model objects for composite models
    model_obj = get_model(parser_args.model)
    fdata_train, fdata_val = get_dataset_filename(datadir, dataset, "training"), \
                             get_dataset_filename(datadir, dataset, "validation")

    # initialize benchmarking object
    bm_obj = BenchmarkCSV(os.path.join(os.getcwd(), f"benchmark_training_{parser_args.model}.csv"))
    print("Start reading data from disk...")
    t0_save = timer()
    ds_train, ds_val = xr.open_dataset(fdata_train), xr.open_dataset(fdata_val)

    benchmark_dict = {"loading data time": timer() - t0_save}

    if callable(model_obj):
        model_func = model_obj[0]
        func_args = (*model_obj[1:], parser_args.exp_name, model_savedir, hparams_dict)
    else:
        model_func = model_obj
        func_args = (parser_args.exp_name, model_savedir, hparams_dict)

    model = model_func(*func_args)











    os.makedirs(model_savedir, exist_ok=True)
    # Read training and validation data


    # handle (hyper-) parameters
    keys_remove = ["input_dir", "output_dir", "id", "no_z_branch", "no_z_tar2in", "model_name", "no_supervision"]
    args_dict = {k: v for k, v in vars(parser_args).items() if (v is not None) & (k not in keys_remove)}
    args_dict["z_branch"] = not parser_args.no_z_branch
    if not parser_args.no_z_tar2in:  # add high-resolved surface topography to preditor variable list
        args_dict["var_tar2in"] = "hsurf_tar"
    supervise = not parser_args.no_supervision
    # set critic learning rate equal to generator if not supplied
    if not "lr_critic": args_dict["lr_critic"] = args_dict["lr_gen"]

    # instantiate model
    wgan_model = WGAN(build_unet, critic_model, args_dict, model_name, supervise, model_savedir)

    # prepare training and validation data
    t0_preproc = timer()

    # slice data temporaly
    # ds_train = ds_train.sel(time=slice("2014-01-01", "2016-12-30"))
    da_train, da_val = HandleUnetData.reshape_ds(ds_train), HandleUnetData.reshape_ds(ds_val)

    norm_dims = ["time", "rlat", "rlon"]
    da_train, mu_train, std_train = HandleUnetData.z_norm_data(da_train, dims=norm_dims,
                                                               save_path=os.path.join(outdir, parser_args.model_name),
                                                               return_stat=True)
    da_val = HandleUnetData.z_norm_data(da_val, mu=mu_train, std=std_train)

    # clean up to save some memory
    del ds_train
    del ds_val
    gc.collect()

    t0_compile = timer()
    benchmark_dict["preprocessing data time"] = t0_compile - t0_preproc

    # compile model and get TF datasets for training and validation
    print("Start compiling WGAN-model.")
    train_iter, val_iter = wgan_model.compile(da_train.astype(np.float32), da_val.astype(np.float32))

    benchmark_dict["model compile time"] = timer() - t0_compile

    # train model
    # define class for creating timer callback
    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.epoch_times = []

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = timer()

        def on_epoch_end(self, epoch, logs={}):
            self.epoch_times.append(timer() - self.epoch_time_start)

    class LRLogger(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._supports_tf_logs = True

        def on_train_begin(self, logs={}):
            self.lr_gen_logs = []
            self.lr_critic_logs = []

        def on_epoch_begin(self, epoch, logs={}):
            self.lr_gen_logs.append(self.model.g_optimizer._decayed_lr(tf.float32))
            self.lr_critic_logs.append(self.model.c_optimizer._decayed_lr(tf.float32))

    # create callbacks for scheduling learning rate and for timing training process
    time_tracker, lr_tracker = TimeHistory(), LRLogger()
    callback_list = [time_tracker, lr_tracker]

    print("Start training of WGAN...")
    history = wgan_model.fit(train_iter, val_iter, callbacks=callback_list)

    # get some parameters from tracked training times and put to dictionary
    training_times = get_training_time_dict(time_tracker.epoch_times,
                                            wgan_model.nsamples * wgan_model.hparams["train_epochs"])
    benchmark_dict = {**benchmark_dict, **training_times}

    print("WGAN training finished. Save model to '{0}' and start creating example plot."
          .format(os.path.join(outdir, parser_args.model_name)))
    # save trained model (generator and critic are saved seperately)
    t0_save = timer()

    wgan_model.generator.save(os.path.join(model_savedir, "{0}_generator_last".format(parser_args.model_name)))
    wgan_model.critic.save(os.path.join(model_savedir, "{0}_critic_last".format(parser_args.model_name)))

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
        stat_info = {"static_model_info":
                         {"trainable_parameters_generator": count_params(wgan_model.generator.trainable_weights),
                          "non-trainable_parameters_generator": count_params(
                              wgan_model.generator.non_trainable_weights),
                          "trainable_parameters_critc": count_params(wgan_model.critic.trainable_weights),
                          "non-trainable_parameters_critic": count_params(wgan_model.critic.non_trainable_weights)},
                     "data_info": {"training data size": da_train.nbytes, "validation data size": da_val.nbytes,
                                   "nsamples": wgan_model.nsamples, "shape_samples": wgan_model.shape_in,
                                   "batch_size": wgan_model.hparams["batch_size"]}}

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