__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-05-31"
__update__ = "2022-06-01"

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
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.layer_utils import count_params
from unet_model import build_unet, get_lr_scheduler
from wgan_model import WGAN                  # to get helper for splitting input and target variables
from handle_data_unet import HandleUnetData
from handle_data_class import HandleDataClass
from benchmark_utils import BenchmarkCSV, get_training_time_dict


def main(parser_args):
    # start timing
    t0 = timer()

    # initialize benchmarking object
    bm_obj = BenchmarkCSV(os.path.join(os.getcwd(), "benchmark_training_unet.csv"))

    # Get some basic directories and flags
    datadir = parser_args.input_dir
    outdir = parser_args.output_dir
    job_id = parser_args.id

    # Read training and validation data
    print("Start reading data from disk...")
    t0_save = timer()
    ds_train, ds_val = xr.open_dataset(os.path.join(datadir, "preproc_era5_crea6_train.nc"), chunks="auto"), \
                       xr.open_dataset(os.path.join(datadir, "preproc_era5_crea6_val.nc"), chunks="auto")

    benchmark_dict = {"loading data time": timer() - t0_save}

    keys_remove = ["input_dir", "output_dir", "id", "no_z_branch"]
    args_dict = {k: v for k, v in vars(parser_args).items() if (v is not None) & (k not in keys_remove)}
    args_dict["z_branch"] = not parser_args.no_z_branch

    t0_preproc = timer()
    # slice data temporally to save memory
    # ds_train  = ds_train.sel(time=slice("2011-01-01", "2016-12-30"))
    
    da_train, da_val = HandleDataClass.reshape_ds(ds_train), HandleDataClass.reshape_ds(ds_val)

    norm_dims = ["time", "rlat", "rlon"]
    da_train, mu_train, std_train = HandleUnetData.z_norm_data(da_train, dims=norm_dims, return_stat=True)
    da_val = HandleUnetData.z_norm_data(da_val, mu=mu_train, std=std_train)

    del ds_train
    del ds_val
    gc.collect()
    
    t0_compile = timer()
    benchmark_dict["preprocessing data time"] = t0_compile - t0_preproc

    train_iter = HandleDataClass.make_tf_dataset(da_train, args_dict["batch_size"]) # lshuffle=False
    val_iter = HandleDataClass.make_tf_dataset(da_val, args_dict["batch_size"], lshuffle=False)
    print('train_iter: {}'.format(train_iter))

    # da_train_in, da_train_tar = WGAN.split_in_tar(da_train)
    # da_val_in, da_val_tar = WGAN.split_in_tar(da_val)

    print("da_train shape: {}".format(da_train.shape))
    nsamples = da_train.shape[0]
    shape_in = (96, 120, 9) # da_train.shape[1:] to-do: hard code

    # define class for creating timer callback
    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.epoch_times = []

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = timer()

        def on_epoch_end(self, epoch, logs={}):
            self.epoch_times.append(timer() - self.epoch_time_start)

    # create callbacks for scheduling learning rate and for timing training process
    lr_scheduler, time_tracker = get_lr_scheduler(), TimeHistory()
    callback_list = [lr_scheduler, time_tracker]

    # build, compile and train the model
    unet_model = build_unet(shape_in, z_branch=args_dict["z_branch"])
    print(unet_model.summary())
    steps_per_epoch = int(np.ceil(nsamples / args_dict["batch_size"]))
    print('steps_per_epoch: {}'.format(steps_per_epoch))
 
    if args_dict["z_branch"]:
        print("Start training with optimization on surface topography (with z_branch).")
        unet_model.compile(optimizer=Adam(args_dict["lr"]),
                           loss={"output_temp": "mae", "output_z": "mae"},
                           loss_weights={"output_temp": 1.0, "output_z": 1.0})

        history = unet_model.fit(train_iter,
                                 epochs=args_dict["train_epochs"],
                                 # batch_size=args_dict["batch_size"],
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=callback_list,
                                 validation_data=val_iter,
                                 validation_steps=3,
                                 verbose=2)

        # history = unet_model.fit(x=da_train_in.values, y={"output_temp": da_train_tar.sel(variables="t_2m_tar").values,
        #                                                   "output_z": da_train_tar.sel(variables="hsurf_tar").values},
        #                          batch_size=args_dict["batch_size"], epochs=args_dict["train_epochs"],
        #                          callbacks=callback_list,
        #                          validation_data=(da_val_in.values, {"output_temp": da_val_tar.sel(variables="t_2m_tar").values,
        #                                                              "output_z": da_val_tar.sel(variables="hsurf_tar").values}),
        #                          verbose=2)
    else:
        print("Start training without optimization on surface topography (with z_branch).")
        unet_model.compile(optimizer=Adam(learning_rate=args_dict["lr"]), loss="mae")

        history = unet_model.fit(train_iter,
                                 epochs=args_dict["train_epochs"],
                                 batch_size=args_dict["batch_size"],
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=callback_list,
                                 validation_data=val_iter,  # to-do: how to select target variables
                                 validation_steps=3,
                                 verbose=2)

        # history = unet_model.fit(x=da_train_in.values, y=da_train_tar.isel(variable=0).values,
        #                          batch_size=args_dict["batch_size"], epochs=args_dict["train_epochs"],
        #                          callbacks=callback_list,
        #                          validation_data=(da_val_in.values, da_val_tar.isel(variable=0).values),
        #                          verbose=2)

    # get some parameters from tracked training times and put to dictionary
    training_times = get_training_time_dict(time_tracker.epoch_times, nsamples*args_dict["batch_size"])
    benchmark_dict = {**benchmark_dict, **training_times}
    # also track losses
    benchmark_dict["final training loss"] = history.history["output_temp_loss"][-1]
    benchmark_dict["final validation loss"] = history.history["val_output_temp_loss"][-1]

    # save trained model
    model_name = "trained_downscaling_unet_t2m_hour_exp{0:d}".format(bm_obj.exp_number)
    print("Save trained model '{0}' to '{1}'".format(model_name, outdir))
    t0_save = timer()
    unet_model.save(os.path.join(outdir, "{0}.h5".format(model_name)), save_format="h5")
    benchmark_dict = {**benchmark_dict, "saving model time": timer() - t0_save}

    # finally, track total runtime...
    benchmark_dict["total runtime"] = timer() - t0
    benchmark_dict["job id"] = job_id
    # currently untracked variables
    benchmark_dict["#nodes"], benchmark_dict["#cpus"], benchmark_dict["#gpus"] = None, None, None
    benchmark_dict["#mpi tasks"], benchmark_dict["node id"], benchmark_dict["max. gpu power"] = None, None, None
    benchmark_dict["gpu energy consumption"] = None
    # ... and save CSV-file with tracked data on disk
    bm_obj.populate_csv_from_dict(benchmark_dict)

    js_file = os.path.join(os.getcwd(), "benchmark_training_static.json")
    if not os.path.isfile(js_file):
        stat_info = {"static_model_info": {"trainable_parameters": count_params(unet_model.trainable_weights),
                                           "non-trainable_parameters": count_params(unet_model.non_trainable_weights)},
                     "data_info": {"training data size": -999., "validation data size": -999.,
                                   "nsamples": nsamples, "shape_samples": shape_in,
                                   "batch_size": args_dict["batch_size"]}}

        with open(js_file, "w") as jsf:
            js.dump(stat_info, jsf)

    print("Finished job at {0}".format(dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-in", dest="input_dir", type=str, required=True,
                        help="Directory where input netCDF-files are stored.")
    parser.add_argument("--output_dir", "-out", dest="output_dir", type=str, required=True,
                        help="Output directory where model is savded.")
    parser.add_argument("--job_id", "-id", dest="id", type=int, required=True, help="Job-id from Slurm.")
    parser.add_argument("--number_epochs", "-nepochs", dest="train_epochs", type=int, required=True,
                        help="Numer of epochs to train U-Net.")
    parser.add_argument("--batch_size", "-bs", dest="batch_size", type=int, default=32, 
                        help="Number of samples per mini-batch.")
    parser.add_argument("--learning_rate", "-lr", dest="lr", type=float, required=True,
                        help="Learning rate to train U-Net.")
    parser.add_argument("--no_z_branch", "-no_z", dest="no_z_branch", default=False, action="store_true",
                        help="Flag if U-net is optimzed on additional output branch for topography" +
                             "(see Sha et al., 2020)")
    parser.add_argument("--model_name", "-model_name", dest="model_name", type=str, required=True,
                        help="Name for the trained U-Net.")

    args = parser.parse_args()
    main(args)

