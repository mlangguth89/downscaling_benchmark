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
from tensorflow.python.keras.utils.layer_utils import count_params
from unet_model import build_unet
from wgan_model import WGAN, critic_model
from handle_data_unet import HandleUnetData
from benchmark_utils import BenchmarkCSV, get_training_time_dict


def main(parser_args):
    # start timing
    t0 = timer()

    # initialize benchmarking object
    bm_obj = BenchmarkCSV(os.path.join(os.getcwd(), "benchmark_training_wgan.csv"))

    # Get some basic directories and flags
    datadir = parser_args.input_dir
    outdir = parser_args.output_dir
    job_id = parser_args.id

    # Read training and validation data
    print("Start reading data from disk...")
    t0_save = timer()
    #ds_train, ds_val = xr.open_dataset(os.path.join(datadir, "preproc_era5_crea6_train.nc"), chunks="auto"), \
    #                   xr.open_dataset(os.path.join(datadir, "preproc_era5_crea6_val.nc"), chunks="auto")

    ds_train, ds_val = xr.open_dataset(os.path.join(datadir, "preproc_era5_crea6_train.nc")), \
                       xr.open_dataset(os.path.join(datadir, "preproc_era5_crea6_val.nc"))


    benchmark_dict = {"loading data time": timer() - t0_save}

    keys_remove = ["input_dir", "output_dir", "id", "no_z_branch"]
    args_dict = {k: v for k, v in vars(parser_args).items() if (v is not None) & (k not in keys_remove)}
    args_dict["z_branch"] = not parser_args.no_z_branch
    # set critic learning rate equal to generator if not supplied
    if not "lr_critic": args_dict["lr_critic"] = args_dict["lr_gen"]

    wgan_model = WGAN(build_unet, critic_model, args_dict)

    # preprocess data (i.e. normalizing)
    def reshape_ds(ds):
        da = ds.to_array(dim="variables")
        da = da.transpose(..., "variables")
        return da

    t0_preproc = timer()
    # slice data temporaly
    ds_train = ds_train.sel(time=slice("2011-01-01", "2016-12-30"))
    
    da_train, da_val = reshape_ds(ds_train), reshape_ds(ds_val)

    norm_dims = ["time", "rlat", "rlon"]
    da_train, mu_train, std_train = HandleUnetData.z_norm_data(da_train, dims=norm_dims, save_path = os.path.join(outdir,parser_args.model_name), return_stat=True)
    da_val = HandleUnetData.z_norm_data(da_val,save_path =  os.path.join(outdir,parser_args.model_name))

    del ds_train
    del ds_val
    gc.collect()
    
    t0_compile = timer()
    benchmark_dict["preprocessing data time"] = t0_compile - t0_preproc

    # compile model and get dataset iterators
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
            self._supports_tf_logs=True

        def on_train_begin(self, logs={}):
            self.lr_gen_logs = []
            self.lr_critic_logs=[]

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
                                            wgan_model.nsamples*wgan_model.hparams["train_epochs"])
    benchmark_dict = {**benchmark_dict, **training_times}

    print(lr_tracker.lr_gen_logs)
    print(lr_tracker.lr_critic_logs)

    print("WGAN training finished. Save model to '{0}' and start creating example plot."
          .format(os.path.join(outdir, parser_args.model_name)))
    # save trained model (generator and critic are saved seperately)
    t0_save = timer()
    model_savedir = os.path.join(outdir, parser_args.model_name)
    os.makedirs(model_savedir, exist_ok=True)

    wgan_model.generator.save(os.path.join(model_savedir, "{0}_generator".format(parser_args.model_name)))
    wgan_model.critic.save(os.path.join(model_savedir, "{0}_critic".format(parser_args.model_name)))

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
                      "non-trainable_parameters_generator": count_params(wgan_model.generator.non_trainable_weights),
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
    parser.add_argument("--job_id", "-id", dest="id", type=int, required=True, help="Job-id from Slurm.")
    parser.add_argument("--number_epochs", "-nepochs", dest="train_epochs", type=int, required=True,
                        help="Numer of epochs to train WGAN.")
    parser.add_argument("--learning_rate_generator", "-lr_gen", dest="lr_gen", type=float, required=True,
                        help="Learning rate to train generator of WGAN.")
    parser.add_argument("--learning_rate_critic", "-lr_critic", dest="lr_critic", type=float, default=None,
                        help="Learning rate to train critic of WGAN.")
    parser.add_argument("--learning_rate_decay", "-lr_decay", dest="lr_decay", default=False, action="store_true",
                        help="Flag to perform learning rate decay.")
    parser.add_argument("--decay_start_epoch", "-decay_start", dest="decay_start", type=int,
                        help="Start epoch for learning rate decay.")
    parser.add_argument("--decay_end_epoch", "-decay_end", dest="decay_end", type=int,
                        help="End epoch for learning rate decay.")
    parser.add_argument("--learning_rate_generator_end", "-lr_gen_end", dest="lr_gen_end", type=float, default=None,
                        help="End learning rate to configure learning rate decay.")
    parser.add_argument("--number_features", "-ngf", dest="ngf", type=int, default=None,
                        help="Number of features/channels in first conv-layer.")
    parser.add_argument("--gradient_penalty_weight", "-gp_weight", dest="gp_weight", type=float, default=None,
                        help="Gradient penalty weight used to optimize critic.")
    parser.add_argument("--optimizer", "-opt", dest="optimizer", type=str, default="adam",
                        help="Optimizer to train WGAN.")
    parser.add_argument("--discriminator_steps", "-d_steps", dest="d_steps", type=int, default=6,
                        help="Substeps to train critic/discriminator of WGAN.")
    parser.add_argument("--reconstruction_weight", "-recon_wgt", dest="recon_weight", type=float, default=1000.,
                        help="Reconstruction weight used by generator.")
    parser.add_argument("--no_z_branch", "-no_z", dest="no_z_branch", default=False, action="store_true",
                        help="Flag if U-net is optimzed on additional output branch for topography" +
                             "(see Sha et al., 2020)")
    parser.add_argument("--model_name", "-model_name", dest="model_name", type=str, required=True,
                        help="Name for the trained WGAN.")

    args = parser.parse_args()
    main(args)

