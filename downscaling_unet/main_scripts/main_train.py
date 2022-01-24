__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2022-01-22"

import os, sys
import argparse
import time
import json
from timeit import default_timer as timer
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as ku
from handle_data_unet import *
from unet_model import build_unet, get_lr_scheduler
import numpy as np


def main(parser_args):

    method = main.__name__

    datadir = parser_args.input_dir
    outdir = parser_args.output_dir

    hour = parser_args.hour
    nepochs = parser_args.nepochs

    # Read and normalize data for training
    data_obj = HandleUnetData(datadir)

    int_data, tart_data, opt_norm = data_obj.normalize("train", daytime=hour)
    inv_data, tarv_data = data_obj.normalize("val", daytime=hour, opt_norm=opt_norm)

    benchmark_dict = data_obj.timing

    print(benchmark_dict)
    print(data_obj.data_info["memory_datasets"])
    print(data_obj.data_info["nsamples"])

    shape_in = (96, 128, 3)

    if "login" in data_obj.host:
        unet_model = build_unet(shape_in, z_branch=True)
        ku.plot_model(unet_model, to_file=os.path.join(outdir, "unet_downscaling_model.png"), show_shapes=True)

    # define class for creating timer callback
    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.epoch_times = []

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, epoch, logs={}):
            self.epoch_times.append(time.time() - self.epoch_time_start)

    z_branch = True  # flag if additionally training on surface elevation is performed

    lr_scheduler, time_tracker = get_lr_scheduler(), TimeHistory()
    # create callbacks
    callback_list = [lr_scheduler, time_tracker]

    # build, compile and train the model
    unet_model = build_unet(shape_in, z_branch=z_branch)

    if z_branch:
        unet_model.compile(optimizer=Adam(learning_rate=5*10**(-4)),
                       loss={"output_temp": "mae", "output_z": "mae"},
                       loss_weights={"output_temp": 1.0, "output_z": 1.0})

        history = unet_model.fit(x=int_data.values, y={"output_temp": tart_data.isel(variable=0).values,
                                                       "output_z": tart_data.isel(variable=1).values},
                                 batch_size=32, epochs=nepochs, callbacks=callback_list,
                                 validation_data=(inv_data.values, {"output_temp": tarv_data.isel(variable=0).values,
                                                                    "output_z": tarv_data.isel(variable=1).values}), 
                                 verbose=2)
    else:
        unet_model.compile(optimizer=Adam(learning_rate=5*10**(-4)), loss="mae")


        history = unet_model.fit(x=int_data.values, y=tart_data.isel(variable=0).values, batch_size=32,
                                 epochs=nepochs, callbacks=callback_list,
                                 validation_data=(inv_data.values, tarv_data.isel(variable=0).values), 
                                 verbose=2)

    epoch_times = time_tracker.epoch_times

    training_times = {"training" : {"total_training_time": np.sum(epoch_times), "avg_epoch_time": np.mean(epoch_times),
                      "min_training_time": np.amin(epoch_times), "max_training_time": np.amax(epoch_times[1:]),
                      "first_epoch_time": epoch_times[0], "number_samples": np.shape(int_data)[0]}}
    benchmark_dict = {**benchmark_dict, **training_times}

    print(history.history["output_temp_loss"][-1])
    print(history.history["val_output_temp_loss"][-1])

    print("Total training time: {0:.2f}s".format(np.sum(epoch_times)))
    print("Max. time per epoch: {0:.4f}s, min. time per epoch: {1:.4f}s".format(np.amax(epoch_times),
                                                                                np.amin(epoch_times)))

    # save trained model
    time0_save = timer()
    unet_model.save(os.path.join(outdir, "trained_downscaling_unet_t2m_hour{0:0d}.h5".format(hour)), save_format="h5")
    save_time = timer() - time0_save
    benchmark_dict = {**benchmark_dict, "save_time": save_time}
    print("%{0}: Saving trained model to '{1}' took {2:.2f}s.".format(method, outdir, save_time))

    with open(os.path.join(outdir, "benchmark_times.json"), "w") as f:
        json.dump(benchmark_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-in", dest="input_dir", type=str, required=True,
                        help="Directory where input netCDF-files are stored.")
    parser.add_argument("--output_dir", "-out", dest="output_dir",
                        type=str, required=True, help="Output directory where model is savded.")
    parser.add_argument("--number_epochs", "-nepochs", dest="nepochs", type=int, default=70,
                        help="Number of epochs for training.")
    parser.add_argument("--hour", "-hour", dest="hour", type=int, default=12,
                        help="Daytime hour for which model will be trained.")

    args = parser.parse_args()
    main(args)


