__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2022-02-01"

import os, sys
import argparse
from timeit import default_timer as timer
import json as js
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as ku
from tensorflow.python.keras.utils.layer_utils import count_params
from handle_data_unet import HandleUnetData
from unet_model import build_unet, get_lr_scheduler
from benchmark_utils import BenchmarkCSV, get_training_time_dict


def main(parser_args):

    # start timing
    t0 = timer()

    method = main.__name__

    # parse arguments
    job_id = parser_args.job_id
    datadir = parser_args.input_dir
    outdir = parser_args.output_dir

    z_branch = not parser_args.no_z_branch
    hour = parser_args.hour
    nepochs = parser_args.nepochs
    batch_size = parser_args.batch_size

    # initialize benchmarking object
    bm_obj = BenchmarkCSV(os.path.join(os.getcwd(), "benchmark_training.csv"))
    # read and normalize data for training
    data_obj = HandleUnetData(datadir, "training", purpose="train")
    data_obj.append_data("validation", purpose="val")

    int_data, tart_data, opt_norm = data_obj.normalize("train", daytime=hour)
    inv_data, tarv_data = data_obj.normalize("val", daytime=hour, opt_norm=opt_norm)

    # get dictionary for tracking benchmark parameters
    tot_time_load = data_obj.timing["loading_times"]["train"] + data_obj.timing["loading_times"]["val"]
    benchmark_dict = {"loading data time": tot_time_load}

    # some information on training data
    nsamples = int_data.shape[0]
    shape_in = int_data.shape[1:]

    # visualize model architecture on login-node
    if "login" in data_obj.host:
        unet_model = build_unet(shape_in, z_branch=True)
        ku.plot_model(unet_model, to_file=os.path.join(outdir, "unet_downscaling_model.png"), show_shapes=True)

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
    unet_model = build_unet(shape_in, z_branch=z_branch)

    if z_branch:
        print("%{0}: Start training with optimization on surface topography (with z_branch).".format(method))
        unet_model.compile(optimizer=Adam(learning_rate=5*10**(-4)),
                           loss={"output_temp": "mae", "output_z": "mae"},
                           loss_weights={"output_temp": 1.0, "output_z": 1.0})

        history = unet_model.fit(x=int_data.values, y={"output_temp": tart_data.isel(variable=0).values,
                                                       "output_z": tart_data.isel(variable=1).values},
                                 batch_size=batch_size, epochs=nepochs, callbacks=callback_list,
                                 validation_data=(inv_data.values, {"output_temp": tarv_data.isel(variable=0).values,
                                                                    "output_z": tarv_data.isel(variable=1).values}), 
                                 verbose=2)
    else:
        print("%{0}: Start training without optimization on surface topography (with z_branch).".format(method))
        unet_model.compile(optimizer=Adam(learning_rate=5*10**(-4)), loss="mae")

        history = unet_model.fit(x=int_data.values, y=tart_data.isel(variable=0).values, batch_size=batch_size,
                                 epochs=nepochs, callbacks=callback_list,
                                 validation_data=(inv_data.values, tarv_data.isel(variable=0).values), 
                                 verbose=2)

    # get some parameters from tracked training times and put to dictionary
    training_times = get_training_time_dict(time_tracker.epoch_times, nsamples*nepochs)
    benchmark_dict = {**benchmark_dict, **training_times}
    # also track losses
    benchmark_dict["final training loss"] = history.history["output_temp_loss"][-1]
    benchmark_dict["final validation loss"] = history.history["val_output_temp_loss"][-1]

    # save trained model
    model_name = "trained_downscaling_unet_t2m_hour{0:0d}_exp{1:d}".format(hour, bm_obj.exp_number)
    print("%{0}: Save trained model '{1}' to '{2}'".format(method, model_name, outdir))
    t0_save = timer()
    unet_model.save(os.path.join(outdir, "{0}.h5".format(model_name)), save_format="h5")
    benchmark_dict = {**benchmark_dict, "saving model time": timer() - t0_save}

    # finally, track total runtime...
    benchmark_dict["total runtime"] = timer() - t0
    benchmark_dict["job id"] = job_id
    # currently untracked variables
    benchmark_dict["#nodes"], benchmark_dict["#cpus"], benchmark_dict["#gpus"]= None, None, None
    benchmark_dict["#mpi tasks"] = None
    # ... and save CSV-file with tracked data on disk
    bm_obj.populate_csv_from_dict(benchmark_dict)

    js_file = os.path.join(os.getcwd(), "benchmark_training_static.json")
    if not os.path.isfile(js_file):
        data_mem = data_obj.data_info["memory_datasets"]
        stat_info = {"static_model_info": {"trainable_parameters": count_params(unet_model.trainable_weights),
                                           "non-trainable_parameters": count_params(unet_model.non_trainable_weights)},
                     "data_info": {"training data size": data_mem["train"]/2., "validation data size": data_mem["val"]/2.,
                                   "nsamples" : nsamples, "shape_samples": shape_in, "batch_size": batch_size}}

        with open(js_file, "w") as jsf:
            js.dump(stat_info, jsf)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-in", dest="input_dir", type=str, required=True,
                        help="Directory where input netCDF-files are stored.")
    parser.add_argument("--output_dir", "-out", dest="output_dir",
                        type=str, required=True, help="Output directory where model is savded.")
    parser.add_argument("--number_epochs", "-nepochs", dest="nepochs", type=int, default=70,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", "-bs", dest="batch_size", type=int, default=32,
                        help = "Batch size during model training.")
    parser.add_argument("--job_id", "-id", dest="job_id", type=int, help="Job-id from Slurm.")
    parser.add_argument("--hour", "-hour", dest="hour", type=int, default=12,
                        help="Daytime hour for which model will be trained.")
    parser.add_argument("--no_z_branch", "-no_z", dest="no_z_branch", default=False, action="store_true",
                        help="Flag if U-net is optimzed on additional output branch for topography" +
                             "(see Sha et al., 2020)")

    args = parser.parse_args()
    main(args)


