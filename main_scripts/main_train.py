# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to train downscaling models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-10-06"
__update__ = "2024-03-08"

import os
import argparse
from datetime import datetime as dt
print("Start with importing packages at {0}".format(dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")))
import json as js
from timeit import default_timer as timer
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from all_normalizations import ZScore
from model_engine import ModelEngine
from handle_data_class import prepare_dataset
from other_utils import print_gpu_usage, print_cpu_usage, copy_filelist, get_training_time_dict

# Open issues:
# * nepochs and, if required, d_steps  must be parsed with hparams_dict as model is uninstantiated at this point and thus no default parameters
#   are available

def main(parser_args):
    # start timing
    t0 = timer()

    # Get some basic directories and flags
    datadir = parser_args.input_dir
    outdir = parser_args.output_dir
    job_id = parser_args.id
    dataset = parser_args.dataset.lower()
    js_norm = parser_args.js_norm

    print(f"Start training job with ID: {job_id}")

    # initialize checkpoint-directory path for saving the model
    model_savedir = os.path.join(outdir, parser_args.exp_name)

    # read configuration files for model and dataset
    with parser_args.conf_ds as dsf:
        ds_dict = js.load(dsf)

    with parser_args.conf_md as mdf:
        hparams_dict = js.load(mdf)

    # get normalization object if corresponding JSON-file is parsed
    if js_norm:
        data_norm = ZScore(ds_dict["norm_dims"])
        data_norm.read_norm_from_file(js_norm)
        norm_dims, write_norm = None, False
    else:
        data_norm, write_norm = None, True
        norm_dims = ds_dict["norm_dims"]        

    # get model instance and set-up batch size
    model_instance = ModelEngine(parser_args.model)

    # get tensoflow dataset objects for training and validation data
    # training dataset
    t0_train = timer()
    tfds_train, train_info = prepare_dataset(datadir, dataset, ds_dict, hparams_dict, "train", ds_dict["predictands"], 
                                             norm_obj=data_norm, norm_dims=norm_dims) 
    
    data_norm, shape_in, nsamples, tfds_train_size = train_info["data_norm"], train_info["shape_in"], \
                                                     train_info["nsamples"], train_info["dataset_size"]
    ds_obj_train = train_info.get("ds_obj", None)

    # Tracking training data preparation time if all data is already loaded into memory
    if ds_obj_train is None:
        ttrain_load = timer() - t0_train
        print(f"Training data loading time: {ttrain_load:.2f}s.")
    else:
        # training data will be loaded on-the-fly
        ttrain_load = None
    
    if write_norm:
        data_norm.save_norm_to_file(os.path.join(model_savedir, "norm.json"))
    
    # validation dataset
    t0_val = timer()
    tfds_val, val_info = prepare_dataset(datadir, dataset, ds_dict, hparams_dict, "val", ds_dict["predictands"], 
                                         norm_obj=data_norm) 
    
    ds_obj_val = val_info.get("ds_obj", None)
    
    # Tracking validation data preparation time if all data is already loaded into memory
    if ds_obj_val is None:
        tval_load = timer() - t0_val
        print(f"Validation data loading time: {tval_load:.2f}s.")
    else:
        # validation data will be loaded on-the-fly
        tval_load = None

    print("Finished data preparation")

    # instantiate model...
    # Note: Parse varnames from train_info since list of varnames might get updated depending on model and dataset configuration
    model = model_instance(shape_in, list(train_info["varnames_tar"]), hparams_dict, model_savedir, parser_args.exp_name) 

    # ... compile
    model.compile(**model.compile_options)

    # copy configuration and normalization JSON-file to model-directory (incl. renaming)
    os.makedirs(model_savedir, exist_ok=True)
    filelist, filelist_new = [parser_args.conf_ds.name, parser_args.conf_md.name], [f"config_ds_{dataset}.json", f"custom_config_{parser_args.model}.json"]
    if not write_norm:
        filelist.append(js_norm), filelist_new.append(os.path.basename(js_norm))
    
    copy_filelist(filelist, model_savedir, filelist_new)
    model.save_hparams_to_json()

    # train model
    steps_per_epoch = int(np.ceil(nsamples / ds_dict["batch_size"]))

    # run training (note that callbacks are part of the fit_options by default)
    print(f"Start training of {parser_args.model.capitalize()}...")
    history = model.fit(x=tfds_train, epochs=model.hparams["nepochs"],
                        steps_per_epoch=steps_per_epoch, validation_data=tfds_val, validation_steps=300,
                        verbose=2, **model.fit_options)

    # get some parameters from tracked training times and put to dictionary
    training_times = get_training_time_dict(model.fit_options["callbacks"][0].epoch_times,
                                            nsamples * model.hparams["nepochs"])
    if not ttrain_load:
        ttrain_load = sum(ds_obj_train.reading_times) #+ tval_load
        print(f"Training data loading time: {ttrain_load:.2f}s.")
        print(f"Average throughput: {ds_obj_train.ds_proc_size / 1.e+06 / training_times['Total training time']:.3f} MB/s")

    # save model
    t0_save = timer()
    model_savedir_last = os.path.join(model_savedir, f"{parser_args.exp_name}_last")
    model.save(filepath=model_savedir_last)
    
    saving_time = timer() - t0_save
    print(f"Model saving time: {saving_time:.2f}s")

    # plot model
    try:
        model.plot_model(model_savedir, show_shapes=True) # , show_layer_actiavtions=True)
    except:
        plot_model(model, os.path.join(model_savedir, f"plot_{parser_args.exp_name}.png"),
                   show_shapes=True) #, show_layer_activations=True)

    # final timing
    tend = timer()
    
    # get trainable and untrainable parameters
    trainable_params, untrainable_params = model.count_params()
    print(f"# trainable parameters: {trainable_params}, # untrainable parameters: {untrainable_params}")
    
    # end timing
    tot_run_time = tend - t0
    print(f"Total runtime: {tot_run_time:.1f}s")
    # some statistics on memory usage
    print_gpu_usage("Final GPU memory: ")
    print_cpu_usage("Final CPU memory: ")

    print("Finished training at {0}".format(dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")))
    print("**************************************************************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-in", dest="input_dir", type=str, required=True,
                        help="Directory where input netCDF-files are stored.")
    parser.add_argument("--output_dir", "-out", dest="output_dir", type=str, required=True,
                        help="Output directory where model is savded.")
    parser.add_argument("--downscaling_model", "-model", dest="model", type=str, required=True,
                        help="Name of model architeture used for downscaling.")
    parser.add_argument("--downscaling_dataset", "-dataset", dest="dataset", type=str, required=True,
                        help="Name of dataset to be used for downscaling model.")
    parser.add_argument("--experiment_name", "-exp_name", dest="exp_name", type=str, required=True,
                        help="Name for the current experiment.")
    parser.add_argument("--configuration_model", "-conf_md", dest="conf_md", type=argparse.FileType("r"), required=True,
                        help="JSON-file to configure model to be trained.")
    parser.add_argument("--configuration_dataset", "-conf_ds", dest="conf_ds", type=argparse.FileType("r"),
                        required=True, help="JSON-file to configure dataset to be used for training.")
    parser.add_argument("--json_norm_file", "-js_norm", dest="js_norm", type=str, default=None,
                        help="JSON-file providing normalization parameters.")
    parser.add_argument("--job_id", "-id", dest="id", type=int, required=True, help="Job-id from Slurm.")

    args = parser.parse_args()
    main(args)
  
