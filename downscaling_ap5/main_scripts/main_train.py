# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to train downscaling models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-10-06"
__update__ = "2023-04-17"

import os
import argparse
from datetime import datetime as dt
print("Start with importing packages at {0}".format(dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")))
import gc
import json as js
from timeit import default_timer as timer
import numpy as np
import xarray as xr
from all_normalizations import ZScore
from model_utils import ModelEngine, TimeHistory, handle_opt_utils, get_loss_from_history
from handle_data_class import HandleDataClass, get_dataset_filename
from other_utils import print_gpu_usage, print_cpu_usage, copy_filelist
from benchmark_utils import BenchmarkCSV, get_training_time_dict


# Open issues:
# * d_steps must be parsed with hparams_dict as model is uninstantiated at this point and thus no default parameters
#   are available
# * flag named_targets must be set to False in hparams_dict for WGAN to work with U-Net 
# * ensure that dataset defaults are set (such as predictands for WGAN)
# * customized choice on predictors missing
# * replacement of manual benchmarking by JUBE-benchmarking to avoid duplications

def main(parser_args):
    # start timing
    t0 = timer()

    # Get some basic directories and flags
    datadir = parser_args.input_dir
    outdir = parser_args.output_dir
    job_id = parser_args.id
    dataset = parser_args.dataset.lower()
    js_norm = parser_args.js_norm

    # initialize checkpoint-directory path for saving the model
    model_savedir = os.path.join(outdir, parser_args.exp_name)

    # read configuration files for model and dataset
    with parser_args.conf_ds as dsf:
        ds_dict = js.load(dsf)

    with parser_args.conf_md as mdf:
        hparams_dict = js.load(mdf)
    
    named_targets = hparams_dict.get("named_targets", False)

    # get normalization object if corresponding JSON-file is parsed
    if js_norm:
        data_norm = ZScore(ds_dict["norm_dims"])
        data_norm.read_norm_from_file(js_norm)
        norm_dims, write_norm = None, False
    else:
        data_norm, write_norm = None, True
        norm_dims = ds_dict["norm_dims"]

    # initialize benchmarking object
    bm_obj = BenchmarkCSV(os.path.join(os.getcwd(), f"benchmark_training_{parser_args.model}.csv"))

    # get model instance and set-up batch size
    model_instance = ModelEngine(parser_args.model)
    # Note: bs_train is introduced to allow substepping in the training loop, e.g. for WGAN where n optimization steps
    # are applied to train the critic, before the generator is trained once.
    # The validation dataset however does not perform substeeping and thus doesn't require an increased mini-batch size.
    bs_train = ds_dict["batch_size"] * (hparams_dict["d_steps"] + 1) if "d_steps" in hparams_dict else ds_dict["batch_size"]
    nepochs = hparams_dict["nepochs"] * (hparams_dict["d_steps"] + 1) if "d_steps" in hparams_dict else hparams_dict["nepochs"]

    # start handling training and validation data
    # training data
    print("Start preparing training data...")
    t0_train = timer()
    varnames_tar = list(ds_dict["predictands"])
    fname_or_patt_train = get_dataset_filename(datadir, dataset, "train", ds_dict.get("laugmented", False))

    # if fname_or_patt_train is a filename (string without wildcard), all data will be loaded into memory
    # if fname_or_patt_train is a filename pattern (string with wildcard), the TF-dataset will iterate over subsets of
    # the dataset
    if "*" in fname_or_patt_train:
        ds_obj, tfds_train = HandleDataClass.make_tf_dataset_dyn(datadir, fname_or_patt_train, bs_train, nepochs,
                                                                 nfiles2merge=30, var_tar2in=ds_dict["var_tar2in"],
                                                                 predictands=ds_dict["predictands"],
                                                                 named_targets=named_targets,
                                                                 norm_obj=data_norm, norm_dims=norm_dims)
        data_norm = ds_obj.data_norm
        nsamples, shape_in = ds_obj.nsamples, (*ds_obj.data_dim[::-1], ds_obj.n_predictors)
        tfds_train_size = ds_obj.dataset_size
    else:
        ds_train = xr.open_dataset(fname_or_patt_train)
        da_train = HandleDataClass.reshape_ds(ds_train.astype("float32", copy=False))

        if not data_norm:
            # data_norm must be freshly instantiated (triggering later parameter retrieval)
            data_norm = ZScore(ds_dict["norm_dims"])

        da_train = data_norm.normalize(da_train)
        tfds_train = HandleDataClass.make_tf_dataset_allmem(da_train, bs_train, var_tar2in=ds_dict["var_tar2in"],
                                                            named_targets=named_targets)
        nsamples, shape_in = da_train.shape[0], tfds_train.element_spec[0].shape[1:].as_list()
        tfds_train_size = da_train.nbytes

    if write_norm:
        data_norm.save_norm_to_file(os.path.join(model_savedir, "norm.json"))

    print(f"TF training dataset preparation time: {timer() - t0_train:.2f}s.")

    # validation data
    print("Start preparing validation data...")
    t0_val = timer()
    fdata_val = get_dataset_filename(datadir, dataset, "val", ds_dict.get("laugmented", False))
    with xr.open_dataset(fdata_val) as ds_val:
        ds_val = data_norm.normalize(ds_val)
    da_val = HandleDataClass.reshape_ds(ds_val)

    tfds_val = HandleDataClass.make_tf_dataset_allmem(da_val.astype("float32", copy=True), ds_dict["batch_size"],
                                                      lshuffle=True, var_tar2in=ds_dict["var_tar2in"],
                                                      named_targets=named_targets)
    
    # clean up to save some memory
    del ds_val
    gc.collect()

    tval_load = timer() - t0_val
    print(f"Validation data preparation time: {tval_load:.2f}s.")

    # Read data from disk and preprocess (normalization and conversion to TF dataset)
    if "ds_obj" in locals():
        # training data will be loaded on-the-fly
        ttrain_load = None
    else:
        ttrain_load = timer() - t0_train
        print(f"Data loading time: {ttrain_load:.2f}s.")

    # instantiate model
    model = model_instance(shape_in, varnames_tar, hparams_dict, model_savedir, parser_args.exp_name)
    model.varnames_tar = varnames_tar

    # get optional compile options and compile
    compile_opts = handle_opt_utils(model, "get_compile_opts")
    model.compile(**compile_opts)

    # train model
    time_tracker = TimeHistory()
    steps_per_epoch = int(np.ceil(nsamples / ds_dict["batch_size"]))

    # get optional fit options and start training/fitting
    fit_opts = handle_opt_utils(model, "get_fit_opts")
    print(f"Start training of {parser_args.model.capitalize()}...")
    history = model.fit(x=tfds_train, callbacks=[time_tracker], epochs=model.hparams["nepochs"],
                        steps_per_epoch=steps_per_epoch, validation_data=tfds_val, validation_steps=300,
                        verbose=2, **fit_opts)

    # get some parameters from tracked training times and put to dictionary
    training_times = get_training_time_dict(time_tracker.epoch_times,
                                            nsamples * model.hparams["nepochs"])
    if not ttrain_load:
        ttrain_load = sum(ds_obj.reading_times) + tval_load
        print(f"Data loading time: {ttrain_load:.2f}s.")
        print(f"Average throughput: {ds_obj.ds_proc_size / 1.e+06 / training_times['Total training time']:.3f} MB/s")
    benchmark_dict = {**{"data loading time": ttrain_load}, **training_times}
    print(f"Model '{parser_args.exp_name}' training time: {training_times['Total training time']:.2f} s. " +
          f"Save model to '{model_savedir}'")

    # save trained model
    t0_save = timer()

    os.makedirs(model_savedir, exist_ok=True)
    model.save(filepath=model_savedir)

    # final timing
    tend = timer()
    saving_time = tend - t0_save
    tot_run_time = tend - t0
    print(f"Model saving time: {saving_time:.2f}s")
    print(f"Total runtime: {tot_run_time:.1f}s")
    # some statistics on memory usage
    print_gpu_usage("Final GPU memory: ")
    print_cpu_usage("Final CPU memory: ")

    # populate benchmark dictionary
    benchmark_dict.update({"saving model time": saving_time, "total runtime": tot_run_time})
    benchmark_dict.update({"job id": job_id, "#nodes": 1, "#cpus": len(os.sched_getaffinity(0)), "#gpus": 1,
                           "#mpi tasks": 1, "node id": None, "max. gpu power": None, "gpu energy consumption": None})
    try:
        benchmark_dict["final training loss"] = get_loss_from_history(history, "loss")
    except KeyError:
        benchmark_dict["final training loss"] = get_loss_from_history(history, "recon_loss")
    try:
        benchmark_dict["final validation loss"] = get_loss_from_history(history, "val_loss")
    except KeyError:
        benchmark_dict["final validation loss"] = get_loss_from_history(history, "val_recon_loss")
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
                     "data_info": {"training data size": tfds_train_size, "validation data size": da_val.nbytes,
                                   "nsamples": nsamples, "shape_samples": shape_in,
                                   "batch_size": ds_dict["batch_size"]}}

        with open(js_file, "w") as jsf:
            js.dump(stat_info, jsf)

    print("Finished job at {0}".format(dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")))

    # copy configuration and normalization JSON-file to output-directory
    filelist = [parser_args.conf_ds.name, parser_args.conf_md.name]
    if not write_norm: filelist.append(js_norm)
    copy_filelist(filelist, model_savedir)


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
