__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-05-31"
__update__ = "2022-05-31"

import os, sys
import argparse
from datetime import datetime as dt
print("Start with importing packages at {0}".format(dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")))
import numpy as np
import xarray as xr
from unet_model import build_unet
from wgan_model import WGAN, critic_model
from handle_data_unet import HandleUnetData


def main(parser_args):

    # Get some basic directories and flags
    datadir = parser_args.input_dir
    outdir = parser_args.output_dir

    z_branch = not parser_args.no_z_branch

    # Read training and validation data
    ds_train, ds_val = xr.open_dataset(os.path.join(datadir, "era5_to_ifs_train_corrected.nc")), \
                       xr.open_dataset(os.path.join(datadir, "era5_to_ifs_val_corrected.nc"))

    print("Datasets for trining, validation and testing loaded.")

    wgan_model = WGAN(build_unet, critic_model,
                      {"lr_decay": parser_args.lr_decay, "lr": parser_args.lr,
                       "train_epochs": parser_args.nepochs, "recon_weight": parser_args.recon_wgt,
                       "d_steps": parser_args.d_steps,
                       "optimizer": parser_args.optimizer, "z_branch": z_branch})

    # prepare data
    def reshape_ds(ds):
        da = ds.to_array(dim="variables")  # .squeeze()
        da = da.transpose(..., "variables")
        return da

    da_train, da_val = reshape_ds(ds_train), reshape_ds(ds_val)

    norm_dims = ["time", "lat", "lon"]
    da_train, mu_train, std_train = HandleUnetData.z_norm_data(da_train, dims=norm_dims, return_stat=True)
    da_val = HandleUnetData.z_norm_data(da_val, mu=mu_train, std=std_train)

    print("Start compiling WGAN-model.")
    train_iter, val_iter = wgan_model.compile(da_train.astype(np.float32), da_val.astype(np.float32))

    # train model
    print("Start training of WGAN...")
    history = wgan_model.fit(train_iter, val_iter)

    print("WGAN training finished. Save model to '{0}' and start creating example plot.".format(os.path.join(outdir, parser_args.model_name)))
    # save trained model
    model_savedir = os.path.join(outdir, parser_args.model_name)
    os.makedirs(model_savedir, exist_ok=True)
    wgan_model.save_weights(os.path.join(model_savedir, parser_args.model_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-in", dest="input_dir", type=str, required=True,
                        help="Directory where input netCDF-files are stored.")
    parser.add_argument("--output_dir", "-out", dest="output_dir", type=str, required=True,
                        help="Output directory where model is savded.")
    parser.add_argument("--number_epochs", "-nepochs", dest="nepochs", type=int, required=True,
                        help="Numer of epochs to train WGAN.")
    parser.add_argument("--learning_rate", "-lr", dest="lr", type=float, required=True,
                        help="Learning rate to train WGAN.")
    parser.add_argument("--learning_rate_decay", "-lr_decay", dest="lr_decay", default=False, action="store_true",
                        help="Flag to perform learning rate decay.")
    parser.add_argument("--optimizer", "-opt", dest="optimizer", type=str, default="adam",
                        help = "Optimizer to train WGAN.")
    parser.add_argument("--discriminator_steps", "-d_steps", dest="d_steps", type=int, default=6,
                        help = "Substeps to train critic/discriminator of WGAN.")
    parser.add_argument("--reconstruction_weight", "-recon_wgt", dest="recon_wgt", type=float, default=1000.,
                        help = "Reconstruction weight used by generator.")
    parser.add_argument("--no_z_branch", "-no_z", dest="no_z_branch", default=False, action="store_true",
                        help="Flag if U-net is optimzed on additional output branch for topography" +
                             "(see Sha et al., 2020)")
    parser.add_argument("--model_name", "-model_name", dest="model_name", type=str, required=True,
                        help="Name for the trained WGAN.")

    args = parser.parse_args()
    main(args)

