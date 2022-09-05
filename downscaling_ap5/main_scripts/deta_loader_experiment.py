import sys

sys.path.append('../')

import argparse
import xarray as xr
import os
import numpy as np
from models.wgan_model import WGAN, critic_model
from models.unet_model import build_unet
from handle_data.handle_data_unet import HandleUnetData
import time

def main(parser_args):
    """
    Measuring the time for load era-5 dataset using tensorflow implementation
    """

    datadir = parser_args.input_dir
    outdir = parser_args.output_dir
    job_id = parser_args.id

    ds_train = xr.open_dataset(os.path.join(datadir, "preproc_era5_crea6_small.nc"))
    start = time.time()
    keys_remove = ["input_dir", "output_dir", "id", "no_z_branch"]
    args_dict = {k: v for k, v in vars(parser_args).items() if (v is not None) & (k not in keys_remove)}
    args_dict["z_branch"] = not parser_args.no_z_branch
    # set critic learning rate equal to generator if not supplied
    if not "lr_critic": args_dict["lr_critic"] = args_dict["lr_gen"]

    wgan_model = WGAN(build_unet, critic_model, args_dict)



    def reshape_ds(ds):
        da = ds.to_array(dim="variables")
        da = da.transpose(..., "variables")
        return da

    da_train = reshape_ds(ds_train)

    norm_dims = ["time", "rlat", "rlon"]
    da_train, mu_train, std_train = HandleUnetData.z_norm_data(da_train, dims=norm_dims, return_stat=True)
    train_iter, val_iter = wgan_model.compile(da_train.astype(np.float32), da_train.astype(np.float32))
    it = iter(train_iter)
    start_2 = time.time()
    batch = next(it)
    end = time.time()
    print(f'tot time {end - start} seconds')
    print('time for 1 batch',end - start_2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-in", dest="input_dir", default="C:/Users/max_b/PycharmProjects/downscaling_maelstrom/", type=str, required=False,
                        help="Directory where input netCDF-files are stored.")
    parser.add_argument("--output_dir", "-out", dest="output_dir", default="C:/Users/max_b/PycharmProjects/downscaling_maelstrom/", type=str, required=False,
                        help="Output directory where model is savded.")
    parser.add_argument("--job_id", "-id", dest="id", type=int, default=11, required=False, help="Job-id from Slurm.")
    parser.add_argument("--number_epochs", "-nepochs", dest="train_epochs", default=30, type=int, required=False,
                        help="Numer of epochs to train WGAN.")
    parser.add_argument("--learning_rate_generator", "-lr_gen", dest="lr_gen", type=float, default=5.e-05, required=False,
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
    parser.add_argument("--model_name", "-model_name", dest="model_name", default='test', type=str, required=False,
                        help="Name for the trained WGAN.")

    args = parser.parse_args()
    # indir = "",
    # outdir = "",
    # nepochs = ,
    # lr_gen = ,
    # lr_critic = 1.e-06,
    # lr_end = 5.e-06,
    # lr_decay = True,
    # model_name = "my_wgan_model"
    main(args)



# # data-directories
# indir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_ifs/netcdf_data/all_files/
# outdir=/p/project/deepacf/maelstrom/langguth1/downscaling_jsc_repo/downscaling_ap5/trained_models/
#
# # declare directory-variables which will be modified by config_runscript.py
# nepochs=30
# lr_gen=5.e-05
# lr_critic=1.e-06
# lr_end=5.e-06
# lr_decay=True
# model_name=my_wgan_model
