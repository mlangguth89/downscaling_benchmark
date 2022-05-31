__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-05-19"
__update__ = "2022-05-26"

import os, sys
from collections import OrderedDict
import tensorflow as tf
import tensorflow.keras as keras

# all the layers used for U-net
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPool2D, Dense, Flatten, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
# other modules
import argparse
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

from unet_model import build_unet, conv_block

from typing import List, Tuple, Union

list_or_tuple = Union[List, Tuple]


def critic_model(shape, num_conv: int = 4, channels_start: int = 64, kernel: tuple = (3, 3),
                 stride: tuple = (2, 2), activation: str = "relu", lbatch_norm: bool = True):
    """
    Set-up convolutional discriminator model that is followed by two dense-layers
    :param shape: input shape of data (either real or generated data)
    :param num_conv: number of convolutional layers
    :param channels_start: number of channels in first layer
    :param kernel: convolutional kernel, e.g. (3,3)
    :param stride: stride for convolutional layer
    :param activation: activation function to use (applies to convolutional and fully-connected layers)
    :param lbatch_norm: flag to perform batch normalization on convolutional layers
    :return:
    """
    critic_in = Input(shape=shape)
    x = critic_in

    channels = channels_start

    if num_conv < 2:
        raise ValueError("Number of convolutional layers num_conv must be 2 at minimum.")

    for _ in range(num_conv):
        x = conv_block(x, channels, kernel, stride, activation=activation, l_batch_normalization=lbatch_norm)
        channels *= 2

    # finally perform global average pooling and finalize by fully connected layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(channels_start)(x)
    # ... and end with linear output layer
    out = Dense(1, activation="linear")(x)

    critic = Model(inputs=critic_in, outputs=out)

    return critic  # , out


class WGAN(keras.Model):
    """
    Class for Wassterstein GAN models
    """

    def __init__(self, generator: keras.Model, critic: keras.Model, hparams: dict):
        """
        Initiate Wasserstein GAN model.
        :param generator: A generator model returning a data field
        :param critic: A critic model which returns a critic scalar on the data field
        :param hparams: dictionary of hyperparameters
        :param l_embedding: Flag to enable time embeddings
        """

        super(WGAN, self).__init__()

        self.generator, self.critic = generator, critic
        self.hparams = WGAN.get_hparams_dict(hparams)
        if self.hparams["l_embed"]:
            raise ValueError("Embedding is not implemented yet.")
        # set in compile-method
        self.train_iter, self.val_iter = None, None
        self.nsamples = None
        self.lr_scheduler = None
        self.c_optimizer, self.g_optimizer = None, None

    def compile(self, da_train, da_val):
        """
        Turn datasets into tf.Datasets, compile model and set learning rate schedule (optionally)
        :param da_train: Data Array providing the training data (must have variables-dimension)
        :param da_val: Data Array providing the validation data (must have variables-dimension)
        :return: tf.Datasets for training and validation data
        """
        invars = [var for var in da_train["variables"].values if var.endswith("_in")]

        # determine shape-dimensions from data
        shape_all = da_train.sel({"variables": invars}).shape
        self.nsamples, in_shape = shape_all[0], shape_all[1:]

        tar_shape = (*in_shape[:-1], 1)    # critic only accounts for first channel (should be the downscaling target)
        # instantiate models
        self.generator = self.generator(in_shape, channels_start=self.hparams["ngf"], z_branch=self.hparams["z_branch"])
        self.critic = self.critic(tar_shape)

        train_iter, val_iter = self.make_data_generator(da_train, ds_val=da_val)
        # call Keras compile method
        super(WGAN, self).compile()
        # set optimizers
        self.c_optimizer, self.g_optimizer = self.hparams["d_optimizer"], self.hparams["g_optimizer"]

        # get learning rate schedule if desired
        if self.hparams["lr_decay"]:
            self.lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.get_lr_decay(), verbose=1)

        return train_iter, val_iter

    def get_lr_decay(self):
        """
        Get callable of learning rate scheduler which can be used as callabck in Keras models.
        Exponential decay is applied to change the learning rate from the start to the end value.
        :return: learning rate scheduler
        """
        decay_st, decay_end = self.hparams["decay_start"], self.hparams["decay_end"]
        lr_start, lr_end = self.hparams["lr"], self.hparams["lr_end"]

        if not decay_end > decay_st:
            raise ValueError("Epoch for end of learning rate decay must be large than start epoch. " +
                             "Your values: {0:d}, {1:d})".format(decay_st, decay_end))

        ne_decay = decay_end - decay_st
        # calculate decay rate from start and end learning rate
        decay_rate = 1./ne_decay*np.log(lr_end/lr_start)

        def lr_scheduler(epoch, lr):
            if epoch < decay_st:
                return lr
            elif decay_st <= epoch < decay_end:
                return lr * tf.math.exp(decay_rate)
            elif epoch >= decay_end:
                return lr

        return lr_scheduler

    def fit(self, train_iter, val_iter, callbacks: List = [None]):

        callbacks = [e for e in [self.lr_scheduler] + callbacks if e is not None]
        steps_per_epoch = int(np.ceil(self.nsamples / self.hparams["batch_size"]))

        return super(WGAN, self).fit(x=train_iter, callbacks=callbacks, epochs=self.hparams["train_epochs"],
                                     steps_per_epoch=steps_per_epoch, validation_data=val_iter, validation_steps=3, verbose=2)

    def train_step(self, data_iter: tf.data.Dataset, embed=None) -> OrderedDict:
        """
        Training step for Wasserstein GAN.
        :param data_iter: Tensorflow Dataset providing training data
        :param embed: embedding (not implemented yet)
        :return: Ordered dictionary with several losses of generator and critic
        """

        predictors, predictands = data_iter

        # train the critic d_steps-times
        for i in range(self.hparams["d_steps"]):
            with tf.GradientTape() as tape_critic:
                ist, ie = i * self.hparams["batch_size"], (i + 1) * self.hparams["batch_size"]
                # critic only operates on first channel
                predictands_critic = tf.expand_dims(predictands[ist:ie, :, :, 0], axis=-1)
                # generate (downscaled) data
                gen_data = self.generator(predictors[ist:ie, :, :, :], training=True)
                # calculate critics for both, the real and the generated data
                critic_gen = self.critic(gen_data[0], training=True)
                critic_gt = self.critic(predictands_critic, training=True)
                # calculate the loss (incl. gradient penalty)
                c_loss = WGAN.critic_loss(critic_gt, critic_gen)
                gp = self.gradient_penalty(predictands_critic, gen_data[0])

                d_loss = c_loss + self.hparams["gp_weight"] * gp

            # calculate gradients and update discrimintor
            d_gradient = tape_critic.gradient(d_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(d_gradient, self.critic.trainable_variables))

        # train generator
        with tf.GradientTape() as tape_generator:
            # generate (downscaled) data
            gen_data = self.generator(predictors[-self.hparams["batch_size"]:, :, :, :], training=True)
            # get the critic and calculate corresponding generator losses (critic and reconstruction loss)
            critic_gen = self.critic(gen_data[0], training=True)
            cg_loss = WGAN.critic_gen_loss(critic_gen)
            rloss = self.recon_loss(predictands[-self.hparams["batch_size"]:, :, :, :], gen_data)

            g_loss = cg_loss + self.hparams["recon_weight"] * rloss

        g_gradient = tape_generator.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        return OrderedDict(
            [("c_loss", c_loss), ("gp_loss", self.hparams["gp_weight"] * gp), ("d_loss", d_loss), ("cg_loss", cg_loss),
             ("recon_loss", rloss * self.hparams["recon_weight"]), ("g_loss", g_loss)])

    def test_step(self, val_iter: tf.data.Dataset) -> OrderedDict:
        """
        Implement step to test trained generator on validation data.
        :param val_iter: Tensorflow Dataset with validation data
        :return: dictionary with reconstruction loss on validation data
        """
        predictors, predictands = val_iter

        gen_data = self.generator(predictors, training=False)

        rloss = self.recon_loss(predictands, gen_data)

        return OrderedDict([("recon_loss_val", rloss)])

    def predict_step(self, test_iter: tf.data.Dataset) -> OrderedDict:

        predictors, _ = test_iter

        return self.generator(predictors, training=False)

    def make_data_generator(self, ds: xr.DataArray, ds_val: xr.DataArray = None, embed=None, embed_val=None,
                            var2drop: str = "z_tar") -> tf.data.Dataset:

        if not self.hparams["z_branch"]:
            ds = ds.drop(var2drop, dim="variables")

        ds_in, ds_tar = WGAN.split_in_tar(ds)

        if self.hparams["l_embed"]:
            if not embed:
                raise ValueError("Embedding is enabled, but no embedding data was parsed.")
            data_iter = tf.data.Dataset.from_tensor_slices((ds_in, ds_tar, embed))
        else:
            data_iter = tf.data.Dataset.from_tensor_slices((ds_in, ds_tar))

        # repeat must be before shuffle to get varying mini-batches per epoch
        # increase batch-size to allow substepping
        data_iter = data_iter.repeat().shuffle(10000).batch(self.hparams["batch_size"] * (self.hparams["d_steps"] + 1))
        # data_iter = data_iter.prefetch(tf.data.AUTOTUNE)

        if ds_val is not None:
            ds_val_in, ds_val_tar = WGAN.split_in_tar(ds_val)

            if self.hparams["l_embed"]:
                if not embed_val:
                    raise ValueError("Embedding is enabled, but no embedding data for validation dataset is parsed.")
                val_iter = tf.data.Dataset.from_tensor_slices((ds_val_in, ds_val_tar, embed_val))
            else:
                val_iter = tf.data.Dataset.from_tensor_slices((ds_val_in, ds_val_tar))

            val_iter = val_iter.repeat().batch(self.hparams["batch_size"])

            return data_iter, val_iter
        else:
            return data_iter

    def gradient_penalty(self, real_data, gen_data):
        """
        Calculates gradient penalty based on 'mixture' of generated and ground truth data
        :param real_data: the ground truth data
        :param gen_data: the generated/predicted data
        :return: gradient penalty
        """
        # get mixture of generated and ground truth data
        alpha = tf.random.normal([self.hparams["batch_size"], 1, 1, 1], 0., 1.)
        mix_data = real_data + alpha * (gen_data - real_data)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(mix_data)
            critic_mix = self.critic(mix_data, training=True)

        # calculate the gradient on the mixture data...
        grads_mix = gp_tape.gradient(critic_mix, [mix_data])[0]
        # ... and norm it
        norm = tf.sqrt(tf.reduce_mean(tf.square(grads_mix), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.) ** 2)

        return gp

    def recon_loss(self, real_data, gen_data):
        # initialize reconstruction loss
        rloss = 0.
        # get number of output heads (=2 if z_branch is activated)
        n = 1
        if self.hparams["z_branch"]:
            n = 2
        # get MAE for all output heads
        for i in range(n):
            rloss += tf.reduce_mean(tf.abs(tf.squeeze(gen_data[i]) - real_data[:, :, :, i]))

        return rloss

    # required for customized models, see here: https://www.tensorflow.org/guide/keras/save_and_serialize
    def get_config(self):
        return self.hparams

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def split_in_tar(da: xr.DataArray, target_var: str = "t2m") -> (xr.DataArray, xr.DataArray):
        """
        Split data array with variables-dimension into input and target data for downscaling.
        :param da: The unsplitted data array.
        :param target_var: Name of target variable which should consttute the first channel
        :return: The splitted data array.
        """
        invars = [var for var in da["variables"].values if var.endswith("_in")]
        tarvars = [var for var in da["variables"].values if var.endswith("_tar")]

        # ensure that ds_tar has a channel coordinate even in case of single target variable
        roll = False
        if len(tarvars) == 1:
            sl_tarvars = tarvars
        else:
            sl_tarvars = slice(*tarvars)
            if tarvars[0] != target_var:     # ensure that target variable appears as first channel
                roll = True

        da_in, da_tar = da.sel({"variables": invars}), da.sel(variables=sl_tarvars)
        if roll: da_tar = da_tar.roll(variables=1, roll_coords=True)

        return da_in, da_tar

    @staticmethod
    def get_hparams_dict(hparams_user: dict) -> dict:
        """
        Merge user-defined and default hyperparameter dictionary to retrieve a complete customized one.
        :param hparams_user: dictionary of hyperparameters parsed by user
        :return: merged hyperparameter dictionary
        """

        hparams_default = WGAN.get_hparams_default()

        # check if parsed hyperparameters are known
        unknown_keys = [key for key in hparams_user.keys() if key not in hparams_default]
        if unknown_keys:
            print("The following parsed hyperparameters are unknown and thus are ignored: {0}".format(
                ", ".join(unknown_keys)))

        # get complete hyperparameter dictionary while checking type of parsed values
        hparams_merged = {**hparams_default, **hparams_user}
        hparams_dict = {}
        for key in hparams_default:
            if isinstance(hparams_merged[key], type(hparams_default[key])):
                hparams_dict[key] = hparams_merged[key]
            else:
                raise TypeError("Parsed hyperparameter '{0}' must be of type '{1}', but is '{2}'"
                                .format(key, type(hparams_default[key]), type(hparams_merged[key])))

        if hparams_dict["optimizer"].lower() == "adam":
            adam = keras.optimizers.Adam
            hparams_dict["d_optimizer"] = adam(learning_rate=hparams_dict["lr"]/10., beta_1=0.0,
                                               beta_2=0.9)  # increase beta-values ?
            hparams_dict["g_optimizer"] = adam(learning_rate=hparams_dict["lr"], beta_1=0.0, beta_2=0.9)
        elif hparams_dict["optimizer"].lower() == "rmsprop":
            rmsprop = keras.optimizers.RMSprop
            hparams_dict["d_optimizer"] = rmsprop(lr=hparams_dict["lr"])  # increase beta-values ?
            hparams_dict["g_optimizer"] = rmsprop(lr=hparams_dict["lr"])
        else:
            raise ValueError("'{0}' is not a valid optimizer. Either choose Adam or RMSprop-optimizer")

        return hparams_dict

    @staticmethod
    def get_hparams_default() -> dict:
        """
        Return default hyperparameter dictionary.
        """
        hparams_dict = {"batch_size": 32, "lr_gen": 1.e-05, "lr_critic": 1.e-06, "train_epochs": 50, "z_branch": False,
                        "lr_decay": False, "decay_start": 5, "decay_end": 10, "lr_gen_end": 1.e-06,
                        "lr_critc_end": 1.e-07, "l_embed": False, "ngf": 56, "d_steps": 5, "recon_weight": 1000.,
                        "gp_weight": 10., "optimizer": "adam"}

        return hparams_dict

    @staticmethod
    def critic_loss(critic_real, critic_gen):
        c_loss = tf.reduce_mean(critic_gen - critic_real)

        return c_loss

    @staticmethod
    def critic_gen_loss(critic_gen):
        cg_loss = -tf.reduce_mean(critic_gen)

        return cg_loss


# auxiliary functions


def reshape_ds(ds):
    da = ds.to_array(dim="variables")  # .squeeze()
    da = da.transpose(..., "variables")
    return da


def z_norm_data(data, mu=None, std=None, dims=None, return_stat=False):
    if mu is None and std is None:
        if not dims:
            dims = list(data.dims)
        mu = data.mean(dim=dims)
        std = data.std(dim=dims)

    data_out = (data - mu) / std

    if return_stat:
        return data_out, mu, std
    else:
        return data_out


# for querying dictionary
def provide_default(dict_in, keyname, default=None, required=False):
    """
    Returns values of key from input dictionary or alternatively its default

    :param dict_in: input dictionary
    :param keyname: name of key which should be added to dict_in if it is not already existing
    :param default: default value of key (returned if keyname is not present in dict_in)
    :param required: Forces existence of keyname in dict_in (otherwise, an error is returned)
    :return: value of requested key or its default retrieved from dict_in
    """

    if not required and default is None:
        raise ValueError("Provide default when existence of key in dictionary is not required.")

    if keyname not in dict_in.keys():
        if required:
            print(dict_in)
            raise ValueError("Could not find '{0}' in input dictionary.".format(keyname))
        return default
    else:
        return dict_in[keyname]


# auxiliary function for colormap
def get_colormap_temp(levels=None):
    """
    Get a nice colormap for plotting topographic height
    :param levels: level boundaries
    :return cmap: colormap-object
    :return norm: normalization object corresponding to colormap and levels
    """
    bounds = np.asarray(levels)

    nbounds = len(bounds)
    col_obj = mpl.cm.PuOr_r(np.linspace(0., 1., nbounds))

    # create colormap and corresponding norm
    cmap = mpl.colors.ListedColormap(col_obj, name="temp" + "_map")
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm, bounds


# for making plot nice
def decorate_plot(ax_plot, plot_xlabel=True, plot_ylabel=True):
    fs = 16
    # add nice features and make plot appear nice
    ax_plot.coastlines(linewidth=0.75)
    ax_plot.coastlines(linewidth=0.75)

    ax_plot.add_feature(cartopy.feature.BORDERS)
    # adjust extent and ticks as well as axis-label
    ax_plot.set_xticks(np.arange(0., 360. + 0.1, 5.))  # ,crs=projection_crs)
    ax_plot.set_yticks(np.arange(-90., 90. + 0.1, 5.))  # ,crs=projection_crs)

    ax_plot.set_extent([3.5, 17., 44.5, 55.])  # , crs=prj_crs)
    ax_plot.minorticks_on()
    ax_plot.tick_params(axis="both", which="both", direction="out", labelsize=12)

    # some labels
    if plot_xlabel:
        ax_plot.set_xlabel("Longitude [°E]", fontsize=fs)
    if plot_ylabel:
        ax_plot.set_ylabel("Latitude[°N]", fontsize=fs)

    return ax_plot


# for creating plot
def create_plots(data1, data2, plt_name, opt_plot={}):
    # get coordinate data
    try:
        time, lat, lon = data1["time"].values, data1["lat"].values, data1["lon"].values
        time_stamp = (pd.to_datetime(time)).strftime("%Y-%m-%d %H:00 UTC")
    except Exception as err:
        print("Failed to retrieve coordinates from data1")
        raise err
    # construct array for edges of grid points
    dy, dx = np.round((lat[1] - lat[0]), 2), np.round((lon[1] - lon[0]), 2)
    lat_e, lon_e = np.arange(lat[0] - dy / 2, lat[-1] + dy, dy), np.arange(lon[0] - dx / 2, lon[-1] + dx, dx)

    title1, title2 = provide_default(opt_plot, "title1", "input T2m"), provide_default(opt_plot, "title2", "target T2m")
    title1, title2 = "{0}, {1}".format(title1, time_stamp), "{0}, {1}".format(title2, time_stamp)
    levels = provide_default(opt_plot, "levels", np.arange(-5., 25., 1.))

    # get colormap
    cmap_temp, norm_temp, lvl = get_colormap_temp(levels)
    # create plot objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True,
                                   subplot_kw={"projection": ccrs.PlateCarree()})

    # perform plotting
    temp1 = ax1.pcolormesh(lon_e, lat_e, np.squeeze(data1.values - 273.15), cmap=cmap_temp, norm=norm_temp)
    temp2 = ax2.pcolormesh(lon_e, lat_e, np.squeeze(data2.values - 273.15), cmap=cmap_temp, norm=norm_temp)

    ax1, ax2 = decorate_plot(ax1), decorate_plot(ax2, plot_ylabel=False)

    ax1.set_title(title1, size=14)
    ax2.set_title(title2, size=14)

    # add colorbar
    cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(temp2, cax=cax, orientation="vertical", ticks=lvl[1::2])
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(plt_name+".png")
    plt.close(fig)


def main(parser_args):

    outdir="../trained_models/"
    z_branch = not parser_args.no_z_branch

    datadir = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_ifs/netcdf_data/all_files/"

    ds_train, ds_val, ds_test = xr.open_dataset(os.path.join(datadir, "era5_to_ifs_train_corrected.nc")), \
                                xr.open_dataset(os.path.join(datadir, "era5_to_ifs_val_corrected.nc")), \
                                xr.open_dataset(os.path.join(datadir, "era5_to_ifs_test_corrected.nc"))

    print("Datasets for trining, validation and testing loaded.")

    wgan_model = WGAN(build_unet, critic_model,
                      {"lr_decay": parser_args.lr_decay, "lr": parser_args.lr,
                       "train_epochs": parser_args.nepochs, "recon_weight": parser_args.recon_wgt,
                       "d_steps": parser_args.d_steps,
                       "optimizer": parser_args.optimizer, "z_branch": z_branch})

    # prepare data
    da_train, da_val = reshape_ds(ds_train), reshape_ds(ds_val)

    norm_dims = ["time", "lat", "lon"]
    da_train, mu_train, std_train = z_norm_data(da_train, dims=norm_dims, return_stat=True)
    da_val = z_norm_data(da_val, mu=mu_train, std=std_train)

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

    # do predictions
    da_test = reshape_ds(ds_train)
    da_test = z_norm_data(da_test, dims=norm_dims)

    da_test_in, da_test_tar = WGAN.split_in_tar(da_test)
    test_iter = tf.data.Dataset.from_tensor_slices((da_test_in, da_test_tar))
    test_iter = test_iter.batch(wgan_model.hparams["batch_size"])

    y_pred = wgan_model.predict(test_iter, batch_size=wgan_model.hparams["batch_size"],verbose=1)

    # denorm data from predictions and convert to xarray
    coords = da_test_tar.isel(variables=0).squeeze().coords
    dims = da_test_tar.isel(variables=0).squeeze().dims

    y_pred_trans = xr.DataArray(y_pred[0].squeeze(), coords=coords, dims=dims)

    y_pred_trans = y_pred_trans.squeeze()*std_train.sel(variables="t2m_tar").squeeze().values + \
                   mu_train.sel(variables="t2m_tar").squeeze().values
    y_pred_trans = xr.DataArray(y_pred_trans, coords=coords, dims=dims)

    # create plot
    tind = 0
    plt_name = os.path.join(model_savedir, "plot_tind_{0:d}_{1}.pdf".format(tind, os.path.join(outdir, parser_args["model_name"])))
    create_plots(y_pred_trans.isel(time=tind), ds_train["t2m_tar"].isel(time=tind), plt_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
