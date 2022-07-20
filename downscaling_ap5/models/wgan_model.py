__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-05-19"
__update__ = "2022-06-28"

import os, glob
from collections import OrderedDict
import random
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler

# all the layers used for U-net
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPool2D, Dense, Flatten, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
# other modules
import xarray as xr
import numpy as np

from unet_model import build_unet, conv_block
from other_utils import subset_files_on_date, to_list

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
        self.shape_in, self.shape_tar, self.nsamples = None, None, None
        self.lr_scheduler = None
        self.c_optimizer, self.g_optimizer = None, None

    def compile(self, data_dir, months_train, months_val, predictors, predictands):
        """
        Turn datasets into tf.Datasets, compile model and set learning rate schedule (optionally)
        :param data_dir: Data Array providing the training data (must have variables-dimension)
        :param months_train: list of months used to build training dataset (strings with format <YYYY>-<MM>)
        :param months_val: list of months used to build validation dataset (strings with format <YYYY>-<MM>)
        :param predictors: list of predictor variables
        :param predictands: list of predictand (target) variable(-s)
        :return: tf.Datasets for training and validation data
        """
        self.nsamples, mu_all, std_all = WGAN.get_data_statistics(data_dir, months_train, predictors + predictands)

        norm_dict = {"mu_in": mu_all[predictors].to_array().as_numpy(), "std_in": std_all[predictors].to_array().as_numpy(),
                     "mu_tar": mu_all[predictands].to_array().as_numpy(), "std_tar": std_all[predictands].to_array().as_numpy()}

        # set-up dataset iterators for traing and validation dataset
        train_iter, data_shp = self.make_data_generator(data_dir, months_train, predictors, predictands, norm_dict)
        val_iter, _ = self.make_data_generator(data_dir, months_val, predictors, predictands, norm_dict)

        self.shape_in, self.shape_tar = data_shp["shape_in"], data_shp["shape_tar"]

        # instantiate models
        self.generator = self.generator(self.shape_in, channels_start=self.hparams["ngf"],
                                        z_branch=self.hparams["z_branch"])
        self.critic = self.critic(self.shape_tar)

        # call Keras compile method
        super(WGAN, self).compile()
        # set optimizers
        self.c_optimizer, self.g_optimizer = self.hparams["d_optimizer"], self.hparams["g_optimizer"]

        # get learning rate schedule if desired
        if self.hparams["lr_decay"]:
            self.lr_scheduler = LearningRateSchedulerWGAN(self.get_lr_decay(), verbose=1)

        return train_iter, val_iter

    def get_lr_decay(self):
        """
        Get callable of learning rate scheduler which can be used as callabck in Keras models.
        Exponential decay is applied to change the learning rate from the start to the end value.
        Note that the exponential decay is calculated based on the learning rate of the generator, but applies to both.
        :return: learning rate scheduler
        """
        decay_st, decay_end = self.hparams["decay_start"], self.hparams["decay_end"]
        lr_start, lr_end = self.hparams["lr_gen"], self.hparams["lr_gen_end"]

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

    def make_data_generator(self, data_dir: str, month_list: List, predictors: List, predictands: List, norm_data: dict,
                            shuffle: bool = True, embed=None, seed: int = 42) -> tf.data.Dataset:
        """
        Creates TensorFlow dataset for input ipeline to neural networks.
        :param data_dir: directory where monthly netCDF-files are saved (e.g. preproc_2016-01.nc)
        :param month_list: list of months (strings with format <YYYY>-<MM>)
        :param predictors: list of predictor variables
        :param predictands: list of predictands
        :param norm_data: dictionary containing data statistics for normalization (keys: mu_in, std_in, mu_tar, std_tar)
        :param shuffle: Flag to perform shuffling on the dataset
        :param embed: embedding of daytime and month (not implemented yet)
        :param seed: seed for random shuffling
        """

        nc_files_dir = glob.glob(os.path.join(data_dir, "preproc*.nc"))

        all_vars = to_list(predictors) + to_list(predictands)

        # filter files based on months of interest
        nc_files = []
        for yr_mm in month_list:
            nc_files = nc_files + subset_files_on_date(nc_files_dir, yr_mm, date_alias="%Y-%m")

        if not nc_files:
            raise FileNotFoundError("Could not find any datafiles under '{0}' containing data for the months: {1}"
                                    .format(data_dir, ", ".join(month_list)))

        # shuffle if desired
        if shuffle:
            random.seed(seed)
            random.shuffle(nc_files)

        # auxiliary function for generator
        def gen(nc_files_ds):

            for file in nc_files_ds:
                ds = xr.open_dataset(file, engine='netcdf4')
                ds = ds[all_vars].astype(np.float32)
                ntimes = len(ds["time"])
                for t in range(ntimes):
                    ds_t = ds.isel({"time": t})
                    in_data, tar_data = ds_t[predictors].to_array(dim="variables").transpose(..., "variables"), \
                                        ds_t[predictands].to_array(dim="variables").transpose(..., "variables")
                    yield tuple((in_data.values, tar_data.values))

        s0 = next(iter(gen(nc_files)))
        gen_mod = gen(nc_files)
        # NOTE: critic only accounts for 1st channel (= downscaling target) -> shape_tar-value set accordingly
        sample_shp = {"shape_in": s0[0].shape, "shape_tar": (*s0[1].shape[:-1], 1)}

        # create TF dataset from generator function
        tfds_dat = tf.data.Dataset.from_generator(lambda: gen_mod,
                                                  output_signature=(tf.TensorSpec(sample_shp["shape_in"], dtype=s0[0].dtype),
                                                                    tf.TensorSpec(sample_shp["shape_tar"], dtype=s0[1].dtype)))

        # Define normalization function to be applied to mini-batches...
        def normalize_batch(batch: tuple, norm_dict):

            mu_in, std_in = tf.constant(norm_dict["mu_in"], dtype=batch[0].dtype), \
                            tf.constant(norm_dict["std_in"], dtype=batch[0].dtype)
            mu_tar, std_tar = tf.constant(norm_dict["mu_tar"], dtype=batch[1].dtype), \
                              tf.constant(norm_dict["std_tar"], dtype=batch[1].dtype)

            in_normed, tar_normed = tf.divide(tf.subtract(batch[0], mu_in), std_in), \
                                    tf.divide(tf.subtract(batch[1], mu_tar), std_tar)

            return in_normed, tar_normed

        def parse_example(in_data, tar_data):
            return normalize_batch((in_data, tar_data), norm_data)

        # ...and configure dataset
        tfds_dat = tfds_dat.shuffle(buffer_size=20000, seed=seed).batch(self.hparams["batch_size"]).map(parse_example)
        tfds_dat = tfds_dat.repeat(self.hparams["batch_size"] * (self.hparams["d_steps"] + 1)).prefetch(1000)

        return tfds_dat, sample_shp

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
    def get_data_statistics(data_dir: str, month_list: List, variables: List) -> (int, xr.Dataset, xr.Dataset):
        """
        Get number of samples from several monthly netCDF-files using open_mfdataset.
        :param data_dir: directory where monthly netCDF-files are saved (e.g. preproc_2016-01.nc)
        :param month_list: list of months (strings with format <YYYY>-<MM>)
        :param variables: List of variables for which mean and standard deviation should be computed
        :return: number of samples from scanned files, mean and standard deviations of variables
        """
        nc_files_dir = glob.glob(os.path.join(data_dir, "preproc*.nc"))
        # filter files based on months of interest
        nc_files = []
        for yr_mm in month_list:
            nc_files = nc_files + subset_files_on_date(nc_files_dir, yr_mm, date_alias="%Y-%m")

        data_all = xr.open_mfdataset(nc_files)

        nsamples = len(data_all["time"])
        norm_dims = ["time", "lat", "lon"]
        mu_data, std_data = data_all[variables].mean(dim=norm_dims), data_all[variables].std(dim=norm_dims)

        return nsamples, mu_data.astype(np.float32).compute(), std_data.astype(np.float32).compute()

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
            hparams_dict["d_optimizer"] = adam(learning_rate=hparams_dict["lr_critic"], beta_1=0.0, beta_2=0.9)
            hparams_dict["g_optimizer"] = adam(learning_rate=hparams_dict["lr_gen"], beta_1=0.0, beta_2=0.9)
        elif hparams_dict["optimizer"].lower() == "rmsprop":
            rmsprop = keras.optimizers.RMSprop
            hparams_dict["d_optimizer"] = rmsprop(lr=hparams_dict["lr_critic"])  # increase beta-values ?
            hparams_dict["g_optimizer"] = rmsprop(lr=hparams_dict["lr_gen"])
        else:
            raise ValueError("'{0}' is not a valid optimizer. Either choose Adam or RMSprop-optimizer")

        return hparams_dict

    @staticmethod
    def get_hparams_default() -> dict:
        """
        Return default hyperparameter dictionary.
        """
        hparams_dict = {"batch_size": 32, "lr_gen": 1.e-05, "lr_critic": 1.e-06, "train_epochs": 50, "z_branch": False,
                        "lr_decay": False, "decay_start": 5, "decay_end": 10, "lr_gen_end": 1.e-06, "l_embed": False,
                        "ngf": 56, "d_steps": 5, "recon_weight": 1000., "gp_weight": 10., "optimizer": "adam"}

        return hparams_dict

    @staticmethod
    def critic_loss(critic_real, critic_gen):
        c_loss = tf.reduce_mean(critic_gen - critic_real)

        return c_loss

    @staticmethod
    def critic_gen_loss(critic_gen):
        cg_loss = -tf.reduce_mean(critic_gen)

        return cg_loss


class LearningRateSchedulerWGAN(LearningRateScheduler):

    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerWGAN, self).__init__(schedule, verbose)

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, "g_optimizer"):
            raise AttributeError('Model must have a "g_optimizer" for optimizing the generator.')

        if not hasattr(self.model, "c_optimizer"):
            raise AttributeError('Model must have a "c_optimizer" for optimizing the critic.')

        if not (hasattr(self.model.g_optimizer, "lr") and hasattr(self.model.c_optimizer, "lr")):
          raise ValueError('Optimizer for generator and critic must both have a "lr" attribute.')
        try:  # new API
          lr_g, lr_c = float(K.get_value(self.model.g_optimizer.lr)), \
                       float(K.get_value(self.model.c_optimizer.lr))
          lr_g, lr_c = self.schedule(epoch, lr_g), self.schedule(epoch, lr_c)
        except TypeError:  # Support for old API for backward compatibility
          raise NotImplementedError("WGAN learning rate schedule is not compatible with old API. Update TF Keras.")

        if not (isinstance(lr_g, (tf.Tensor, float, np.float32, np.float64)) and
                isinstance(lr_c, (tf.Tensor, float, np.float32, np.float64))):
            raise ValueError('The output of the "schedule" function '
                             f'should be float. Got: {lr_g} (generator) and {lr_c} (critic)' )
        if isinstance(lr_g, tf.Tensor) and not lr_g.dtype.is_floating \
           and isinstance(lr_c, tf.Tensor) and lr_c.dtype.is_floating:
            raise ValueError(
                f'The dtype of `lr_g` and `lr_c` Tensor should be float. Got: {lr_g.dtype} (generator)'
                f'and {lr_c.dtype} (critic)' )
        # set updated learning rate
        K.set_value(self.model.g_optimizer.lr, K.get_value(lr_g))
        K.set_value(self.model.c_optimizer.lr, K.get_value(lr_c))
        if self.verbose > 0:
            print(f'\nEpoch {epoch + 1}: LearningRateScheduler setting learning '
                f'rate for generator to {lr_g}, for critic to {lr_c}.')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr_generator'] = K.get_value(self.model.g_optimizer.lr)
        logs['lr_critic'] = K.get_value(self.model.c_optimizer.lr)

