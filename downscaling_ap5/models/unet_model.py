# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Methods to set-up U-net models incl. its building blocks.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2021-XX-XX"
__update__ = "2022-11-25"

# import modules
import os
from typing import List
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
# all the layers used for U-net
from tensorflow.keras.layers import (Concatenate, Conv2D, Conv2DTranspose, Input, MaxPool2D, BatchNormalization,
                                     Activation)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping

import advanced_activations
from other_utils import to_list

# building blocks for Unet


def conv_block(inputs, num_filters: int, kernel: tuple = (3, 3), strides: tuple = (1, 1), padding: str = "same",
               activation: str = "relu", activation_args=None, kernel_init: str = "he_normal",
               l_batch_normalization: bool = True):
    """
    A convolutional layer with optional batch normalization
    :param inputs: the input data with dimensions nx, ny and nc
    :param num_filters: number of filters (output channel dimension)
    :param kernel: tuple for convolution kernel size
    :param strides: tuple for stride of convolution
    :param padding: technique for padding (e.g. "same" or "valid")
    :param activation: activation fuction for neurons (e.g. "relu")
    :param activation_args: arguments for activation function given that advanced layers are applied
    :param kernel_init: initialization technique (e.g. "he_normal" or "glorot_uniform")
    :param l_batch_normalization: flag if batch normalization should be applied
    """
    x = Conv2D(num_filters, kernel, strides=strides, padding=padding, kernel_initializer=kernel_init)(inputs)
    if l_batch_normalization:
        x = BatchNormalization()(x)

    try:
        x = Activation(activation)(x)
    except ValueError:
        ac_layer = advanced_activations(activation, *activation_args)
        x = ac_layer(x)

    return x


def conv_block_n(inputs, num_filters: int, n: int = 2, kernel: tuple = (3, 3), strides: tuple = (1, 1),
                 padding: str = "same", activation: str = "relu", activation_args=None,
                 kernel_init: str = "he_normal", l_batch_normalization: bool = True):
    """
    Sequential application of two convolutional layers (using conv_block).
    :param inputs: the input data with dimensions nx, ny and nc
    :param num_filters: number of filters (output channel dimension)
    :param n: number of convolutional blocks
    :param kernel: tuple for convolution kernel size
    :param strides: tuple for stride of convolution
    :param padding: technique for padding (e.g. "same" or "valid")
    :param activation: activation fuction for neurons (e.g. "relu")
    :param activation_args: arguments for activation function given that advanced layers are applied
    :param kernel_init: initialization technique (e.g. "he_normal" or "glorot_uniform")
    :param l_batch_normalization: flag if batch normalization should be applied
    """
    x = conv_block(inputs, num_filters, kernel, strides, padding, activation, activation_args,
                   kernel_init, l_batch_normalization)
    for _ in np.arange(n - 1):
        x = conv_block(x, num_filters, kernel, strides, padding, activation, activation_args,
                       kernel_init, l_batch_normalization)

    return x


def encoder_block(inputs, num_filters, kernel_maxpool: tuple = (2, 2), l_large: bool = True):
    """
    One complete encoder-block used in U-net.
    :param inputs: input to encoder block
    :param num_filters: number of filters/channel to be used in convolutional blocks
    :param kernel_maxpool: kernel used in max-pooling
    :param l_large: flag for large encoder block (two consecutive convolutional blocks)
    """
    if l_large:
        x = conv_block_n(inputs, num_filters, n=2)
    else:
        x = conv_block(inputs, num_filters)

    p = MaxPool2D(kernel_maxpool)(x)

    return x, p


def subpixel_block(inputs, num_filters, kernel: tuple = (3,3), upscale_fac: int = 2,
                   padding: str = "same", activation: str = "relu", kernel_init: str = "he_normal"):

    x = Conv2D(num_filters * (upscale_fac ** 2), kernel, padding=padding, kernel_initializer=kernel_init,
               activation=activation)(inputs)
    x = tf.nn.depth_to_space(x, upscale_fac)

    return x


def decoder_block(inputs, skip_features, num_filters, kernel: tuple = (3, 3), strides_up: int = 2,
                  subpixel_layer: bool = True, padding: str = "same", activation="relu", kernel_init="he_normal",
                  l_batch_normalization: bool = True):
    """
    One complete decoder block used in U-net (reverting the encoder)
    """
    if subpixel_layer:
        x = subpixel_block(inputs, num_filters, kernel, upscale_fac=strides_up, padding=padding,
                           activation=activation, kernel_init=kernel_init)
    else:
        x = Conv2DTranspose(num_filters, (strides_up, strides_up), strides=strides_up, padding="same")(inputs)

    x = Concatenate()([x, skip_features])
    x = conv_block_n(x, num_filters, 2, kernel, (1, 1), padding, activation, kernel_init=kernel_init, 
                     l_batch_normalization=l_batch_normalization)

    return x


# The particular U-net
def sha_unet(input_shape: tuple, channels_start: int = 56, z_branch: bool = False, subpixel_layer: bool = True,
             tar_channels=["output_temp", "output_z"]) -> Model:
    """
    Builds up U-net model architecture adapted from Sha et al., 2020 (see https://doi.org/10.1175/JAMC-D-20-0057.1).
    :param input_shape: shape of input-data
    :param channels_start: number of channels to use as start in encoder blocks
    :param z_branch: flag if z-branch is used.
    :param subpixel_layer: flag if subpixel layer instead of transposed convolution is used for upsampling
    :param tar_channels: name of output/target channels (needed for associating losses during compilation)
    :return:
    """
    inputs = Input(input_shape)

    """ encoder """
    s1, e1 = encoder_block(inputs, channels_start, l_large=True)
    s2, e2 = encoder_block(e1, channels_start * 2, l_large=False)
    s3, e3 = encoder_block(e2, channels_start * 4, l_large=False)

    """ bridge encoder <-> decoder """
    b1 = conv_block(e3, channels_start * 8)

    """ decoder """
    d1 = decoder_block(b1, s3, channels_start * 4, subpixel_layer=subpixel_layer)
    d2 = decoder_block(d1, s2, channels_start * 2, subpixel_layer=subpixel_layer)
    d3 = decoder_block(d2, s1, channels_start, subpixel_layer=subpixel_layer)

    output_temp = Conv2D(1, (1, 1), kernel_initializer="he_normal", name=tar_channels[0])(d3)
    if z_branch:
        print("Use z_branch...")
        output_z = Conv2D(1, (1, 1), kernel_initializer="he_normal", name=tar_channels[1])(d3)

        model = Model(inputs, [output_temp, output_z], name="t2m_downscaling_unet_with_z")
    else:
        model = Model(inputs, output_temp, name="t2m_downscaling_unet")

    return model


class UNET(keras.Model):
    """
    U-Net submodel class:
    This subclass takes a U-Net implemented using Keras functional API as input to the instanciation.
    """
    def __init__(self, unet_model: keras.Model, shape_in: List, hparams: dict, savedir: str,
                 exp_name: str = "unet_model"):

        super(UNET, self).__init__()

        self.unet = unet_model
        self.shape_in = shape_in
        self.hparams = UNET.get_hparams_dict(hparams)
        if self.hparams["l_embed"]:
            raise ValueError("Embedding is not implemented yet.")
        self.modelname = exp_name
        if not os.path.isdir(savedir):
            os.makedirs(savedir, exist_ok=True)
        self.savedir = savedir

    def call(self, inputs, **kwargs):
        """
        Integrate call-function to make all built-in functions available.
        See https://stackoverflow.com/questions/65318036/is-it-possible-to-use-the-tensorflow-keras-functional-api-train_unet-model-err.6387845within-a-subclassed-mo
        for a reference how a model based on Keras functional API has to be integrated into a subclass.
        """
        return self.unet(inputs, **kwargs)

    def get_compile_opts(self):

        custom_opt = self.hparams["optimizer"].lower()
        varnames_tar = self.varnames_tar

        if custom_opt == "adam":
            opt = keras.optimizers.Adam
        elif custom_opt == "rmsprop":
            opt = keras.optimizers.RMSprop

        opt_dict = {"optimizer": opt(lr=self.hparams["lr"])}

        if self.hparams["z_branch"]:
            opt_dict["loss"] = {varnames_tar[0]: self.hparams["loss_func"],
                                varnames_tar[1]: self.hparams["loss_func"]}
            opt_dict["loss_weights"] = {varnames_tar[0]: self.hparams["loss_weights"][0],
                                        varnames_tar[1]: self.hparams["loss_weights"][1]}
        else:
            opt_dict["loss"] = self.hparams["loss_func"]

        return opt_dict

    def compile(self, **kwargs):
        # instantiate model
        self.unet = self.unet(self.shape_in, z_branch=self.hparams["z_branch"], tar_channels=to_list(self.varnames_tar),
                              subpixel_layer= self.hparams["subpixel_layer"])

        return self.unet.compile(**kwargs)
       # return super(UNET, self).compile(**kwargs)

    def get_lr_scheduler(self):
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

    def get_fit_opts(self):
        """
        Add customized learning rate scheduler, checkpointing and early stopping to list of callbacks.
        """
        callback_list = []
        if self.hparams["lr_decay"]:
            callback_list = callback_list + [LearningRateScheduler(self.get_lr_scheduler(), verbose=1)]

        if self.hparams["lscheduled_train"]:
            callback_list = callback_list + [ModelCheckpoint(self.savedir, monitor="val_loss", verbose=1,
                                                             save_best_only=True, mode="min")]
            #                                + EarlyStopping(monitor="val_recon_loss", patience=8)]

        unet_callbacks = {"unet_callbacks": callback_list}

        return unet_callbacks

    def fit(self, callbacks=None, unet_callbacks: List = None, **kwargs):
        """
        Takes all (non-positional) arguments of Keras fit-method, but expands the list of callbacks to include
        the UNET callbacks (see get_fit_opt-method)
        """
        default_callbacks, unet_callbacks = to_list(callbacks), to_list(unet_callbacks)
        all_callbacks = [e for e in unet_callbacks + default_callbacks if e is not None]

        return self.unet.fit(callbacks=all_callbacks, **kwargs)
        #return super(UNET, self).fit(callbacks=all_callbacks, **kwargs)

    def save(self, **kwargs):
        self.unet.save(**kwargs)

    @staticmethod
    def get_hparams_dict(hparams_user: dict) -> dict:
        """
        Merge user-defined and default hyperparameter dictionary to retrieve a complete customized one
        :param hparams_user: dictionary of hyperparameters parsed by user
        :return: merged hyperparameter dictionary
        """

        hparams_default = UNET.get_hparams_default()

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

        return hparams_dict

    @staticmethod
    def get_hparams_default() -> dict:
        """
        Return default hyperparameter dictionary.
        """
        hparams_dict = {"batch_size": 32, "lr": 5.e-05, "nepochs": 70, "z_branch": True, "loss_func": "mae",
                        "loss_weights": [1.0, 1.0], "lr_decay": False, "decay_start": 5, "decay_end": 30,
                        "lr_end": 1.e-06, "l_embed": False, "ngf": 56, "optimizer": "adam", "lscheduled_train": True,
                        "var_tar2in": "", "subpixel_layer": True}

        return hparams_dict

