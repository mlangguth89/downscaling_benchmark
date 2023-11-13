# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Methods to set-up U-net models incl. its building blocks.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2021-XX-XX"
__update__ = "2023-11-13"

# import modules
import os
from typing import List
import inspect
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
# all the layers used for U-net
from tensorflow.keras.layers import (Concatenate, Conv2D, Conv2DTranspose, Input, MaxPool2D, BatchNormalization,
                                     Activation, AveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping

from advanced_activations import advanced_activation
from other_utils import to_list

# building blocks for Unet


def conv_block(inputs, num_filters: int, kernel: tuple = (3, 3), strides: tuple = (1, 1), padding: str = "same",
               activation: str = "swish", activation_args={}, kernel_init: str = "he_normal",
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
        ac_layer = advanced_activation(activation, *activation_args)
        x = ac_layer(x)

    return x


def conv_block_n(inputs, num_filters: int, n: int = 2, **kwargs):
    """
    Sequential application of two convolutional layers (using conv_block).
    :param inputs: the input data with dimensions nx, ny and nc
    :param num_filters: number of filters (output channel dimension)
    :param n: number of convolutional blocks
    :param kwargs: keyword arguments for conv_block
    """
    x = conv_block(inputs, num_filters, **kwargs)
    for _ in np.arange(n - 1):
        x = conv_block(x, num_filters, **kwargs)

    return x


def encoder_block(inputs, num_filters, l_large: bool = True, kernel_pool: tuple = (2, 2), l_avgpool: bool = False, **kwargs):
    """
    One complete encoder-block used in U-net.
    :param inputs: input to encoder block
    :param num_filters: number of filters/channel to be used in convolutional blocks
    :param l_large: flag for large encoder block (two consecutive convolutional blocks)
    :param kernel_maxpool: kernel used in max-pooling
    :param l_avgpool: flag if average pooling is used instead of max pooling
    :param kwargs: keyword arguments for conv_block
    """
    if l_large:
        x = conv_block_n(inputs, num_filters, n=2, **kwargs)
    else:
        x = conv_block(inputs, num_filters, **kwargs)

    if l_avgpool:
        p = AveragePooling2D(kernel_pool)(x)
    else:
        p = MaxPool2D(kernel_pool)(x)

    return x, p

def subpixel_block(inputs, num_filters, kernel: tuple = (3,3), upscale_fac: int = 2,
        padding: str = "same", activation: str = "swish", activation_args: dict = {},
        kernel_init: str = "he_normal"):

    x = Conv2D(num_filters * (upscale_fac ** 2), kernel, padding=padding, kernel_initializer=kernel_init,
               activation=None)(inputs)
    try:
        x = Activation(activation)(x)
    except ValueError:
        ac_layer = advanced_activation(activation, *activation_args)
        x = ac_layer(x)
    
    x = tf.nn.depth_to_space(x, upscale_fac)

    return x



def decoder_block(inputs, skip_features, num_filters, strides_up: int = 2, l_subpixel: bool = False, **kwargs_conv_block):
    """
    One complete decoder block used in U-net (reverting the encoder)
    """
    if l_subpixel:
        x = subpixel_block(inputs, num_filters, upscale_fac=strides_up, **kwargs_conv_block)
    else:
        x = Conv2DTranspose(num_filters, (strides_up, strides_up), strides=strides_up, padding="same")(inputs)
        
        activation = kwargs_conv_block.get("activation", "relu")
        activation_args = kwargs_conv_block.get("activation_args", {})

        try:
            x = Activation(activation)(x)
        except ValueError:
            ac_layer = advanced_activation(activation, *activation_args)
            x = ac_layer(x)

    x = Concatenate()([x, skip_features])
    x = conv_block_n(x, num_filters, 2, **kwargs_conv_block)

    return x


# The particular U-net
def sha_unet(input_shape: tuple, n_predictands_dyn: int, hparams_unet: dict, concat_out: bool = False,
             tar_channels=["output_dyn", "output_z"]) -> Model:
    """
    Builds up U-net model architecture adapted from Sha et al., 2020 (see https://doi.org/10.1175/JAMC-D-20-0057.1).
    :param input_shape: shape of input-data
    :param channels_start: number of channels to use as start in encoder blocks
    :param n_predictands: number of target variables (dynamic output variables)
    :param z_branch: flag if z-branch is used.
    :param advanced_unet: flag if advanced U-net is used (LeakyReLU instead of ReLU, average pooling instead of max pooling and subpixel-layer)
    :param concat_out: boolean if output layers will be concatenated (disables named target channels!)
    :param tar_channels: name of output/target channels (needed for associating losses during compilation)
    :return:
    """
    # basic configuration of U-Net 
    channels_start = hparams_unet["ngf"]
    z_branch = hparams_unet["z_branch"]
    kernel_pool = hparams_unet["kernel_pool"]   
    l_avgpool = hparams_unet["l_avgpool"]
    l_subpixel = hparams_unet["l_subpixel"]

    config_conv = {"kernel": hparams_unet["kernel"], "strides": hparams_unet["strides"], "padding": hparams_unet["padding"], 
                   "activation": hparams_unet["activation"], "activation_args": hparams_unet["activation_args"], 
                   "kernel_init": hparams_unet["kernel_init"], "l_batch_normalization": hparams_unet["l_batch_normalization"]}

    # build U-Net
    inputs = Input(input_shape)

    """ encoder """
    s1, e1 = encoder_block(inputs, channels_start, l_large=True, kernel_pool=kernel_pool, l_avgpool=l_avgpool,**config_conv)
    s2, e2 = encoder_block(e1, channels_start * 2, l_large=False, kernel_pool=kernel_pool, l_avgpool=l_avgpool,**config_conv)
    s3, e3 = encoder_block(e2, channels_start * 4, l_large=False, kernel_pool=kernel_pool, l_avgpool=l_avgpool,**config_conv)

    """ bridge encoder <-> decoder """
    b1 = conv_block(e3, channels_start * 8, **config_conv)

    """ decoder """
    d1 = decoder_block(b1, s3, channels_start * 4, l_subpixel=l_subpixel, **config_conv)
    d2 = decoder_block(d1, s2, channels_start * 2, l_subpixel=l_subpixel, **config_conv)
    d3 = decoder_block(d2, s1, channels_start, l_subpixel=l_subpixel, **config_conv)

    output_dyn = Conv2D(n_predictands_dyn, (1, 1), kernel_initializer=config_conv["kernel_init"], name=tar_channels[0])(d3)
    if z_branch:
        print("Use z_branch...")
        output_static = Conv2D(1, (1, 1), kernel_initializer=config_conv["kernel_init"], name=tar_channels[1])(d3)

        if concat_out:
            model = Model(inputs, tf.concat([output_dyn, output_static], axis=-1), name="downscaling_unet_with_z")
        else:
            model = Model(inputs, [output_dyn, output_static], name="downscaling_unet_with_z")
    else:
        model = Model(inputs, output_dyn, name="downscaling_unet")

    return model


class UNET(keras.Model):
    """
    U-Net submodel class:
    This subclass takes a U-Net implemented using Keras functional API as input to the instanciation.
    """
    def __init__(self, unet_model: keras.Model, shape_in: List, varnames_tar: List, hparams: dict, savedir: str,
                 exp_name: str = "unet_model"):

        super(UNET, self).__init__()

        self.unet = unet_model
        self.shape_in = shape_in
        self.varnames_tar = varnames_tar
        self.hparams = UNET.get_hparams_dict(hparams)
        self.n_predictands = len(varnames_tar)                      # number of predictands
        self.n_predictands_dyn = self.n_predictands - 1 if self.hparams["z_branch"] else self.n_predictands
        if self.hparams.get("l_embed", False):
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

        if custom_opt == "adam":
            opt = keras.optimizers.Adam
        elif custom_opt == "rmsprop":
            opt = keras.optimizers.RMSprop

        opt_dict = {"optimizer": opt(learning_rate=self.hparams["lr"])}

        if self.hparams["z_branch"]:
            opt_dict["loss"] = {f"{var}": self.hparams["loss_func"] for var in self.varnames_tar}
            opt_dict["loss_weights"] = {f"{var}": self.hparams["loss_weights"][i] for i, var in
                                        enumerate(self.varnames_tar)}
        else:
            opt_dict["loss"] = self.hparams["loss_func"]

        return opt_dict

    def compile(self, **kwargs):

        # instantiate model
        if self.hparams["z_branch"]:     # model has named branches (see also opt_dict in get_compile_opts)
            tar_channels = [f"{var}" for var in self.varnames_tar]
            self.unet = self.unet(self.shape_in, self.n_predictands_dyn, self.hparams,
                                  concat_out=False, tar_channels=tar_channels)
        else:
            self.unet = self.unet(self.shape_in, self.n_predictands_dyn, self.hparams)

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
            savedir_best = os.path.join(self.savedir, f"{self.modelname}_best")
            os.makedirs(savedir_best, exist_ok=True)
            callback_list = callback_list + [ModelCheckpoint(savedir_best, monitor="val_t_2m_tar_loss", verbose=1,
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

    def plot_model(self, **kwargs):
        self.unet.plot_model(**kwargs)

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

        hparams_dict = {"kernel": (3, 3), "strides": (1, 1), "padding": "same", "activation": "relu", "activation_args": {},      # arguments for building blocks of U-Net:
                        "kernel_init": "he_normal", "l_batch_normalization": True, "kernel_pool": (2, 2), "l_avgpool": False,     # see keyword-aguments of sha_unet, conv_block,
                        "l_subpixel": False, "z_branch": True, "ngf": 56,                                                         # encoder_block and decoder_block
                        "batch_size": 32, "lr": 5.e-05, "nepochs": 70, "loss_func": "mae", "loss_weights": [1.0, 1.0],            # training parameters
                        "lr_decay": False, "decay_start": 5, "decay_end": 30, "lr_end": 1.e-06, "l_embed": False, 
                        "optimizer": "adam", "lscheduled_train": True}

        return hparams_dict

