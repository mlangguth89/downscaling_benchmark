# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Class for building blocks of U-Net as well as model classes for Sha U-Net and DeepRu.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2021-XX-XX"
__update__ = "2023-12-14"

# import modules
import os
from typing import List
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
# all the layers used for U-net
from tensorflow.keras.layers import (Concatenate, Conv2D, Conv2DTranspose, Input, MaxPool2D, BatchNormalization,
                                     Activation, AveragePooling2D, Add, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping

from abstract_model_class import AbstractModelClass
from advanced_activations import advanced_activation
from custom_losses import get_custom_loss


# class for building blocks of U-Net 
class UNetModelBase:
    """
    This class contains all building blocks of U-Nets
    """
    @staticmethod
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

    def conv_block_n(self, inputs, num_filters: int, n: int = 2, **kwargs):
        """
        Sequential application of two convolutional layers (using conv_block).
        :param inputs: the input data with dimensions nx, ny and nc
        :param num_filters: number of filters (output channel dimension)
        :param n: number of convolutional blocks
        :param kwargs: keyword arguments for conv_block
        """
        x = self.conv_block(inputs, num_filters, **kwargs)
        for _ in np.arange(n - 1):
            x = self.conv_block(x, num_filters, **kwargs)

        return x

    
    def encoder_block(self, inputs, num_filters, l_large: bool = True, kernel_pool: tuple = (2, 2), l_avgpool: bool = False, **kwargs):
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
            x = self.conv_block_n(inputs, num_filters, n=2, **kwargs)
        else:
            x = self.conv_block(inputs, num_filters, **kwargs)

        if l_avgpool:
            p = AveragePooling2D(kernel_pool)(x)
        else:
            p = MaxPool2D(kernel_pool)(x)

        return x, p

    @staticmethod
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



    def decoder_block(self, inputs, skip_features, num_filters, strides_up: int = 2, l_subpixel: bool = False, **kwargs_conv_block):
        """
        One complete decoder block used in U-net (reverting the encoder)
        """
        if l_subpixel:
            kwargs_subpixel = kwargs_conv_block.copy()
            for ex_key in ["strides", "l_batch_normalization"]:
                kwargs_subpixel.pop(ex_key, None)
            x = self.subpixel_block(inputs, num_filters, upscale_fac=strides_up, **kwargs_subpixel)
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
        x = self.conv_block_n(x, num_filters, 2, **kwargs_conv_block)

        return x
    
    def encoder_block_deepru(self, inputs, channels, strides, kernel: tuple = (3, 3), nconv_res: int = 3, padding: str = "same",
                             activation="LeakyReLU", kernel_init="he_normal", l_batch_normalization: bool = True):
        # save the input for the residual connection
        skip_features = inputs

        x = self.conv_block(inputs, channels, kernel, strides, padding=padding, activation=activation, kernel_init=kernel_init,
                            l_batch_normalization=l_batch_normalization)

        x = self.residual_block(x, channels, kernel, nconv_res, padding, activation, kernel_init, l_batch_normalization)

        return skip_features, x


    def decoder_block_deepru(self, inputs, skip_features, channels, size, interpolation: str = "bilinear", nconv_res: int = 3,
                             kernel: tuple = (3, 3), padding: str = "same", activation="LeakyReLU", kernel_init="he_normal",
                             l_batch_normalization: bool = True):
        x = UpSampling2D(size, interpolation=interpolation)(inputs)

        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, channels, kernel, padding=padding, activation=activation, kernel_init=kernel_init,
                            l_batch_normalization=l_batch_normalization)
        x = self.residual_block(x, channels, kernel, nconv_res, padding, activation, kernel_init, l_batch_normalization)

        return x

    
    def residual_block(self, inputs, channels, kernel: tuple = (3,3), nconv_res: int = 3, padding: str = "same",
                       activation="LeakyReLU", kernel_init="he_normal", l_batch_normalization: bool = True):

        # process the input with convolutional layers (incl. non-linear activation and optional batch normalization)
        x = self.conv_block_n(inputs, channels, nconv_res, kernel=kernel, padding=padding, activation=activation, kernel_init=kernel_init,
                              l_batch_normalization=l_batch_normalization)

        # the actual residual connection: adding inpput and processed data
        x = Add()([x, inputs])
        # finally apply non-linear activation and optional batch normalization
        # (following He et al., 2016: https://doi.org/10.1109/CVPR.2016.90, but not shown in Fig.6 of Hoehlein et al., 2020)
        try:
            x = Activation(activation)(x)
        except ValueError:
            ac_layer = advanced_activation(activation)
            x = ac_layer(x)

        # batch normalization is considered to be beneficial, see
        # https://stackoverflow.com/questions/64792460/how-to-code-a-residual-block-using-two-layers-of-a-basic-cnn-algorithm-built-wit
        if l_batch_normalization:
            x = BatchNormalization()(x)

        return x


class UNet_Sha(AbstractModelClass):
    
    def __init__(self, shape_in: List, hparams: dict, varnames_tar: List, savedir: str, expname: str, concat_out: bool = False):
        
        super().__init__(shape_in, hparams, varnames_tar, savedir, expname)
        
        self.concat_out = concat_out
        self.modelname = "unet_sha"
        
        # get building blocks for U-Net
        building_blocks = UNetModelBase()
        self.conv_block = building_blocks.conv_block
        self.encoder_block, self.decoder_block = building_blocks.encoder_block, building_blocks.decoder_block
        
        # set hyperparmaters
        self.set_hparams(hparams)
        if self.hparams.get("z_branch"):
            self._n_predictands_dyn = self._n_predictands - 1
        else:
            self._n_predictands_dyn = self._n_predictands

        self.set_model()
        # set compile and fit options as well as custom objects
        self.set_compile_options()
        self.set_custom_objects(loss=self.compile_options['loss'])
        self.set_fit_options()
        
    def set_model(self):
        
        hparams_unet = self.hparams
        # some auxiliary variables to ease handling of configuration parameters
        channels_start = hparams_unet["channels_start"]
        kernel_pool = hparams_unet["kernel_pool"]
        l_avgpool = hparams_unet["l_avgpool"]
        l_subpixel = hparams_unet["l_subpixel"]

        config_conv = {"kernel": hparams_unet["kernel"], "strides": hparams_unet["strides"], "padding": hparams_unet["padding"], 
                       "activation": hparams_unet["activation"], "activation_args": hparams_unet["activation_args"], 
                       "kernel_init": hparams_unet["kernel_init"], "l_batch_normalization": hparams_unet["l_batch_normalization"]}

        # build U-Net
        inputs = Input(self._input_shape)

        """ encoder """
        s1, e1 = self.encoder_block(inputs, channels_start, l_large=True, kernel_pool=kernel_pool, l_avgpool=l_avgpool,**config_conv)
        s2, e2 = self.encoder_block(e1, channels_start * 2, l_large=False, kernel_pool=kernel_pool, l_avgpool=l_avgpool,**config_conv)
        s3, e3 = self.encoder_block(e2, channels_start * 4, l_large=False, kernel_pool=kernel_pool, l_avgpool=l_avgpool,**config_conv)

        """ bridge encoder <-> decoder """
        b1 = self.conv_block(e3, channels_start * 8, **config_conv)

        """ decoder """
        d1 = self.decoder_block(b1, s3, channels_start * 4, l_subpixel=l_subpixel, **config_conv)
        d2 = self.decoder_block(d1, s2, channels_start * 2, l_subpixel=l_subpixel, **config_conv)
        d3 = self.decoder_block(d2, s1, channels_start, l_subpixel=l_subpixel, **config_conv)

        output_dyn = Conv2D(self._n_predictands_dyn, (1, 1), kernel_initializer=config_conv["kernel_init"], name=self._varnames_tar[0])(d3)
        if self.hparams.get("z_branch", False):
            print("Use z_branch...")
            output_static = Conv2D(1, (1, 1), kernel_initializer=config_conv["kernel_init"], name=self._varnames_tar[1])(d3)

            if self.concat_out:
                self.model = Model(inputs, tf.concat([output_dyn, output_static], axis=-1), name="downscaling_unet_with_z")
            else:
                self.model = Model(inputs, [output_dyn, output_static], name="downscaling_unet_with_z")
        else:
            self.model = Model(inputs, output_dyn, name="downscaling_unet")
            
    def set_compile_options(self):
        
        custom_opt = self.hparams["optimizer"].lower()

        if custom_opt == "adam":
            opt = keras.optimizers.Adam
        elif custom_opt == "rmsprop":
            opt = keras.optimizers.RMSprop

        compile_opts = {"optimizer": opt(learning_rate=self.hparams["lr"])}

        if self.hparams.get("z_branch", False):
            compile_opts["loss"] = {f"{var}": get_custom_loss(self.hparams["loss_func"]) for var in self._varnames_tar}
            compile_opts["loss_weights"] = {f"{var}": self.hparams["loss_weights"][i] for i, var in
                                            enumerate(self._varnames_tar)}
        else:
            compile_opts["loss"] = get_custom_loss(self.hparams["loss_func"])
            
        self.compile_options = compile_opts
        
    def get_fit_options(self):
        
        # get name of loss component to track
        loss_monitor = f"val_{self.compile_options['loss'].keys()[0]}_loss" if self.hparams.get("z_branch", False) else "val_loss"
        
        unet_callbacks = []
        
        if self.hparams["lr_decay"]:
            unet_callbacks.append(LearningRateScheduler(self.get_lr_scheduler(), verbose=1))
        
        if self.hparams["lcheckpointing"]:
            savedir_best = os.path.join(self._savedir, f"{self._expname}_best")
            os.makedirs(savedir_best, exist_ok=True)

            unet_callbacks.append(ModelCheckpoint(savedir_best, monitor=loss_monitor, verbose=1, save_best_only=True, mode="min"))
            
        if self.hparams["learlystopping"]:
            unet_callbacks.append(EarlyStopping(monitor=loss_monitor, patience=8))
            
        if unet_callbacks is not None:
            return {"callbacks": unet_callbacks}
        else:
            return {}  
        
    def set_hparams_default(self):
        """
        Note: Hyperparameter defaults of generator and critic model must be set in the respective model classes whose instances are just parsed here.
              Thus, the default just sets empty dictionaries.
        """
        self.hparams_default = {"kernel": (3, 3), "strides": (1, 1), "padding": "same", "activation": "swish", "activation_args": {},      # arguments for building blocks of U-Net:
                                "kernel_init": "he_normal", "l_batch_normalization": True, "kernel_pool": (2, 2), "l_avgpool": True,       # see keyword-aguments of conv_block,
                                "l_subpixel": True, "z_branch": True, "channels_start": 56,                                                # encoder_block and decoder_block
                                "batch_size": 32, "lr": 5.e-05, "nepochs": 35, "loss_func": "mae", "loss_weights": [1.0, 1.0], "named_targets": True,             # training parameters
                                "lr_decay": False, "decay_start": 3, "decay_end": 20, "lr_end": 1.e-06, "l_embed": False, 
                                "optimizer": "adam", "lcheckpointing": True, "learlystopping": False, }
        
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

    

class UNet_DeepRU(UNet_Sha):
    
    def __init__(self, shape_in: List, hparams: dict, varnames_tar: List, savedir: str, expname: str):
        
        # get building blocks for U-Net
        building_blocks = UNetModelBase()
        self.encoder_block_dru, self.decoder_block_dru = building_blocks.encoder_block_deepru, building_blocks.decoder_block_deepru
        
        super().__init__(shape_in, hparams, varnames_tar, savedir, expname)

        self.modelname = "deepru"
        
        # set hyperparmaters
        self.set_hparams(hparams)
        
        # set model
        self.set_model()

        # set compile and fit options as well as custom objects
        self.set_compile_options()
        self.set_custom_objects(loss=self.compile_options['loss'])
        self.set_fit_options()
        
    def set_model(self):
        
        # get basic parameters
        hparams_deepru = self.hparams
        
        channels = np.arange(hparams_deepru["channels_start"], hparams_deepru["channels_start"] 
                             + 6*hparams_deepru["dchannels"] + 1, hparams_deepru["dchannels"])
        
        strides_list = hparams_deepru["strides_list"]
        
        assert len(channels) == len(hparams_deepru["strides_list"]) + 1, \
               f"Length of channels-list ({len(channels)}) must contain one element more than strides_list ({len(strides_list)})."
        
        encoder_decoder_args = {key: hparams_deepru[key] for key in ["kernel", "padding", "nconv_res", "activation",
                                                                     "kernel_init", "l_batch_normalization"]}
        
        # build the DeepRU-network
        inputs = Input(self._input_shape)
        
        latent_in = self.conv_block(inputs, channels[0], (5, 5), activation=hparams_deepru["activation"])
        
        """encoder"""
        s1, e1 = self.encoder_block_dru(latent_in, channels[1], strides=strides_list[0], **encoder_decoder_args)
        s2, e2 = self.encoder_block_dru(e1, channels[2], strides=strides_list[1], **encoder_decoder_args)
        s3, e3 = self.encoder_block_dru(e2, channels[3], strides=strides_list[2], **encoder_decoder_args)
        s4, e4 = self.encoder_block_dru(e3, channels[4], strides=strides_list[3], **encoder_decoder_args)
        s5, e5 = self.encoder_block_dru(e4, channels[5], strides=strides_list[4], **encoder_decoder_args)

        """ bridge encoder <-> decoder """
        s6, b = self.encoder_block_dru(e5, channels[6], strides=strides_list[5], **encoder_decoder_args)

        """decoder"""
        d6 = self.decoder_block_dru(b, s6, channels[5], size=strides_list[5], interpolation = hparams_deepru["interpolation"], **encoder_decoder_args)
        d5 = self.decoder_block_dru(d6, s5, channels[4], size=strides_list[4], interpolation = hparams_deepru["interpolation"], **encoder_decoder_args)
        d4 = self.decoder_block_dru(d5, s4, channels[3], size=strides_list[3], interpolation = hparams_deepru["interpolation"], **encoder_decoder_args)
        d3 = self.decoder_block_dru(d4, s3, channels[2], size=strides_list[2], interpolation = hparams_deepru["interpolation"], **encoder_decoder_args)
        d2 = self.decoder_block_dru(d3, s2, channels[1], size=strides_list[1], interpolation = hparams_deepru["interpolation"], **encoder_decoder_args)
        d1 = self.decoder_block_dru(d2, s1, channels[0], size=strides_list[0], interpolation = hparams_deepru["interpolation"], **encoder_decoder_args)

        output_dyn = Conv2D(self._n_predictands, (3, 3), kernel_initializer = hparams_deepru["kernel_init"],
                            padding = hparams_deepru["padding"], name="output_dyn")(d1)

        self.model = Model(inputs, output_dyn, name="downscaling_deepru")
        
    def set_hparams_default(self):
        
        self.hparams_default = {"kernel": (3, 3), "nconv_res": 3, "padding": "same", "activation": "LeakyReLU", "kernel_init": "he_normal",
                                "l_batch_normalization": True, "interpolation": "bilinear", "channels_start": 64, "dchannels": 64,
                                "strides_list": [(2, 1), (1, 3), (2, 1), (2, 2), (2, 2), (2, 2)],               # for domain size of 96x120 grid points
                                "batch_size": 32, "lr": 1.e-03, "nepochs": 35, "loss_func": "mse",              # training parameters
                                "lr_decay": False, "decay_start": 3, "decay_end": 20, "lr_end": 1.e-06, "l_embed": False, 
                                "optimizer": "adam", "lcheckpointing": True, "learlystopping": False}
