import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

# all the layers used for U-net
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPool2D, Dense, Flatten
)
from tensorflow.keras.models import Model
from unet_model import build_unet, conv_block

from typing import List, Tuple, Union

list_or_tuple = Union[List, Tuple]


def critic(shape, num_conv: int = 4, channels_start: int = 64, kernel: tuple = (3,3),
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
        x = conv_block(x, channels_start, kernel, stride, activation=activation, l_batch_normalization=lbatch_norm)
        channels *= 2

    # finally flatten encoded data, add two fully-connected layers...
    x = Flatten()(x)
    x = Dense(channels/2, activation=activation)(x)
    x = Dense(channels/4, activation=activation)(x)
    # ... and end with linear output layer
    out = Dense(1, activation="linear")(x)

    critic_model = Model(input=critic_in, outputs=out)

    return critic_model, out


class WGAN(object):
    """
    Class for Wassterstein GAN models
    """
    known_modes = ["train", "predict"]

    def __init__(self, generator: keras.Model, critic: keras.Model, hparams: dict, input_shape: list_or_tuple,
                 target_shape: list_or_tuple, mode: str = "train", embedding: List = None):

        # sanity checks on parsed models
        if not isinstance(generator, keras.Model):
            raise ValueError("Generator must be a Keras model instance, but is of type '{0}'".format(type(generator)))

        if not isinstance(critic, keras.Model):
            raise ValueError("Critic must be a Keras model instance, but is of type '{0}'".format(type(critic)))

        if mode not in WGAN.known_modes:
            raise ValueError("Parsed mode '{0}' is unknown. Possible choices: {1}"
                             .format(str(mode), ", ".format(WGAN.known_modes)))

        self.generator, self.critic = generator, critic
        self.mode = mode
        self.hparams = WGAN.get_hparams_dict(hparams)


    @staticmethod
    def get_hparams_default():
        hparams_dict = {
            "batch_size": 4,
            "lr": 1.e-03,
            "train_epochs": 10,
            "lr_decay": False,
            "decay_start": 5,
            "lr_end": 1.e-04,
            "ngf": 56,
            "d_steps": 5,
            "recon_weight": 20.,
            "gp_weight": 10.,
            "optimizer": keras.optimizers.Adam(learning_rate = 1.e-03, beta_1 = 0.5, beta_2 = 0.9)
        }

        return hparams_dict

    @staticmethod
    def get_hparams_dict(hparams_user):

        hparams_default = WGAN.get_hparams_default()

        # check if parsed hyperparameters are known
        unknown_keys = [key for key in hparams_user.keys() if key not in hparams_default]
        if unknown_keys:
            print("The following parsed hyperparameters are unknown and thus are ignored: {0}"
                  .format(", ".join(unknown_keys)))

        # get complete hyperparameter dictionary while checking type of parsed values
        hparams_merged = {**hparams_user, **hparams_default}
        hparams_dict = {}
        for key in hparams_default:
            if isinstance(hparams_merged[key], type(hparams_default[key])):
                hparams_dict[key] = hparams_merged[key]
            else:
                raise TypeError("Parsed hyperparameter '{0}' must be of type '{1}', but is '{2}'"
                                .format(key, type(hparams_default[key]), type(hparams_merged[key])))

        return hparams_dict






