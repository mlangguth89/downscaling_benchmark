import numpy as np
import tensorflow as tf

# all the layers used for U-net
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPool2D, Dense, Flatten
)
from tensorflow.keras.models import Model
from unet_model import build_unet, conv_block


def discriminator(shape, conv_depth: int = 4, channels_start: int = 64, kernel: tuple = (3,3),
                  stride: tuple = (2, 2), activation: str = "relu", lbatch_norm: bool = True):
    """
    Set-up convolutional discriminator model that is followed by two dense-layers
    :param tar_shape: input shape of data (either real or generated data)
    :param conv_depth: number of convolutional layers
    :param channels_start: number of channels in first layer
    :param kernel: convolutional kernel, e.g. (3,3)
    :param stride: stride for convolutional layer
    :param activation: activation function to use (applies to convolutional and fully-connected layers)
    :param lbatch_norm: flag to perform batch normalization on convolutional layers
    :return:
    """
    discrim_input = Input(shape=shape)
    x = discrim_input

    channels = channels_start

    for _ in range(conv_depth):
        x = conv_block(x, channels_start, kernel, stride, activation=activation, l_batch_normalization=lbatch_norm)
        channels *= 2

    # finally flatten encoded data, add two fully-connected layers...
    x = Flatten()(x)
    x = Dense(channels/2, activation=activation)(x)
    x = Dense(channels/4, activation=activation)(x)
    # ... and end with linear output layer
    out = Dense(1, activation="linear")(x)

    discrim_model = Model(input=discrim_input, outputs=out)

    return discrim_model, out

