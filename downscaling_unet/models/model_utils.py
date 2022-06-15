__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-05-26"
__update__ = "2022-05-31"

"""
Some auxiliary methods to create Keras models.
"""
# import modules
import numpy as np
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPool2D)


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
        ac_layer = advanced_activation(activation, *activation_args)
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


def advanced_activation(activation_name, *args, **kwargs):
    """
    Get layer to enable one of Keras' advanced activation, see https://keras.io/api/layers/activation_layers/
    :param activation_name: name of the activation function to apply
    :return: the respective layer to deploy desired activation
    """
    known_activations = ["LeakyReLU", "Softmax", "PReLU", "ELU", "ThresholdedReLU"]

    activation_name = activation_name.lower()

    if activation_name == "leakyrelu":
        layer = layers.LeakyReLU(*args, **kwargs)
    elif activation_name == "softmax":
        layer = layers.Softmax(*args, **kwargs)
    elif activation_name == "elu":
        layer = layers.ELU(*args, **kwargs)
    elif activation_name == "prelu":
        layer = layers.PReLU(*args, **kwargs)
    elif activation_name == "thresholdedrelu":
        layer = layers.ThresholdedReLU(*args, **kwargs)
    else:
        raise ValueError("{0} is not a valid advanced activation function. Choose one of the following: {1}"
                         .format(activation_name, ", ".join(known_activations)))

    return layer
