__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2021-XX-XX"
__update__ = "2022-05-26"

"""
Methods to set-up U-net models incl. its building blocks.
"""

# import modules
import numpy as np
import tensorflow as tf

# all the layers used for U-net
from tensorflow.keras.layers import (Concatenate, Dense, Conv2D,Reshape, Conv2DTranspose, Input,
                                     MaxPool2D, RepeatVector)
from tensorflow.keras.models import Model
from model_utils import conv_block, conv_block_n

# building blocks for Unet


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


def decoder_block(inputs, skip_features, num_filters, kernel: tuple=(3,3), strides_up: int = 2, padding: str = "same",
                  activation="relu", kernel_init="he_normal", l_batch_normalization: bool = True):
    """
    One complete decoder block used in U-net (reverting the encoder)
    """
    x = Conv2DTranspose(num_filters, (strides_up, strides_up), strides=strides_up, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block_n(x, num_filters, 2, kernel, (1, 1), padding, activation, kernel_init=kernel_init, 
                     l_batch_normalization=l_batch_normalization)

    return x


# The particular U-net
def build_unet(input_shape: tuple, channels_start: int = 56, z_branch: bool = False, l_embed: bool = False,
               embed_sh: tuple = (12, 24), tar_channels=["output_temp", "output_z"]) -> Model:
    """
    Builds up U-net model
    :param input_shape: shape of input-data
    :param channels_start: number of channels to use as start in encoder blocks
    :param z_branch: flag if z-branch is used.
    :param l_embed: flag if embedding with one-hot-encoding vectors is performed. The default embeds the date in terms
                    of the month of the year as well as the hour of the day (see embed_sh)
    :param embed_sh: shape of one-hot-encodings for date embedding. Default (12, 24) corresonds to the twelve months of
                     a year and the 24 hours of a day
    :param tar_channels: name of output channels, only active if z_branch is True
    :return:
    """
    # 'normal' input stream
    inputs = Input(input_shape)

    if l_embed:     # optional embedding with one-hot-encoding vectors
        n_nodes = np.prod(input_shape[:-1])
        tar_dim = tuple(list(input_shape[:-1]) + [1])

        embed_vec_in = [Input(shape=i) for i in embed_sh]

        # project embedding to scalar value
        embed_vec_proj = [Dense(1)(i) for i in embed_vec_in]
        # repeat and reshape to match input-data for concatenation
        embed_vec_proj = [RepeatVector(int(n_nodes))(i) for i in embed_vec_proj]
        embed_vec_proj = [Reshape(tar_dim)(i) for i in embed_vec_proj]

        merge_list = [inputs] + [embed for embed in embed_vec_proj]

        merge = Concatenate()(merge_list)

        all_inputs = [inputs] + [input_embed for input_embed in embed_vec_in]
    else:   # no merging with embedding required
        merge = inputs
        all_inputs = inputs

    """ encoder """
    s1, e1 = encoder_block(merge, channels_start, l_large=True)
    s2, e2 = encoder_block(e1, channels_start * 2, l_large=False)
    s3, e3 = encoder_block(e2, channels_start * 4, l_large=False)

    """ bridge encoder <-> decoder """
    b1 = conv_block(e3, channels_start * 8)

    """ decoder """
    d1 = decoder_block(b1, s3, channels_start * 4)
    d2 = decoder_block(d1, s2, channels_start * 2)
    d3 = decoder_block(d2, s1, channels_start)

    output_temp = Conv2D(1, (1, 1), kernel_initializer="he_normal", name=tar_channels[0])(d3)
    if z_branch:
        output_z = Conv2D(1, (1, 1), kernel_initializer="he_normal", name=tar_channels[1])(d3)

        model = Model(all_inputs, [output_temp, output_z], name="t2m_downscaling_unet_with_z")
    else:
        model = Model(all_inputs, output_temp, name="t2m_downscaling_unet")

    return model


def get_lr_scheduler():
    # define a learning-rate scheduler
    def lr_scheduler(epoch, lr):
        if epoch < 5:
            return lr
        elif 5 <= epoch < 30:
            return lr * tf.math.exp(-0.1)
        elif epoch >= 30:
            return lr

    lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    return lr_scheduler_cb
