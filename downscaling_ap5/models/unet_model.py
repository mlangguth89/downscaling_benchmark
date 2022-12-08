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
from tensorflow.keras.layers import (Concatenate, Conv2D,Reshape, Conv2DTranspose, Input,
                                     MaxPool2D, RepeatVector, Embedding, Dense)
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
               embed_sh: tuple = (12, 24), embed_latent_dim: int = 8, tar_channels=["output_temp", "output_z"]) -> Model:
    """
    Builds up U-net model
    :param input_shape: shape of input-data
    :param channels_start: number of channels to use as start in encoder blocks
    :param z_branch: flag if z-branch is used.
    :param l_embed: flag if embedding with one-hot-encoding vectors is performed. The default embeds the date in terms
                    of the month of the year as well as the hour of the day (see embed_sh)
    :param embed_sh: maximum indices for date input. Default (12, 24) corresonds to the twelve months of
                     a year and the 24 hours of a day
    :param embed_latent_dim: output dimension of embedding (latent representation vector)
    :param tar_channels: name of output channels, only active if z_branch is True
    :return:
    """
    # 'normal' input stream
    inputs = Input(input_shape)

    """ encoder """
    s1, e1 = encoder_block(inputs, channels_start, l_large=True)
    s2, e2 = encoder_block(e1, channels_start * 2, l_large=False)
    s3, e3 = encoder_block(e2, channels_start * 4, l_large=False)

    """ bridge encoder <-> decoder """
    b1 = conv_block(e3, channels_start * 8)

    if l_embed:  # optional embedding at the encode-decoder bridge
        bridge_shape = b1.shape[1:]
        n_nodes = int(np.prod(bridge_shape[:-1]))
        tar_dim = tuple(list(bridge_shape[:-1]) + [embed_latent_dim])

        embed_inputs = [Input(shape=(1,)) for _ in embed_sh]
        embeds = [Embedding(n, embed_latent_dim, input_length=1)(embed_inputs[i]) for i, n in enumerate(embed_sh)]

        embeds = Concatenate()([embed[:,0,:] for embed in embeds])
        embeds = [Dense(embed_latent_dim*2)(embeds)

        embeds_vec = RepeatVector(n_nodes)(embeds)
        embeds_vec = Reshape(tar_dim)(embeds_vec)

        merge_list = [b1] + [embeds_vec]

        b1 = Concatenate()(merge_list)
        # append input
        all_inputs = [inputs] + embed_inputs

    else:  
        # no merging of embeddings required, so leave inputs unchanged
        all_inputs = inputs

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
