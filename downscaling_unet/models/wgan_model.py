from collections import OrderedDict
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


def critic_model(shape, num_conv: int = 4, channels_start: int = 64, kernel: tuple = (3,3),
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

    critic = Model(input=critic_in, outputs=out)

    return critic, out


class WGAN(keras.Model):
    """
    Class for Wassterstein GAN models
    """
    known_modes = ["train", "predict"]

    def __init__(self, generator: keras.Model, critic: keras.Model, hparams: dict, input_shape: list_or_tuple,
                 target_shape: list_or_tuple, mode: str = "train", embedding_shape: List = None):

        super(WGAN, self).__init__()
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
        if self.hparams["l_embed"] and not embedding:
            raise ValueError("Embedding must be parsed if hyperparameter l_embed is set to True.")
        # set in compile-method
        self.d_optimizer = None
        self.g_optimizer = None

    def compile(self):
        super(WGAN, self).compile()

        self.d_optimizer, self.g_optimizer = self.hparams["d_optimizer"], self.hparams["g_optimizer"]



    def gradient_penalty(self, real_data, gen_data):
        """
        Calculates gradient penalty based on 'mixture' of generated and ground truth data
        :param real_data: the ground truth data
        :param gen_data: the generated/predicted data
        :return: gradient penalty
        """
        # get mixture of generated and ground truth data
        alpha = tf.random.normal([self.hparams["batch_size"], 1, 1, 1], 0., 1.)
        mix_data = real_data + alpha*(gen_data - real_data)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(mix_data)
            critic_mix = self.critic(mix_data, training=True)

        # calculate the gradient on the mixture data...
        grads_mix = gp_tape.gradient(critic_mix, [mix_data])[0]
        # ... and norm it
        norm = tf.sqrt(tf.reduce_mean(tf.square(grads_mix), axis=[1, 2, 3]))
        gp = tf.reduce_mean(tf.square(norm - 1.))

        return gp

    def make_data_generator(self, ds, ds_val=None, embed=None, embed_val=None):

        if self.hparams["l_embed"]:
            if not embed:
                raise ValueError("Embedding is enabled, but no embedding data was parsed.")
            data_iter = tf.Data.Dataset.from_tensor_slices((ds, embed))
        else:
            data_iter = tf.Data.Dataset.from_tensor_slices(ds)

        data_iter = data_iter.shuffle(10000).repeat(self.hparams["train_epochs"]).batch(self.hparams["batch_size"])
        data_iter = iter(data_iter)

        if ds_val:
            if self.hparams["l_embed"]:
                if not embed_val:
                    raise ValueError("Embedding is enabled, but no embedding data for validation dataset is parsed.")
                val_iter = tf.Data.Dataset.from_tensor_slices((ds_val, embed_val))
            else:
                val_iter = tf.Data.Dataset.from_tensor_slices(ds_val)

            val_iter.repeat().batch(self.hparams["batch_size"])

            return data_iter, val_iter
        else:
            return data_iter

    def train_step(self, predictors, predictands, step, embed = None):

        # train the critic d_steps-times
        for substep in range(self.hparams["d_steps"]):
            with tf.GradientTape() as tape_critic:
                # generate (downscaled) data
                gen_data = self.generator(predictors, training=True)
                # calculate critics for both, the real and the generated data
                critic_gen, critic_gt = self.critic(gen_data, training=True), self.critic(predictands, training=True)
                # calculate the loss (incl. gradient penalty)
                c_loss = WGAN.critic_loss(critic_gt, critic_gen)
                gp = self.gradient_penalty(predictands, gen_data)
                d_loss = c_loss + self.hparams["gp_weight"]*gp

            # calculate gradients and update discrimintor
            d_gradient = tape_critic.gradient(d_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(d_gradient, self.critic.trainable_variables))

        # train generator
        with tf.GradientTape() as tape_generator:
            # generate (downscaled) data
            gen_data = self.generator(predictors, training=True)
            # get the critic and calculate corresponding generator losses (critic and reconstruction loss)
            critic_gen = self.critic(gen_data, training=True)
            cg_loss = WGAN.critic_gen_loss(critic_gen)
            recon_loss = WGAN.recon_loss(predictands, gen_data)

            g_loss = cg_loss + self.hparams["recon_weight"]*recon_loss

        g_gradient = tape_generator.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        return OrderedDict([("c_loss", c_loss), ("gp_loss", gp), ("d_loss", d_loss), ("cg_loss", cg_loss),
                            ("recon_loss", recon_loss), ("g_loss", g_loss)])






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

        adam = keras.optimizers.Adam
        hparams_dict["d_optimizer"] = adam(learning_rate = hparams_dict["lr"], beta_1 = 0.5, beta_2 = 0.9)
        hparams_dict["g_optimizer"] = adam(learning_rate = hparams_dict["lr"], beta_1 = 0.5, beta_2 = 0.9)

        return hparams_dict

    @staticmethod
    def get_hparams_default():
        hparams_dict = {
            "batch_size": 4,
            "lr": 1.e-03,
            "train_epochs": 10,
            "lr_decay": False,
            "decay_start": 5,
            "lr_end": 1.e-04,
            "l_embed": False,
            "ngf": 56,
            "d_steps": 5,
            "recon_weight": 20.,
            "gp_weight": 10.,
        }

        return hparams_dict

    @staticmethod
    def critic_loss(critic_real, critic_gen):
        c_loss = tf.reduce_mean(critic_gen - critic_real)

        return c_loss

    @staticmethod
    def critic_gen_loss(critic_gen):
        cg_loss = -tf.reduce_mean(critic_gen)

        return cg_loss

    @staticmethod
    def recon_loss(real_data, gen_data):
        recon_loss = tf.reduce_mean(tf.abs(gen_data - real_data))

        return recon_loss







