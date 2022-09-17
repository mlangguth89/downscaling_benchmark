__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-05-19"
__update__ = "2022-09-08"

import os, sys
from collections import OrderedDict
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging

# all the layers used for U-net
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
# other modules
import numpy as np

from unet_model import build_unet, conv_block
from handle_data_class import HandleDataClass

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

    def __init__(self, generator: keras.Model, critic: keras.Model, hparams: dict, model_name: str = "wgan_model",
                 lsupervise: bool = True, savedir: str = None):
        """
        Initiate Wasserstein GAN model.
        :param generator: A generator model returning a data field
        :param critic: A critic model which returns a critic scalar on the data field
        :param hparams: dictionary of hyperparameters
        :param model_name: name of the WGAN-model
        :param lsupervise: flag to supervise training with early stopping (best model saved and early stopping with
                           patience of eight epochs); if True, savedir must be provided
        :param savedir: directory where checkpointed model will be saved
        :param l_embedding: Flag to enable time embeddings
        """

        super(WGAN, self).__init__()

        self.generator, self.critic = generator, critic
        self.hparams = WGAN.get_hparams_dict(hparams)
        self.modelname = model_name
        self.lsupervise = lsupervise
        if lsupervise and savedir is None:
            raise ValueError("Add savedir to enable model saving when using checkpointing" +
                             "and early stopping (lsupervise=True)")
        elif lsupervise:
            os.makedirs(savedir, exist_ok=True)
        self.savedir = savedir
        # set in compile-method
        self.train_iter, self.val_iter = None, None
        self.shape_in, self.nsamples = None, None
        self.c_optimizer, self.g_optimizer = None, None
        self.lr_scheduler = None
        self.checkpoint, self.earlystopping = None, None

    def compile(self, da_train, da_val):
        """
        Turn datasets into tf.Datasets, compile model and set learning rate schedule (optionally)
        :param da_train: Data Array providing the training data (must have variables-dimension)
        :param da_val: Data Array providing the validation data (must have variables-dimension)
        :return: tf.Datasets for training and validation data
        """
        invars = [var for var in da_train["variables"].values if var.endswith("_in")]
        invars = invars + ["hsurf_tar"]

        # determine shape-dimensions from data
        shape_all = da_train.sel({"variables": invars}).shape
        self.nsamples, self.shape_in = shape_all[0], shape_all[1:]
        self.embed_shape = [24]


        tar_shape = (*self.shape_in[:-1], 1)   # critic only accounts for 1st channel (should be the downscaling target)
        # instantiate models
        self.generator = self.generator(self.shape_in, channels_start=self.hparams["ngf"],
                                        z_branch=self.hparams["z_branch"], l_embed=self.hparams["l_embed"])
        self.critic = self.critic(tar_shape)

        # Note: The batch-size is increased for the training datset to allow substepping in WGAN
        # (n+1 optimization steps to train generator and critic). The validation dataset however does not perform
        # sub-stepping and thus does not be instantiated with an increased mini-batch size.
        train_iter = HandleDataClass.make_tf_dataset(da_train, self.hparams["batch_size"]
                                                     * (self.hparams["d_steps"] + 1),
                                                     lembed_date=self.hparams["l_embed"],
                                                     var_tar2in=self.hparams["var_tar2in"])
        val_iter = HandleDataClass.make_tf_dataset(da_val, self.hparams["batch_size"], lshuffle=False,
                                                   lembed_date=self.hparams["l_embed"],
                                                   var_tar2in=self.hparams["var_tar2in"])

        # call Keras compile method
        super(WGAN, self).compile()
        # set optimizers
        self.c_optimizer, self.g_optimizer = self.hparams["d_optimizer"], self.hparams["g_optimizer"]

        # get learning rate schedule if desired
        if self.hparams["lr_decay"]:
            self.lr_scheduler = LearningRateSchedulerWGAN(self.get_lr_decay(), verbose=1)

        if self.lsupervise:
            self.checkpoint = ModelCheckpointWGAN(self.savedir, monitor="val_recon_loss", verbose=1,
                                                  save_best_only=True, mode="min")
            # self.earlystopping = EarlyStopping(monitor="val_recon_loss", patience=8)

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

        wgan_callbacks = [self.lr_scheduler, self.checkpoint, self.earlystopping]

        callbacks = [e for e in wgan_callbacks + callbacks if e is not None]
        steps_per_epoch = int(np.ceil(self.nsamples / self.hparams["batch_size"]))

        return super(WGAN, self).fit(x=train_iter, callbacks=callbacks, epochs=self.hparams["train_epochs"],
                                     steps_per_epoch=steps_per_epoch, validation_data=val_iter, validation_steps=300,
                                     verbose=2)

    def train_step(self, data_iter: tf.data.Dataset, embed=None) -> OrderedDict:
        """
        Training step for Wasserstein GAN.
        :param data_iter: Tensorflow Dataset providing training data
        :param embed: embedding (not implemented yet)
        :return: Ordered dictionary with several losses of generator and critic
        """

        if self.hparams["l_embed"]:
            predictors, predictands, embeds_m, embeds_h = data_iter
        else:
            predictors, predictands = data_iter
        # train the critic d_steps-times
        for i in range(self.hparams["d_steps"]):
            with tf.GradientTape() as tape_critic:
                ist, ie = i * self.hparams["batch_size"], (i + 1) * self.hparams["batch_size"]
                if self.hparams["l_embed"]:
                    predictor_list = [predictors[ist:ie, :, :, :], embeds_m[ist:ie,...], embeds_h[ist:ie,...]]
                else:
                    predictor_list = predictors[ist:ie, :, :, :]
                # critic only operates on first channel
                predictands_critic = tf.expand_dims(predictands[ist:ie, :, :, 0], axis=-1)
                # generate (downscaled) data
                gen_data = self.generator(predictor_list, training=True)
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
            if self.hparams["l_embed"]:
                predictor_list = [predictors[-self.hparams["batch_size"]:, ...],
                                  embeds_m[-self.hparams["batch_size"]:,...],embeds_h[-self.hparams["batch_size"]:,...]]
            else:
                predictor_list = predictors[-self.hparams["batch_size"]:, ...]
            gen_data = self.generator(predictor_list, training=True)
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
        if self.hparams["l_embed"]:
            predictors, predictands, embeds_m, embeds_h = val_iter
            predictor_list = [predictors, embeds_m, embeds_h]
        else:
            predictors, predictands = val_iter
            predictor_list = predictors

        gen_data = self.generator(predictor_list, training=False)
        rloss = self.recon_loss(predictands, gen_data)

        return OrderedDict([("recon_loss", rloss)])

    def predict_step(self, test_iter: tf.data.Dataset) -> OrderedDict:

        if self.hparams["l_embed"]:
            predictors, _, embeds_m, embeds_h = test_iter
            predictor_list = [predictors, embeds_m, embeds_h]
        else:
            predictors, _ = test_iter
            predictor_list = predictors

        return self.generator(predictor_list, training=False)

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
                        "ngf": 56, "d_steps": 5, "recon_weight": 1000., "gp_weight": 10., "optimizer": "adam", 
                        "var_tar2in": ""}

        return hparams_dict

    @staticmethod
    def critic_loss(critic_real, critic_gen):
        """
        The critic is optimized to maximize the difference between the generated and the real data max(real - gen).
        This is equivalent to minimizing the negative of this difference, i.e. min(gen - real) = max(real - gen)
        :param critic_real: critic on the real data
        :param critic_gen: critic on the generated data
        :param c_loss: loss to optize the critic
        """
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

class ModelCheckpointWGAN(ModelCheckpoint):

    def __init__(self, filepath,  monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                 mode='auto', save_freq='epoch', options=None, **kwargs):
        super(ModelCheckpointWGAN, self).__init__(filepath,  monitor, verbose, save_best_only,
                                                  save_weights_only, mode, save_freq, options=options, **kwargs)

    def _save_model(self, epoch, batch, logs):
        """Saves the model.
        ML: The source-code is largely identical to Keras v2.6.0 implementation except that two models,
            the critic and the generator, are saved separately in filepath_gen and filepath_critic (see below).
            Modified source-code is envelopped between 'ML S' and 'ML E'-comment strings.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
              is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, batch, logs)
            # ML S
            filepath_gen = os.path.join(filepath, f"{self.model.modelname}_generator")
            filepath_critic = os.path.join(filepath, f"{self.model.modelname}_critic")
            # ML E

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning('Can save best model only with %s available, skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s' % (epoch + 1, self.monitor,
                                                               self.best, current, filepath))
                            self.best = current
                            # ML S
                            if self.save_weights_only:
                                self.model.generator.save_weights(
                                    filepath_gen, overwrite=True, options=self._options)
                                self.model.critic.save_weights(
                                    filepath_critic, overwrite=True, options=self._options)
                            else:
                                self.model.generator.save(filepath_gen, overwrite=True, options=self._options)
                                self.model.critic.save(filepath_critic, overwrite=True, options=self._options)
                            # ML E
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    # ML S
                    if self.save_weights_only:
                        self.model.generator.save_weights(
                            filepath_gen, overwrite=True, options=self._options)
                        self.model.critic.save_weights(
                            filepath_critic, overwrite=True, options=self._options)
                    else:
                        self.model.generator.save(filepath_gen, overwrite=True, options=self._options)
                        self.model.critic.save(filepath_critic, overwrite=True, options=self._options)
                    # ML E
                self._maybe_remove_file()
            except IsADirectoryError as e:  # h5py 3.x
                raise IOError('Please specify a non-directory filepath for'  
                              'ModelCheckpoint. Filepath used is an existing directory: {}'.format(filepath))
            except IOError as e:  # h5py 2.x
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                                  'ModelCheckpoint. Filepath used is an existing directory: {}'.format(filepath))
                # Re-throw the error for any other causes.
                raise e
