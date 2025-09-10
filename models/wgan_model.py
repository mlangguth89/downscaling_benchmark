# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

"""
Class for Wasserstein GAN (WGAN) model.

To-Dos:
    - Implement learning rate schedule and warmup for Horovod
"""


__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-05-19"
__update__ = "2025-07-17"

import os
import glob
from typing import List, Tuple, Union
import inspect
from collections import OrderedDict
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model as k_plot_model
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
try:
    import horovod.tensorflow as hvd
except:
    print("Horovod is not installed. Distributed training is not supported.")
    pass

# all the layers used for U-net
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
# other modules
from abstract_model_class import AbstractModelClass
from unet_model import UNetModelBase
from custom_losses import get_custom_loss
from advanced_activations import advanced_activation
from model_utils import save_opt_weights


list_or_tuple = Union[List, Tuple]


class Critic_Simple(AbstractModelClass):
    
    def __init__(self, shape_in: List, hparams: dict, varnames_tar: List):
        super().__init__(shape_in, hparams, varnames_tar, "", "")       # Pass empty savedir- and expname-arguments since this is not a stand-alone model
        building_blocks = UNetModelBase()
        self.conv_block = building_blocks.conv_block
        
        # set submodels
        self.set_hparams(hparams)
        self.set_model()
        
    def set_model(self):
        critic_in = Input(shape=self._input_shape)
        x = critic_in
        
        channels = self.hparams["channels_start"]
        num_conv = int(self.hparams["num_conv"])
        
        assert num_conv > 1, f"Number of convolutional layers is {num_conv:d}, but must be at minimum 2."
        

        for _ in range(num_conv):
            x = self.conv_block(x, channels, self.hparams["kernel"], self.hparams["stride"], 
                                activation=self.hparams["activation"], l_batch_normalization=self.hparams["lbatch_norm"])
            channels *= 2
            
        # finally perform global average pooling and finalize by fully connected layers
        x = GlobalAveragePooling2D()(x)
        try:
            x = Dense(self.hparams["channels_start"], activation=self.hparams["activation"])(x)
        except ValueError as _:
            ac = advanced_activation(self.hparams["activation"]) 
            x = Dense(self.hparams["channels_start"], activation=ac)(x)
        # ... and end with linear output layer
        out = Dense(1, activation="linear")(x)

        self.model = Model(inputs=critic_in, outputs=out)
                   
    def set_compile_options(self):
        raise RuntimeError(f"Critic model is supposed to be part of a composite model such as WGAN, but not as standalone model for training.")
        
    def set_fit_options(self):
        raise RuntimeError(f"Critic model is supposed to be part of a composite model such as WGAN, but not as standalone model for training.")
        
    def set_hparams_default(self):
        """
        Note: hyperparameter defaults of generator and critic model must be set in the respective model classes whose instances are just parsed here.
        """
        self.hparams_default = {"num_conv": 4, "channels_start": 64, "activation": "swish",
                                "lbatch_norm": True, "kernel": (3, 3), "stride": (2, 2), "lr": 1.e-06, "lr_end": 1.e-07}


class Sha_WGAN_Model(keras.Model):
    def __init__(self, generator, critic, hparams):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.hparams = hparams  
        self._n_predictands, self._n_predictands_dyn = self._get_npredictands()        
        
    def _get_npredictands(self):
        """
        Return the number of the generator's output channels, i.e. 2 if the U-Net uses an activated z_branch.
        """
        #npredictands = 
        return self.generator.__dict__["_n_predictands"], self.generator.__dict__["_n_predictands_dyn"] 

    def compile(self, optimizer, loss, **kwargs):
        super().compile(**kwargs)
        self.c_optimizer, self.g_optimizer = optimizer
        
        # losses
        self.recon_loss = loss
        self.critic_loss = get_custom_loss("critic")
        self.critic_gen_loss = get_custom_loss("critic_generator")
        
    @tf.function
    def train_step(self, data_iter: tf.data.Dataset, embed=None) -> OrderedDict:
        """
        Training step for Wasserstein GAN
        :param data_iter: Tensorflow Dataset providing training data
        :param embed: embedding (not implemented yet)
        :return: Ordered dictionary with several losses of generator and critic
        """

        predictors, predictands = data_iter

        # train the critic d_steps-times
        for i in range(self.hparams["d_steps"]):
            with tf.GradientTape() as tape_critic:
                ist, ie = i * self.hparams["batch_size"], (i + 1) * self.hparams["batch_size"]
                # critic only operates on predictand channels
                if self._n_predictands_dyn > 1:
                    predictands_critic = predictands[ist:ie, :, :, 0:self._n_predictands_dyn]
                else:
                    predictands_critic = tf.expand_dims(predictands[ist:ie, :, :, 0], axis=-1)
                # generate (downscaled) data
                gen_data = self.generator.model(predictors[ist:ie, ...], training=True)
                # calculate critics for both, the real and the generated data
                critic_gen = self.critic.model(gen_data[..., 0:self._n_predictands_dyn], training=True)
                critic_gt = self.critic.model(predictands_critic, training=True)
                # calculate the loss (incl. gradient penalty)
                c_loss = self.critic_loss(critic_gt, critic_gen)
                gp = self.gradient_penalty(predictands_critic, gen_data[..., 0:self._n_predictands_dyn])

                d_loss = c_loss + self.hparams["gp_weight"] * gp

            # calculate gradients and update discrimintor
            d_gradient = tape_critic.gradient(d_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(d_gradient, self.critic.trainable_variables))

        # train generator
        with tf.GradientTape() as tape_generator:
            # generate (downscaled) data
            gen_data = self.generator.model(predictors[-self.hparams["batch_size"]:, :, :, :], training=True)
            # get the critic and calculate corresponding generator losses (critic and reconstruction loss)
            critic_gen = self.critic.model(gen_data[..., 0:self._n_predictands_dyn], training=True)
            cg_loss = self.critic_gen_loss(critic_gen)
            rloss = self.recon_loss(predictands[-self.hparams["batch_size"]:, :, :, :], gen_data)

            g_loss = cg_loss + self.hparams["recon_weight"] * rloss

        g_gradient = tape_generator.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        return OrderedDict(
            [("c_loss", c_loss), ("gp_loss", self.hparams["gp_weight"] * gp), ("d_loss", d_loss), ("cg_loss", cg_loss),
             ("recon_loss", rloss * self.hparams["recon_weight"]), ("g_loss", g_loss)])

    def test_step(self, val_iter: tf.data.Dataset) -> OrderedDict:
        """
        Implement step to test trained generator on validation data
        :param val_iter: Tensorflow Dataset with validation data
        :return: dictionary with reconstruction loss on validation data
        """
        predictors, predictands = val_iter

        gen_data = self.generator.model(predictors, training=False)
        rloss = self.recon_loss(predictands, gen_data)

        return OrderedDict([("recon_loss", rloss)])

    def predict_step(self, test_iter: tf.data.Dataset) -> OrderedDict:

        predictors, _ = test_iter

        return self.generator.model(predictors, training=False)

    def gradient_penalty(self, real_data, gen_data):
        """
        Calculates gradient penalty based on 'mixture' of generated and ground truth data
        :param real_data: the ground truth data
        :param gen_data: the generated/predicted data
        :return: gradient penalty
        """
        # get mixture of generated and ground truth data
        alpha = tf.random.normal([self.hparams["batch_size"], 1, 1, self._n_predictands_dyn], 0., 1.)
        mix_data = real_data + alpha * (gen_data - real_data)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(mix_data)
            critic_mix = self.critic.model(mix_data, training=True)

        # calculate the gradient on the mixture data...
        grads_mix = gp_tape.gradient(critic_mix, [mix_data])[0]
        # ... and norm it
        norm = tf.sqrt(tf.reduce_mean(tf.square(grads_mix), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.) ** 2)

        return gp
    
    def save(self, filepath: str, expname: str, overwrite: bool = True, include_optimizer: bool = True, save_format: str = "tf",
             signatures=None, options=None, save_traces: bool = True, suffix: str = "_last"):
        """
        TODO: This should be in the AbstractModelClass to also allow inheritcance.
        Note 2025-07-21: This does not work yet.

        Save generator and critic seperately.
        The parameters of this method are equivalent to Keras.model.save ensuring full functionality.
        :param filepath: path to SavedModel or H5 file to save both models
        :param overwrite: Whether to silently overwrite any existing file at the target location, or provide the user
                          with a manual prompt.
        :param include_optimizer: If True, save optimizer's state together.
        :param save_format: Currently, only the 'tf' format is supported.
        :param signatures: Signatures to save with the SavedModel. Applicable to the 'tf' format only.
                           Please see the `signatures` argument in `tf.saved_model.save` for details.
        :param options: (only applies to SavedModel format) `tf.saved_model.SaveOptions` object that specifies options
                        for saving to SavedModel.
        :param save_traces: (only applies to SavedModel format) When enabled, the SavedModel will store the function
                            traces for each layer. This can be disabled, so that only the configs of each layer are
                            stored.  Defaults to `True`. Disabling this will decrease
                            serialization time and reduce file size, but it requires that
                            all custom layers/models implement a `get_config()` method.
        :return: -
        """       
        assert save_format != "h5", f"h5 is not supported as save format for this model"            

        # save generator and critic seperately
        generator_path, critic_path = Path(filepath).joinpath(f"{expname}_generator{suffix}"), \
                                      Path(filepath).joinpath(f"{expname}_critic{suffix}")
        
        os.makedirs(generator_path, exist_ok =True)
        os.makedirs(critic_path, exist_ok =True)
        
        if tf.__version__ >= "2.12.0":
            self.generator.save(generator_path, overwrite, save_format)
            self.critic.save(critic_path, overwrite, save_format)
        else:
            self.generator.save(generator_path, overwrite, include_optimizer, save_format, signatures, options, save_traces)
            self.critic.save(critic_path, overwrite, include_optimizer, save_format, signatures, options, save_traces)
        
        # save weights and optimizer state seperately, since the latter is not supported by Keras' save-method due to a bug
        # https://github.com/keras-team/tf-keras/issues/504
        # Note that it also does not work when choosing the h5-format (and when setting include_otimizer = False as in previous TF versions) 
        if include_optimizer:    # required to resume training    
            self.generator.save_weights(generator_path.joinpath(f"{expname}_generator{suffix}"), overwrite=overwrite,
                                        save_format=save_format, options=options)
            self.critic.save_weights(critic_path.joinpath(f"{expname}_critic{suffix}"), overwrite=overwrite,
                                     save_format=save_format, options=options)
        
            
            generator_opt = generator_path.joinpath(f"{expname}_generator_opt{suffix}.pkl")
            critic_opt = critic_path.joinpath(f"{expname}_critic_opt{suffix}.pkl")
            
            print(f"Save generator optimiter state to {generator_opt}...")
            save_opt_weights(self.g_optimizer, generator_opt)
            print(f"Save critic optimiter state to {critic_opt}...")
            save_opt_weights(self.c_optimizer, critic_opt)

    
    

class Sha_WGAN(AbstractModelClass):
    
    def __init__(self, generator: AbstractModelClass, critic: AbstractModelClass, shape_in: List, hparams: dict, varnames_tar: List, savedir: str, expname: str,
                 with_horovod: bool = False):
        """
        Imsantiate a Wasserstein GAN (WGAN) model.
        :param generator: generator model returning the downsampled data
        :param critic: critic model returning the critic score on the data
        :param shape_in: shape of the input data
        :param hparams: hyperparameters for WGAN
        :param varnames_tar: list of target variable names
        :param savedir: directory to save the model
        :param expname: name of the experiment
        :param with_horovod: whether to use horovod for distributed training
        """
        if not shape_in:                    # shape_in can be None when loading model for inference -> set dummy-value to allow model construction
            shape_in = [8, 8, 8]
        super().__init__(shape_in, hparams, varnames_tar, savedir, expname)

        # flag if horovod is used and import required modules
        self.with_horovod = with_horovod
        self.main_process = True 
        if self.with_horovod:
            self.main_process = hvd.rank() == 0

        # set hyperparmaters
        self.set_hparams(hparams)
        # set submodels
        self.generator, self.critic = self.set_model(generator, critic)
        # set compile and fit options as well as custom objects
        self.set_compile_options()
        self.set_custom_objects(loss=self.compile_options['loss'])
        self.set_fit_options()

    
    def set_compile_options(self):

        # Ensure that training is not distributed (not supported yet)
        if self.with_horovod:
            raise RuntimeError("Sha WGAN does not support Horovod yet.")
        
        # set optimizers
        # check if optimizer is valid and set corresponding optimizers for generator and critic
        if self.hparams["optimizer"].lower() == "adam":
            optimizer = keras.optimizers.Adam
            kwargs_opt = {"beta_1": 0.0, "beta_2": 0.9}
        elif self.hparams["optimizer"].lower() == "rmsprop":
            optimizer = keras.optimizers.RMSprop
            kwargs_opt = {}
        else:
            raise ValueError("'{0}' is not a valid optimizer. Either choose Adam or RMSprop-optimizer")

        self.optimizer = (optimizer(self.critic.hparams["lr"], **kwargs_opt), 
                          optimizer(self.generator.hparams["lr"], **kwargs_opt))

        self.loss = self.get_recon_loss()
        
    def get_fit_options(self):
        """
        Add callbacks to the fit options.
        """
        wgan_callbacks = []
        
        if self.hparams["lr_decay"]:
            wgan_callbacks.append(LearningRateSchedulerWGAN(self.get_lr_decay(), verbose=1))
        
        if self.hparams["lcheckpointing"] and self.main_process:   
            # checkpoint model after each epoch (save best model only is set to False)       
            wgan_callbacks.append(ModelCheckpointWGAN(self._savedir, self._expname,
                                                      monitor="val_recon_loss", verbose=1, save_best_only=False, mode="min"))
            
        if self.hparams["learlystopping"]:
            wgan_callbacks.append(EarlyStopping(monitor="val_recon_loss", patience=8))

        if self.with_horovod:            
            wgan_callbacks.append(BroadcastWeightsCallback(self.generator, self.critic, self.optimizer[0], self.optimizer[1]))

        if wgan_callbacks is not None:
            return {"callbacks": wgan_callbacks}
        else:
            return {}  
        
    def set_model(self, generator, critic):
        """
        Setting the WGAN-model is a three-step approach:
            1. Get the generator model
            2. Get the critic model
            3. Put the generator and critic model into the actual WGAN
        """
        # get generator model
        add_opts = {"concat_out": True} if "concat_out" in str(inspect.signature(generator)) else {}         # generator might have a concat_out-argument to handle z_branch-outputs
        gen_model = generator(self._input_shape, self.hparams["hparams_generator"], self._varnames_tar, self._savedir, 
                              self._expname, **add_opts)        
        # correct number of dynamic predictors
        self._n_predictands_dyn = gen_model.__dict__["_n_predictands_dyn"]
        
        # get critic model
        tar_shape = (*self._input_shape[:-1], self._n_predictands_dyn)   # critic only accounts for dynamic predictands
        critic_model = critic(tar_shape, self.hparams["hparams_critic"], self._varnames_tar)
        
        # get hyperparamters of WGAN only
        hparams_wgan_only = self.hparams.copy()
        hparams_wgan_only.pop("hparams_critic")
        hparams_wgan_only.pop("hparams_generator")
                
        # ...and create WGAN model instance
        self.model = Sha_WGAN_Model(gen_model, critic_model, hparams_wgan_only)

        return gen_model, critic_model
    
    def get_recon_loss(self):

        kwargs_loss = {}
        if "vec" in self.hparams["recon_loss"]:
            kwargs_loss = {"nd_vec": self.hparams.get("nd_vec", 2), "n_channels": self._n_predictands}
        elif "channels" in self.hparams["recon_loss"]:
            kwargs_loss = {"n_channels": self._n_predictands}

        loss_fn = get_custom_loss(self.hparams["recon_loss"], **kwargs_loss)
        print(f"Using {self.hparams['recon_loss']} as reconstruction loss function.")

        return loss_fn
        
    
    def get_lr_decay(self):
        """
        Returns a learning rate scheduler that directly sets the learning rate
        for both generator and critic based on the epoch number, using
        exponential decay from their respective lr_start to lr_end.
        """
        decay_st, decay_end = self.hparams["decay_start"], self.hparams["decay_end"]
    
        # Generator LR params
        lr_start_gen = self.hparams["hparams_generator"]["lr"]
        lr_end_gen = self.hparams["hparams_generator"]["lr_end"]
    
        # Critic LR params
        lr_start_critic = self.hparams["hparams_critic"]["lr"]
        lr_end_critic = self.hparams["hparams_critic"]["lr_end"]
    
        if not decay_end > decay_st:
            raise ValueError("Epoch for end of learning rate decay must be larger than start epoch. "
                             f"Your values: {decay_st}, {decay_end}")
    
        def lr_scheduler(epoch, lr=None):
            def compute_lr(lr_start, lr_end):
                if epoch < decay_st:
                    return lr_start
                elif epoch >= decay_end:
                    return lr_end
                else:
                    progress = (epoch - decay_st) / (decay_end - decay_st)
                    return lr_start * tf.exp(progress * tf.math.log(lr_end / lr_start))
    
            return compute_lr(lr_start_gen, lr_end_gen), compute_lr(lr_start_critic, lr_end_critic)
    
        return lr_scheduler


    def plot_model(self, save_dir, **kwargs):
        """
        Plot generator and critic model separately.
        :param save_dir: directory under which plots will be saved
        :param kwargs: All keyword arguments valid for tf.keras.utils.plot_model
        """
        k_plot_model(self.generator, os.path.join(save_dir, f"plot_{self._expname}_generator.png"), **kwargs)
        k_plot_model(self.critic, os.path.join(save_dir, f"plot_{self._expname}_critic.png"), **kwargs)


    def load_checkpoint(self, checkpoint_dir, checkpoint_format: str = "h5"):
        """
        Load model from checkpoint that has been either saved with the save-method or with the Checkpoint-callback.
        Requires that the model is compiled! 
        :param checkpoint_dir": Base-directory where checkpointed model is saved (must contain generator and critic separately)
        :param checkpoint_format: format of checkpoint, must match the format used for saving.
        :return: iteration step of checkpointed model
        """
        generator_path, critic_path = Path(checkpoint_dir).joinpath(f"{self._expname}_generator*"), \
                                      Path(checkpoint_dir).joinpath(f"{self._expname}_critic*")
        
        matching_gen_dir, matching_critic_dir = glob.glob(str(generator_path)), glob.glob(str(critic_path))
        
        if matching_gen_dir:
            generator_path = Path(matching_gen_dir[0])
            suffix_gen = str(generator_path).split("_generator")[-1]
        else:
            raise FileNotFoundError(f"No matching director for generator-model {str(generator_path)} found.")
            
        if matching_critic_dir:
            critic_path = Path(matching_critic_dir[0])
            suffix_critic = str(critic_path).split("_critic")[-1]
        else:
            raise FileNotFoundError(f"No matching director for generator-model {str(critic_path)} found.")
        
        self.generator.load_weights(generator_path.joinpath(f"{self._expname}_generator{suffix_gen}"))
        self.critic.load_weights(critic_path.joinpath(f"{self._expname}_critic{suffix_critic}"))

        opt_gen_path, opt_critic_path = generator_path.joinpath(f"{self._expname}_generator_opt{suffix_gen}.pkl"), \
                                        critic_path.joinpath(f"{self._expname}_critic_opt{suffix_critic}.pkl")
        with open(opt_gen_path, "rb") as f:
            optimizer_weights_gen = pickle.load(f)

        with open(opt_critic_path, "rb") as f:
            optimizer_weights_critic = pickle.load(f)

        # Note the optimizers are directly accessible from the model-class due to the __getattr__-method of AbstractModelClass
        # set state for g_optimizer
        self.g_optimizer._create_all_weights(self.generator.trainable_variables)
        self.g_optimizer.set_weights(optimizer_weights_gen)

        # set state for c_optimizer
        self.c_optimizer._create_all_weights(self.critic.trainable_variables)
        self.c_optimizer.set_weights(optimizer_weights_critic)

        # retrieve iteration step of checkpointed model
        iter_step = (self.g_optimizer.variables()[0]).numpy()

        return iter_step

    

    def load_inference_model(self, model_dir, format="tf"):
            
        # construct directories to generator- and critic model from model directory
        model_dir = Path(model_dir)

        expname = model_dir.name
        suffix = expname.split("_")[-1]

        fname_suffix = ".h5" if format == "h5" else ""

        gen_dir = model_dir.joinpath(expname.replace(suffix, f"generator_{suffix}{fname_suffix}"))

        # load saved models
        generator = keras.models.load_model(gen_dir, compile=False)

        return generator

    def count_params(self):
        """
        Count number of trainable and untrainable parameters
        """
        trainable_param = int(np.sum([K.count_params(p) for p in self.generator.trainable_weights]))
        untrainable_param = int(np.sum([K.count_params(p) for p in self.generator.non_trainable_weights]))

        trainable_param += int(np.sum([K.count_params(p) for p in self.critic.trainable_weights]))
        untrainable_param += int(np.sum([K.count_params(p) for p in self.critic.non_trainable_weights]))
        
        return trainable_param, untrainable_param

                          
    def set_hparams_default(self):
        """
        Note I: Hyperparametrs whose default is None must be parsed in any case.        
        Note II: Hyperparameter defaults of generator and critic model must be set in the respective model classes whose instances are just parsed here.
              Thus, the default just sets empty dictionaries.
        """
        self.hparams_default = {"d_steps": None, "nepochs": None, "batch_size": 32, "lr_decay": False, "decay_start": 3, "decay_end": 20, 
                                "l_embed": False, "recon_weight": 1000., "gp_weight": 10., "optimizer": "adam", 
                                "lcheckpointing": True, "learlystopping": False, "recon_loss": "mae_channels",
                                "hparams_generator": {}, "hparams_critic": {}}

                          
class LearningRateSchedulerWGAN(LearningRateScheduler):
    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerWGAN, self).__init__(schedule, verbose)

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, "g_optimizer"):
            raise AttributeError('Model must have a "g_optimizer" for optimizing the generator.')

        if not hasattr(self.model, "c_optimizer"):
            raise AttributeError('Model must have a "c_optimizer" for optimizing the critic.')

        if not (hasattr(self.model.g_optimizer, "lr") and hasattr(self.model.c_optimizer, "lr")):
            raise ValueError('Both generator and critic optimizers must have an "lr" attribute.')

        try:
            # Use schedule to directly compute both learning rates based on epoch
            lr_g, lr_c = self.schedule(epoch)
        except TypeError:
            raise NotImplementedError(
                "Learning rate schedule must accept a single argument (epoch) and return (lr_g, lr_c)."
            )

        # Validate output types
        if not (isinstance(lr_g, (tf.Tensor, float, np.float32, np.float64)) and
                isinstance(lr_c, (tf.Tensor, float, np.float32, np.float64))):
            raise ValueError(f'The schedule must return float values. Got: {type(lr_g)} (generator), {type(lr_c)} (critic)')

        # Set new learning rates
        K.set_value(self.model.g_optimizer.lr, lr_g)
        K.set_value(self.model.c_optimizer.lr, lr_c)

        if self.verbose > 0:
            print(f'\nEpoch {epoch + 1}: LearningRateScheduler set learning rate '
                  f'for generator to {lr_g:.8f}, for critic to {lr_c:.8f}.')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr_generator'] = K.get_value(self.model.g_optimizer.lr)
        logs['lr_critic'] = K.get_value(self.model.c_optimizer.lr)


class ModelCheckpointWGAN(ModelCheckpoint):
    """
    ModelCheckpoint callback for WGAN mode by savin the generator and critic models separately.
    """
    def __init__(self, filepath, expname, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                 mode='auto', save_freq="epoch", options=None, **kwargs):
        super(ModelCheckpointWGAN, self).__init__(filepath,  monitor, verbose, save_best_only,
                                                  save_weights_only, mode, save_freq, options=options, **kwargs)
        self._expname = expname

    def _save_model(self, epoch, batch, logs):
        """Saves the model according to save_best_only flag:
        - If save_best_only=True: save only the best model
        - If save_best_only=False: save every epoch and also the best model if improved
        """
        logs = logs or {}
    
        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
    
            current = logs.get(self.monitor)
    
            if self.save_best_only:
                # Save only the best model
                if current is None:
                    logging.warning(
                        'Can save best model only with %s available, skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        filepath_best = Path(self._get_file_path(epoch, batch, logs)).joinpath(f"{self._expname}_best")
                        if self.verbose > 0:
                            print(
                                f'\nEpoch {epoch + 1:05d}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving best model to {filepath_best}'
                            )
                        self.best = current
                        self.model.save(
                            filepath_best,
                            self._expname,
                            overwrite=True,
                            include_optimizer=not self.save_weights_only,
                            save_format="tf",
                            suffix="_best"
                        )
                    else:
                        if self.verbose > 0:
                            print(
                                f'\nEpoch {epoch + 1:05d}: {self.monitor} did not improve from {self.best:.5f}'
                            )
            else:
                # save_best_only=False: Save every epoch model
                add_str_epoch = f"_epoch{epoch+1:05d}"
                filepath_epoch = Path(self._get_file_path(epoch, batch, logs)).joinpath(f"{self._expname}{add_str_epoch}")
    
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1:05d}: saving model to {filepath_epoch}')
    
                self.model.save(
                    filepath_epoch,
                    self._expname,
                    overwrite=True,
                    include_optimizer=not self.save_weights_only,
                    save_format="tf",
                    suffix=add_str_epoch
                )
    
                # And also save best model if improved
                if current is None:
                    logging.warning(
                        'Can save best model only with %s available, skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        filepath_best = Path(self._get_file_path(epoch, batch, logs)).joinpath(f"{self._expname}_best")
                        if self.verbose > 0:
                            print(
                                f'\nEpoch {epoch + 1:05d}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving best model to {filepath_best}'
                            )
                        self.best = current
                        self.model.save(
                            filepath_best,
                            self._expname,
                            overwrite=True,
                            include_optimizer=not self.save_weights_only,
                            save_format="tf",
                            suffix="_best"
                        )
                    else:
                        if self.verbose > 0:
                            print(
                                f'\nEpoch {epoch + 1:05d}: {self.monitor} did not improve from {self.best:.5f}'
                            )
    
            self._maybe_remove_file()


class BroadcastWeightsCallback(tf.keras.callbacks.Callback):
    """ 
    Custom callback to broadcast model weights at the end of each batch.
    Ensures all workers stay synchronized during training.
    """

    def __init__(self, generator, critic, g_optimizer, c_optimizer, root_rank=0):
        super(BroadcastWeightsCallback, self).__init__()
        self.generator = generator
        self.critic = critic
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.root_rank = root_rank

    def on_epoch_begin(self, epoch, logs=None):
        # broadcast weights at the beginning of the first epoch
        if epoch == 0:
            hvd.broadcast_variables(self.generator.variables, root_rank=self.root_rank)
            hvd.broadcast_variables(self.critic.variables, root_rank=self.root_rank)
            hvd.broadcast_variables(self.g_optimizer.variables(), root_rank=self.root_rank)
            hvd.broadcast_variables(self.c_optimizer.variables(), root_rank=self.root_rank)            

