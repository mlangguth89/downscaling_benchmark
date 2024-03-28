# Harris et al 2022, WGAN model implementation
"""
Class for Harris et al 2022, conditional Wasserstein GAN model (CWGAN)
"""
import os
from typing import List, Tuple, Union
import inspect
import numpy as np
from abstract_model_class import AbstractModelClass
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model as k_plot_model
from custom_losses import get_custom_loss
from wgan_model import LearningRateSchedulerWGAN

list_or_tuple = Union[List, Tuple]

class CWGAN_Model(keras.Model):
    def __init__(self, generator, critic, hparams):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.hparams = hparams  
        self._n_predictands, self._n_predictands_dyn = self._get_npredictands()
        
    def _get_npredictands(self):
        """
        Return the number of the generator's output channels, i.e. 2 if the U-Net uses an activated z_branch.
        
        NOTE SL: taken from wgan_model.py
        """
        #npredictands = 
        return self.generator.__dict__["_n_predictands"], self.generator.__dict__["_n_predictands_dyn"] 

    def compile(self, optimizer, loss, **kwargs):
        """
        NOTE SL: taken from wgan_model.py
        """
        super().compile(**kwargs)
        self.c_optimizer, self.g_optimizer = optimizer
        
        # losses
        self.recon_loss = loss
        self.critic_loss = get_custom_loss("critic")
        self.critic_gen_loss = get_custom_loss("critic_generator")
        
    def train_step(self, data_iter: tf.data.Dataset, embed=None) -> OrderedDict:
        pass

    def test_step(self, val_iter: tf.data.Dataset) -> OrderedDict:
        pass

    def predict_step(self, test_iter: tf.data.Dataset) -> OrderedDict:

        predictors, _ = test_iter

        return self.generator.model(predictors, training=False)

    def gradient_penalty(self, real_data, gen_data):
        pass
        



class CWGAN(AbstractModelClass):
    """tbd if this can instead inherit of WGAN"""
    
    def __init__(self, generator: AbstractModelClass, critic: AbstractModelClass, shape_in: List, hparams: dict, varnames_tar: List, savedir: str, expname: str):
        
        super().__init__(shape_in, hparams, varnames_tar, savedir, expname)

        self.modelname = "cwgan"
        
        # set hyperparmaters
        self.set_hparams(hparams)
        # set submodels
        self.generator, self.critic = self.set_model(generator, critic)
        # set compile and fit options as well as custom objects
        self.set_compile_options()
        self.set_custom_objects(loss=self.compile_options['loss'])
        self.set_fit_options()
        
    def set_compile_options(self):
        """
        Note SL: loss function and optimiser perhaps need edits
        """
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

        self.optimizer = (optimizer(self.critic.hparams["lr"], **kwargs_opt), optimizer(self.generator.hparams["lr"], **kwargs_opt))
        self.loss = self.get_recon_loss()
        
    def get_fit_options(self):
        """NOTE SL: taken from wgan_model.py and renaned vars"""
        cwgan_callbacks = []
        
        if self.hparams["lr_decay"]:
            cwgan_callbacks.append(LearningRateSchedulerCWGAN(self.get_lr_decay(), verbose=1))
        
        if self.hparams["lcheckpointing"]:
            cwgan_callbacks.append(ModelCheckpointCWGAN(self._savedir, self._expname, monitor="val_recon_loss", verbose=1, save_best_only=True, mode="min"))
            
        if self.hparams["learlystopping"]:
            cwgan_callbacks.append(EarlyStopping(monitor="val_recon_loss", patience=8))
            
        if cwgan_callbacks is not None:
            return {"callbacks": cwgan_callbacks}
        else:
            return {}  
        
    def set_model(self, generator, critic):
        """
        Setting the CWGAN-model is a three-step approach:
            1. Get the generator model
            2. Get the critic model
            3. Put the generator and critic model into the actual CWGAN
            
        NOTE SL: taken from wgan_model.py; needs adaptations
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
        
        # get hyperparamters of CWGAN only
        hparams_wgan_only = self.hparams.copy()
        hparams_wgan_only.pop("hparams_critic")
        hparams_wgan_only.pop("hparams_generator")
                
        # ...and create CWGAN model instance
        self.model = CWGAN_Model(gen_model, critic_model, hparams_wgan_only)

        return gen_model, critic_model
    
    def get_recon_loss(self):
        """
        NOTE SL: taken from wgan_model.py
        TODO: unsure if it needs adaptation
        """

        kwargs_loss = {}
        if "vec" in self.hparams["recon_loss"]:
            kwargs_loss = {"nd_vec": self.hparams.get("nd_vec", 2), "n_channels": self._n_predictands}
        elif "channels" in self.hparams["recon_loss"]:
            kwargs_loss = {"n_channels": self._n_predictands}

        loss_fn = get_custom_loss(self.hparams["recon_loss"], **kwargs_loss)

        return loss_fn
        
    def get_lr_decay(self):
        """
        Get callable of learning rate scheduler which can be used as callabck in Keras models.
        Exponential decay is applied to change the learning rate from the start to the end value.
        Note that the exponential decay is calculated based on the learning rate of the generator, but applies to both.
        :return: learning rate scheduler
        
        NOTE SL: taken from wgan_model.py
        """
        decay_st, decay_end = self.hparams["decay_start"], self.hparams["decay_end"]
        lr_start, lr_end = self.hparams["hparams_generator"]["lr"], self.hparams["hparams_generator"]["lr_end"]

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

    def plot_model(self, save_dir, **kwargs):
        """
        Plot generator and critic model separately.
        :param save_dir: directory under which plots will be saved
        :param kwargs: All keyword arguments valid for tf.keras.utils.plot_model
        
        NOTE SL: taken from wgan_model.py
        """
        k_plot_model(self.generator, os.path.join(save_dir, f"plot_{self._expname}_generator.png"), **kwargs)
        k_plot_model(self.critic, os.path.join(save_dir, f"plot_{self._expname}_critic.png"), **kwargs)

    def save(self, filepath: str, overwrite: bool = True, include_optimizer: bool = True, save_format: str = None,
             signatures=None, options=None, save_traces: bool = True):
        """
        Save generator and critic seperately.
        The parameters of this method are equivalent to Keras.model.save ensuring full functionality.
        :param filepath: path to SavedModel or H5 file to save both models
        :param overwrite: Whether to silently overwrite any existing file at the target location, or provide the user
                          with a manual prompt.
        :param include_optimizer: If True, save optimizer's state together.
        :param save_format: Either `'tf'` or `'h5'`, indicating whether to save the model to Tensorflow SavedModel or
                            HDF5. Defaults to 'tf' in TF 2.X, and 'h5' in TF 1.X.
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
        generator_path, critic_path = os.path.join(filepath, "{0}_generator_last".format(self._expname)), \
                                      os.path.join(filepath, "{0}_critic_last".format(self._expname))
        self.generator.save(generator_path, overwrite, include_optimizer, save_format, signatures, options, save_traces)
        self.critic.save(critic_path, overwrite, include_optimizer, save_format, signatures, options, save_traces)

                          
    def set_hparams_default(self):
        """
        Note: Hyperparameter defaults taken from 1) https://github.com/ECMWFCode4Earth/tesserugged/blob/master/dev/gan/dsrnngan/local_config.yaml and 2) https://github.com/ECMWFCode4Earth/tesserugged/blob/master/dev/gan/dsrnngan/models.py
        
        NOTE SL: taken from wgan_model.py with slight adaptation
        TODO: unsure still about most of the defaults /discuss
        """
        self.hparams_default = {"batch_size": 2, "nepochs": 30, "lr_decay": False, "decay_start": 3, "decay_end": 20, 
                                "l_embed": False, "d_steps": 5, "recon_weight": 1000., "gp_weight": 10., "optimizer": "adam", 
                                "lcheckpointing": True, "learlystopping": False, "recon_loss": "mae_channels",
                                "hparams_generator": {}, "hparams_critic": {}}


            
class LearningRateSchedulerCWGAN(LearningRateSchedulerWGAN):
    """TODO: assume it can be taken from WGAN for now"""
    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerCWGAN, self).__init__(schedule, verbose)

class ModelCheckpointCWGAN(ModelCheckpoint):
    """TODO: maybe also inherit from WGAN"""
    def __init__(self):
        pass
    
    def _save_model(self, epoch, batch, logs):
        pass
    