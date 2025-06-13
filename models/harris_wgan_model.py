# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

# Harris et al 2022, WGAN model implementation
"""
Class for Harris et al 2022, conditional Wasserstein GAN model (cWGAN)
"""

__author__ = "Sebastian Lehner, Michael Langguth"
__email__ = "sebastian.lehner@geosphere.at, m.langguth@fz-juelich.de"
__date__ = "2024-03-28"
__update__ = "2025-04-10"

from typing import List, Tuple, Union, Dict
from collections import OrderedDict
from pathlib import Path
import numpy as np
from abstract_model_class import AbstractModelClass
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, concatenate, LeakyReLU, UpSampling2D, Layer, Conv2D, Add, AveragePooling2D, GlobalAveragePooling2D, \
                                    Dense, BatchNormalization, LayerNormalization, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.models import Model
try:
    import horovod.tensorflow as hvd
except:
    print("Horovod is not installed. Distributed training is not supported.")
    pass
from custom_losses import get_custom_loss
from wgan_model import Sha_WGAN, LearningRateSchedulerWGAN
from model_utils import save_opt_weights
from tensorflow.python.platform import tf_logging as logging

try:
    import horovod.tensorflow as hvd
except:
    print(f"{__file__}: Horovod is not installed. Distributed training is not supported.")
    pass

list_or_tuple = Union[List, Tuple]

class GeneratorHarris(AbstractModelClass):
    # content based on original generator function from models.py (Harris repo)
    # structure based on Critic_Simple from wgan_model.py
    def __init__(self, shape_in: List, hparams: dict, varnames_tar: List):
        # Pass empty savedir- and expname-arguments since this is not a stand-alone model
        super().__init__(shape_in, hparams, varnames_tar, "", "")
        
        # set submodels
        self.set_hparams(hparams)
        self.set_model()
        
    def set_model(self):
        ds_steps = self.hparams["ds_steps"]
        filters_gen = self.hparams["channels_start"]
        kernel = self.hparams["kernel"]
        relu_alpha = self.hparams["relu_alpha"]
        padding = self.hparams["padding"]
        norm = self.hparams["norm"] 
        force_conv = self.hparams["force_conv"]
        # Network inputs
        # low resolution condition
        generator_input = Input(shape=self._input_shape["lo_res_inputs"], name="lo_res_inputs")
        # constant fields
        const_input = Input(shape=self._input_shape["hi_res_inputs"], name="hi_res_inputs")

        # Convolve constant fields down to match other input dimensions
        upscaled_const_input = const_upscale_block(
            const_input, steps=ds_steps, filters=filters_gen
        )
        # noise
        noise_input = Input(shape=self._input_shape["noise_input"], name="noise_input")
        # Concatenate all inputs together
        generator_output = concatenate(
            [generator_input, upscaled_const_input, noise_input]
        )

        # Pass through 3 residual blocks
        for ii in range(3):
            generator_output = residual_block(
                generator_output,
                filters=filters_gen,
                conv_size=kernel,
                stride=1,
                relu_alpha=relu_alpha,
                padding=padding,
                norm=norm,
                force_conv=force_conv,
            )

        # Upsampling from low-res to high-res with alternating residual blocks
        # In the paper, this was [2*filters_gen, filters_gen] for steps of 5 and 2
        block_channels = [2 * filters_gen] * (len(ds_steps) - 1) + [filters_gen]
        for ii, step in enumerate(ds_steps):
            generator_output = UpSampling2D(size=(step, step), interpolation="bilinear")(
                generator_output
            )

            generator_output = residual_block(
                generator_output,
                filters=block_channels[ii],
                conv_size=kernel,
                stride=1,
                relu_alpha=relu_alpha,
                padding=padding,
                norm=norm,
                force_conv=force_conv,
            )

        # Concatenate with original size constants field
        generator_output = concatenate([generator_output, const_input])

        # Pass through 3 residual blocks
        for ii in range(3):
            generator_output = residual_block(
                generator_output,
                filters=filters_gen,
                conv_size=kernel,
                stride=1,
                relu_alpha=relu_alpha,
                padding=padding,
                norm=norm,
                force_conv=force_conv,
            )

        # Output layer
        actv_out = "linear"
        generator_output = Conv2D(
            filters=1, kernel_size=(1, 1), activation=actv_out, name="output"
        )(generator_output)
        
        self.model = Model(
            inputs=[generator_input, const_input, noise_input],
            outputs=generator_output,
            name="gen",
        )
                   
    def set_compile_options(self):
        raise RuntimeError(f"Generator model is supposed to be part of a composite model such as WGAN, but not as standalone model for training.")
        
    def set_fit_options(self):
        raise RuntimeError(f"Generator model is supposed to be part of a composite model such as WGAN, but not as standalone model for training.")
        
    def set_hparams_default(self):
        """
        Note: hyperparameter defaults of generator and critic model must be set in the respective model classes whose instances are just parsed here.
        """
        self.hparams_default = {"channels_start": 128, "activation": "leaky_relu", "kernel": (3, 3), "stride": (2, 2), "lr": 1.e-05, 
                                "ds_steps": [4,], "padding": "reflect", "relu_alpha": 0.2, "lr_end": 1.e-06, "norm": "batch", "force_conv": True}


class CriticHarris(AbstractModelClass):
    # content based on original discriminator function from models.py (Harris repo)
    # structure based on Critic_Simple from wgan_model.py
    def __init__(self, shape_in: List, hparams: dict, varnames_tar: List):
        # Pass empty savedir- and expname-arguments since this is not a stand-alone model
        super().__init__(shape_in, hparams, varnames_tar, "", "")
        
        # set submodels
        self.set_hparams(hparams)
        self.set_model()
        
    def set_model(self):
        ds_steps = self.hparams["ds_steps"]
        filters_critic = self.hparams["channels_start"]
        kernel = self.hparams["kernel"]
        relu_alpha = self.hparams["relu_alpha"]
        padding = self.hparams["padding"]
        norm = self.hparams["norm"]
        nres_blocks = self.hparams["nres_blocks"]
        force_conv = self.hparams["force_conv"] 

        # Network inputs
        # low resolution condition
        generator_input = Input(shape=self._input_shape["lo_res_inputs"], name="lo_res_inputs")
        # constant fields
        const_input = Input(shape=self._input_shape["hi_res_inputs"], name="hi_res_inputs")
        # target image
        generator_output = Input(shape=self._input_shape["output"], name="output")

        # convolve down constant fields to match ERA
        lo_res_const_input = const_upscale_block(
            const_input, steps=ds_steps, filters=filters_critic
        )

        # concatenate constants to lo-res input
        lo_res_input = concatenate([generator_input, lo_res_const_input])

        # concatenate constants to hi-res input
        hi_res_input = concatenate([generator_output, const_input])

        # encode inputs using residual blocks
        # In the paper, this was [filters_disc, 2*filters_disc] for steps of 5 and 2
        block_channels = [filters_critic] * (len(ds_steps) - 1) + [2 * filters_critic]

        for ii, step in enumerate(ds_steps):
            lo_res_input = residual_block(
                lo_res_input,
                filters=block_channels[ii],
                conv_size=kernel,
                stride=1,
                relu_alpha=relu_alpha,
                padding=padding,
                norm = norm,
                force_conv=force_conv,
            )
            hi_res_input = Conv2D(
                filters=block_channels[ii],
                kernel_size=(step, step),
                strides=step,
                padding="valid",
                activation="relu",
            )(hi_res_input)

            hi_res_input = residual_block(
                hi_res_input,
                filters=block_channels[ii],
                conv_size=kernel,
                stride=1,
                relu_alpha=relu_alpha,
                padding=padding,
                norm = norm,
                force_conv=force_conv,
            )

        # concatenate hi- and lo-res inputs channel-wise before passing through critic
        critic_input = concatenate([lo_res_input, hi_res_input])

        # encode in residual blocks
        for _ in range(nres_blocks):
            critic_input = residual_block(
                critic_input,
                filters=filters_critic,
                conv_size=kernel,
                stride=1,
                relu_alpha=relu_alpha,
                padding=padding,
                norm = norm,
                force_conv=force_conv,
            )

        # critic output
        critic_output = GlobalAveragePooling2D()(critic_input)
        # add additional normalization if desired
        critic_output = Dense(64)(critic_output)
        if norm == "batch":
            critic_output = BatchNormalization()(critic_output)
        elif norm == "layer":
           critic_output = LayerNormalization()(critic_output)
        critic_output = ReLU()(critic_output)

        critic_output = Dense(1, name="critic_output")(critic_output)

        self.model = Model(
            inputs=[generator_input, const_input, generator_output],
            outputs=critic_output,
            name="critic",
        )
                   
    def set_compile_options(self):
        raise RuntimeError(f"critic model is supposed to be part of a composite model such as WGAN, but not as standalone model for training.")
        
    def set_fit_options(self):
        raise RuntimeError(f"critic model is supposed to be part of a composite model such as WGAN, but not as standalone model for training.")
        
    def set_hparams_default(self):
        """
        Note: hyperparameter defaults of generator and critic model must be set in the respective model classes whose instances are just parsed here.
        """
        self.hparams_default = {"channels_start": 512, "activation": "leaky_relu", "kernel": (3, 3), "stride": (2, 2), 
                                "lr": 5.e-07, "ds_steps": [4,], "padding": "reflect", "relu_alpha": 0.2, "lr_end": 5.e-08, 
                                "norm": None, "nres_blocks": 2, "force_conv": True}

    
class NoiseGenerator(object):
    """Used for the Generator to generate the random noise input"""
    def __init__(self, noise_shapes, batch_size=32, random_seed=None):
        self.noise_shapes = noise_shapes
        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)

    def noise(self, shape, mean, std):
        shape = [self.batch_size] + shape
        n = self.prng.randn(*shape).astype(np.float32)
        # n = np.zeros(shape, dtype=np.float32)
        if std != 1.0:
            n *= std
        if mean != 0.0:
            n += mean
        return n
    
    def __call__(self, mean=0.0, std=1.):
        return self.noise(self.noise_shapes, mean, std)
    
    

class Harris_WGAN_Model(keras.Model):
    def __init__(self, generator, critic, hparams, expname, noise_gen=None):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.hparams = hparams
        self._expname = expname
        self.noise_gen = noise_gen
        if self.hparams["noise_input"]:
            self.noise_amplitude = 1.
        else:
            self.noise_amplitude = 0.
            self.hparams["noise_channels"] = 1

        if noise_gen:
            assert isinstance(noise_gen, NoiseGenerator), "Parsed noise_gen must be an instance of the NoiseGenerator-class"
        
    def compile(self, optimizer, loss, **kwargs):
        super().compile(**kwargs)
        self.c_optimizer, self.g_optimizer = optimizer
        
        # losses
        if not self.noise_gen:        
            self.noise_gen = NoiseGenerator(
                self.generator._input_shape["lo_res_inputs"][:2]+[self.hparams["noise_channels"]],
                self.hparams["batch_size"]*(self.hparams["d_steps"] + 1)
            )

        # losses
        self.critic_loss = get_custom_loss("critic")
        self.critic_gen_loss = get_custom_loss("critic_generator")
        self.recon_loss = loss
        
    @tf.function    
    def train_step(self, data_iter: Dict, embed=None) -> OrderedDict:
        inputs, outputs = data_iter
        cond = inputs["lo_res_inputs"]
        const = inputs["hi_res_inputs"]
        if self.hparams["ensemble_size"] is None:
            noise = self.noise_gen(std=self.noise_amplitude)
        else:
            # ensemble stacked in an additional dimension at the end
            noise = tf.stack([self.noise_gen(std=self.noise_amplitude) for _ in range(self.hparams["ensemble_size"] + 1)], axis=-1)
        sample = outputs["output"]

        # train critic
        for i in range(self.hparams["d_steps"]):
            with tf.GradientTape() as tape_critic:
                
                ist, ie = i * self.hparams["batch_size"], (i + 1) * self.hparams["batch_size"]
                cond_iter = cond[ist:ie, ...]
                const_iter = const[ist:ie, ...]
                sample_iter = sample[ist:ie, ...]
                noise_iter = noise[ist:ie, ..., 0] # only take the first ensemble member

                gen_in = [cond_iter] + [const_iter] + [noise_iter]
                gen_out = self.generator.model(gen_in, training=True)
                critic_in_gen = [cond_iter] + [const_iter] + [gen_out]
                critic_in_gt = [cond_iter] + [const_iter] + [sample_iter]
                
                # calculate critic for both, the real and the generated data
                critic_gen = self.critic.model(critic_in_gen, training=True)
                critic_gt = self.critic.model(critic_in_gt, training=True)
                # calculate the loss (incl. gradient penalty)
                c_loss = self.critic_loss(critic_gt, critic_gen)
                #gp = GradientPenalty()([sample_iter, gen_out])
                gp = self.gradient_penalty(sample_iter, gen_out, cond_iter, const_iter)
                d_loss = c_loss + self.hparams["gp_weight"] * gp

            # calculate gradients and update critic
            tape_critic = hvd.DistributedGradientTape(tape_critic)

            d_gradient = tape_critic.gradient(d_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(d_gradient, self.critic.trainable_variables))

        # train generator
        with tf.GradientTape() as tape_generator:
            # generate (downscaled) data
            cond_iter = cond[-self.hparams["batch_size"]:, ...]
            const_iter = const[-self.hparams["batch_size"]:, ...]
            noise_iter = noise[-self.hparams["batch_size"]:, ...]
            sample_iter = sample[-self.hparams["batch_size"]:, ...]

            # train generator for each ensemble member
            noise_iter_k = noise_iter[..., 0]
            gen_in = [cond_iter] + [const_iter] + [noise_iter_k]
            gen_data = self.generator.model(gen_in, training=True)
            gen_data_list = [gen_data]
            if self.hparams["ensemble_size"] is not None:
                gen_iter_list = []
                for k in range(self.hparams["ensemble_size"]):
                    noise_iter_k = noise_iter[..., k+1]
                    gen_in = [cond_iter] + [const_iter] + [noise_iter_k]
                    gen_data_iter = self.generator.model(gen_in, training=True)
                    gen_iter_list.append(gen_data_iter)
                    
                gen_data_list.append(tf.stack(gen_iter_list))
            
            critic_in_gen = [cond_iter] + [const_iter] + [gen_data]
            critic_gen = self.critic.model(critic_in_gen, training=True)

            # critic loss for generator
            cg_loss = self.critic_gen_loss(critic_gen)
            # content loss term
            data_ens = gen_data_list[-1]
            cl_loss = self.recon_loss(sample_iter, data_ens)
            #cl_loss = self.recon_loss(sample_iter, gen_data_list[-1])
            # combined loss for generator
            g_loss = cg_loss + cl_loss*self.hparams["recon_weight"]

        tape_generator = hvd.DistributedGradientTape(tape_generator)
        
        g_gradient = tape_generator.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        return OrderedDict(
            [
                ("c_loss", c_loss),
                ("gp_loss", self.hparams["gp_weight"] * gp),
                ("d_loss", d_loss),
                ("cg_loss", cg_loss),
                ("recon_loss", cl_loss * self.hparams["recon_weight"]),
                ("g_loss", g_loss)
            ]
        )
            

    def test_step(self, val_iter: tf.data.Dataset) -> OrderedDict:
        """
        Implement step to test trained generator on validation data
        :param val_iter: Tensorflow Dataset with validation data
        :return: dictionary with reconstruction loss on validation data
        
        NOTE SL: taken from wgan_model.py
        """
        inputs, outputs = val_iter
        cond = inputs["lo_res_inputs"]
        const = inputs["hi_res_inputs"]
        sample = outputs["output"]
        if self.hparams["ensemble_size"] is None:
            noise = self.noise_gen(std=self.noise_amplitude)
        else:
            # ensemble stacked in an additional dimension at the end
            noise = tf.stack([self.noise_gen(std=self.noise_amplitude) for _ in range(self.hparams["ensemble_size"] + 1)], axis=-1)
        
        noise_iter = noise[0:self.hparams["batch_size"]:, ...]
        noise_0 = noise_iter[..., 0]
        gen_in = [cond] + [const] + [noise_0]
        gen_data = self.generator.model(gen_in, training=True)
        gen_data_list = [gen_data]
        if self.hparams["ensemble_size"] is not None:
            gen_iter_list = []
            for k in range(self.hparams["ensemble_size"]):
                noise_k = noise_iter[..., k+1]
                gen_in = [cond] + [const] + [noise_k]
                gen_data_k = self.generator.model(gen_in, training=True)
                gen_iter_list.append(gen_data_k)

            #gen_data_list = tf.stack([gen_data] + gen_iter_list, axis=-1)
            gen_data_list.append(tf.stack(gen_iter_list))

        critic_in_gen = [cond] + [const] + [gen_data]
        critic_gen = self.critic.model(critic_in_gen, training=True)

        # critic loss for generator
        cg_loss = self.critic_gen_loss(critic_gen)
        # content loss term
        cl_loss = self.recon_loss(sample, gen_data_list[-1])

        return OrderedDict([
            ("cg_loss", cg_loss),
            ("recon_loss", cl_loss * self.hparams["recon_weight"]),
        ])

    def predict_step(self, test_iter: tf.data.Dataset) -> OrderedDict:
        inputs, _ = test_iter
        cond = inputs["lo_res_inputs"]
        const = inputs["hi_res_inputs"]
        
        if self.hparams["ensemble_size"] is not None:
            noise = [self.noise_gen(std=self.noise_amplitude) for _ in range(self.hparams["ensemble_size"])]
            gen_list = []
            for noise_iter in noise:
                gen_in = [cond] + [const] + [noise_iter]
                gen_iter = self.generator(gen_in, training=False)
                gen_list.append(gen_iter)
            gen_out = tf.stack(gen_list, axis=-1)
        else:
            noise = self.noise_gen(self.noise_amplitude)
            gen_in = [cond] + [const] + [noise]
            gen_out = self.generator(gen_in, training=False)
        return gen_out

    def gradient_penalty(self, real_data, gen_data, cond_data, const_data):
        """
        Calculates gradient penalty based on 'mixture' of generated and ground truth data
        :param real_data: the ground truth (high-res) data
        :param gen_data: the generated/predicted (high-res) data
        :param cond_data: the conditional (low-res) input data of the generator/critic
        :param const_data: the static (high-res) data of the generator/critic
        :return: gradient penalty
        
        NOTE ML: This is now equivalent to the sage of the GradientPenalty-layer in the original WGAN-implementation,
                 cf. https://github.com/ECMWFCode4Earth/tesserugged/blob/561733660f53a2d3beedc55b593ba68ec260040e/dev/gan/dsrnngan/gan.py#L129C67-L129C69
                 and https://github.com/ECMWFCode4Earth/tesserugged/blob/561733660f53a2d3beedc55b593ba68ec260040e/dev/gan/dsrnngan/layers.py#L13
        
        """
        # get mixture of generated and ground truth data
        #shape_dat = (gen_data - real_data).shape
        #alpha = tf.random.normal([self.hparams["batch_size"], 1, 1, 1], 0., 1.)
        alpha = tf.random.uniform(shape=tf.shape(real_data), minval=0., maxval=1.)
        mix_data = real_data + alpha * (gen_data - real_data)
        critic_in_gen = [cond_data] + [const_data] + [mix_data]

        # cleaner implementation
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(mix_data)
            mixed_score = self.critic.model(critic_in_gen)

        # Compute gradients with respect to mixed data
        grads_mix = gp_tape.gradient(mixed_score, mix_data)

        #with tf.GradientTape() as gp_tape:
            #gp_tape.watch(critic_in_gen[-1])
        #    gp_tape.watch(critic_in_gen)
        #    critic_mix = self.critic.model(critic_in_gen, training=True)
        #
        ## calculate the gradient on the mixture data...
        #grads_mix = gp_tape.gradient(critic_mix, [critic_in_gen])[0]
        # ... and norm it
        #norm = tf.norm(grads_mix, ord=2, axis=list(range(1, len(grads_mix.shape))))
        norm = tf.sqrt(tf.reduce_mean(tf.square(grads_mix), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.) ** 2)

        return gp
    

class Harris_WGAN(Sha_WGAN):
    
    def __init__(self, generator: AbstractModelClass, critic: AbstractModelClass, shape_in: List, hparams: dict,
                 varnames_tar: List, savedir: str, expname: str, with_horovod: bool = False):
        """
        Initialize the Harris_WGAN model class.
        !!! Note: Inherits from Sha_WGAN and overwrites methods when necessary only !!!

        :param generator: The generator model.
        :param critic: The critic model.
        :param shape_in: The input shape of the model. Note: The last two dimensions must denote the number of coarse-grained predictors 
                         and the number of static high-resolution predictors, respectively.
        :param hparams: Dictionary of custom hyperparameters.
        :param varnames_tar: List of target variable names.
        :param savedir: Drectory to save the model.
        :param expname: The name of the experiment.
        :param with_horovod: Whether to use Horovod for distributed training.
        """        
        if not shape_in:                    # shape_in can be None when loading model for inference -> set dummy-value to allow model construction
            shape_in = [1, 1, 1, 1]
        super().__init__(shape_in, hparams, varnames_tar, savedir, expname)

        self.modelname = "harriswgan"
        self.with_horovod = with_horovod
        self.main_process = True 
        if self.with_horovod:
            self.main_process = hvd.rank() == 0
        
        # set hyperparmaters
        self.set_hparams(hparams)
        if not self.hparams["noise_input"]:
            print("Set noise channels to 1.")
            self.hparams["noise_channels"] = 1
        # set submodels
        self.generator, self.critic = self.set_model(generator, critic)
        # set compile and fit options as well as custom objects
        self.set_compile_options()
        self.set_custom_objects(loss=self.compile_options['loss'])
        self.set_fit_options()
        
    def set_compile_options(self):
        """
        Set compile options for the HarrisWGAN model.
        """
        # set optimizers
        if self.hparams["optimizer"].lower() == "adam":
            optimizer = keras.optimizers.Adam
            kwargs_opt = {"beta_1": 0.0, "beta_2": 0.9}
        elif self.hparams["optimizer"].lower() == "rmsprop":
            optimizer = keras.optimizers.RMSprop
            kwargs_opt = {}
        else:
            raise ValueError("'{0}' is not a valid optimizer. Either choose Adam or RMSprop-optimizer")

        self.optimizer = (optimizer(self.critic.hparams["lr"], **kwargs_opt), optimizer(self.generator.hparams["lr"], **kwargs_opt))
        
        # wrap optimizers for distributed training
        #if self.with_horovod:
        #    import horovod.tensorflow as hvd
        #    # wrap optimizers for distributed training
        #    self.optimizer = tuple(hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)
        #                          for opt in self.optimizer)
        
    def get_fit_options(self):
        """
        Get options that will be parsed to the fit-method of the Keras model.
        """
        harriswgan_callbacks = []
        
        if self.hparams["lr_decay"]:
            harriswgan_callbacks.append(LearningRateSchedulerHarrisWGAN(self.get_lr_decay(), verbose=1))
        
        if self.hparams["lcheckpointing"] and self.main_process:            
            harriswgan_callbacks.append(ModelCheckpointHarrisWGAN(self._savedir, self._expname, 
                                                                  monitor="val_recon_loss", verbose=1, save_best_only=False, mode="min"))
            
        if self.hparams["learlystopping"]:
            harriswgan_callbacks.append(EarlyStopping(monitor="val_recon_loss", patience=8))

        if self.with_horovod:            
            harriswgan_callbacks.append(BroadcastWeightsCallback(self.generator, self.critic, self.optimizer[0], self.optimizer[1]))
            #harriswgan_callbacks.append(hvd_callbacks.MetricAverageCallback())

            
        if harriswgan_callbacks is not None:
            return {"callbacks": harriswgan_callbacks}
        else:
            return {}  
        
    def set_model(self, generator, critic):
        """
        Instantiate the generator and critic models and create the HarrisWGAN model instance.
        :param generator: The generator model.
        :param critic: The critic model.
        """
        # get relevant shapes for input and output of generator and critic
        lo_res_in_shp = list(self._input_shape[:-1] )
        hi_res_in_shp = list(np.array(lo_res_in_shp[:2])*int(np.prod(np.array([4,])))) + [self._input_shape[-1]]
        in_noise_shp = lo_res_in_shp[:2] + [self.hparams["noise_channels"]]    
        out_shp = hi_res_in_shp[:2] + [len(self._varnames_tar)]            
        
        shp_gen = {"lo_res_inputs": lo_res_in_shp, "hi_res_inputs": hi_res_in_shp, "noise_input": in_noise_shp}
        # get generator model
        gen_model = generator(shp_gen, self.hparams["hparams_generator"], self._varnames_tar)      
        
        # get critic model
        shp_critic = {"lo_res_inputs": lo_res_in_shp, "hi_res_inputs": hi_res_in_shp, "output": out_shp}
        critc_model = critic(shp_critic, self.hparams["hparams_critic"], self._varnames_tar)
        
        # get hyperparamters of HarrisWGAN only
        hparams_wgan_only = self.hparams.copy()
        hparams_wgan_only.pop("hparams_critic")
        hparams_wgan_only.pop("hparams_generator")
                
        # ...and create Harris_WGAN model instance
        self.model = Harris_WGAN_Model(gen_model, critc_model, hparams_wgan_only, self._expname)

        return gen_model, critc_model

        
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
                return min(lr, lr_end)

        return lr_scheduler

    def plot_model(self, save_dir, **kwargs):
        """
        Plot generator and critci model separately.
        :param save_dir: directory under which plots will be saved
        :param kwargs: All keyword arguments valid for tf.keras.utils.plot_model
        
        NOTE SL: taken from wgan_model.py
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
        
        # construct noise genartor required for ensemble 
        noise_gen = NoiseGenerator(list(generator.get_layer(name='noise_input').input_shape[0][1:]), self.hparams["batch_size"])

        hparams_wgan_only = self.hparams.copy()
        hparams_wgan_only.pop("hparams_critic")
        hparams_wgan_only.pop("hparams_generator")
        
        # get construct model for inference exposing predict-method
        # Note the predict-step makes use of the generator only. Thus, the critic model is not needed here
        wgan_model = Harris_WGAN_Model(generator, None, hparams_wgan_only, expname, noise_gen=noise_gen)
        
        return wgan_model

    # !!! NOTE !!!
    # count_parameters is inherited from Sha_WGAN
    # !!! NOTE !!!
    
    def set_hparams_default(self):
        """
        Note: Hyperparameter defaults taken from 1) https://github.com/ECMWFCode4Earth/tesserugged/blob/master/dev/gan/dsrnngan/local_config.yaml and 2) https://github.com/ECMWFCode4Earth/tesserugged/blob/master/dev/gan/dsrnngan/models.py
        """
        self.hparams_default = {"batch_size": 2, "nepochs": 30, "lr_decay": False, "decay_start": 3, "decay_end": 20, "stream_mode": "lo_input",
                                "l_embed": False, "ds_steps": [4,], "d_steps": 5, "recon_weight": 1000., "gp_weight": 10., "optimizer": "adam", 
                                "lcheckpointing": True, "learlystopping": False, "recon_loss": "ensmeanMSE", "ensemble_size": 8, "noise_input": True,
                                "noise_channels": 4, "hparams_generator": {}, "hparams_critic": {} }


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


class LearningRateSchedulerHarrisWGAN(LearningRateSchedulerWGAN):
    """Note SL: taken from wgan_model.py"""
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


class ModelCheckpointHarrisWGAN(ModelCheckpoint):
    """Note SL: taken from wgan_model.py"""
    def __init__(self, filepath, expname, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                 mode='auto', save_freq="epoch", options=None, **kwargs):
        super(ModelCheckpointHarrisWGAN, self).__init__(filepath,  monitor, verbose, save_best_only,
                                                  save_weights_only, mode, save_freq, options=options, **kwargs)
        self._expname = expname

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
            if self.save_best_only:
                add_str = "_best"
            else:
                add_str = f"_epoch{epoch+1:05d}"
            
            filepath = Path(filepath).joinpath(f"{self._expname}{add_str}")
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
                            self.model.save(filepath, overwrite=True, include_optimizer=not self.save_weights_only, save_format="tf", 
                                            suffix=add_str)#, options=self._options)
                            # ML E
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    # ML S
                    self.model.save(filepath, overwrite=True, include_optimizer=not self.save_weights_only, save_format="tf", 
                                    suffix=add_str)#, options=self._options)
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

####################################################################################
####################################################################################
# some classes and methods that are used in the original Harris GAN implementation
####################################################################################
####################################################################################
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (
            s[0],
            None if s[1] is None else s[1]+2*self.padding[0],
            None if s[2] is None else s[2]+2*self.padding[1],
            s[3]
        )

    def call(self, x):
        i_pad, j_pad = self.padding
        return tf.pad(x, [[0, 0], [i_pad, i_pad], [j_pad, j_pad], [0, 0]], 'REFLECT')


class SymmetricPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (
            s[0],
            None if s[1] is None else s[1]+2*self.padding[0],
            None if s[2] is None else s[2]+2*self.padding[1],
            s[3]
        )

    def call(self, x):
        i_pad, j_pad = self.padding
        return tf.pad(x, [[0, 0], [i_pad, i_pad], [j_pad, j_pad], [0, 0]], 'SYMMETRIC')

@keras.utils.register_keras_serializable(package="Custom", name="Conv2DPadding_2") 
class Conv2DPadding(Layer):
    def __init__(self, filters, kernel_size, stride, padding, dilations, **kwargs):
        super(Conv2DPadding, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilations = dilations
        if not isinstance(dilations, int):
            # padding calculation in build() would need to be adjusted to handle a tuple/list
            raise NotImplementedError("Only integer dilation is supported.")
        if padding is None:
            raise ValueError("padding should not be None")

    def build(self, x):
        if self.padding in ('reflect', 'symmetric'):
            pad = tuple((self.dilations*(s-1))//2 for s in self.kernel_size)  # only works if s is odd, or dilation is even
            if self.padding == 'reflect':
                self.padref = ReflectionPadding2D(padding=pad)
            elif self.padding == 'symmetric':
                self.symref = SymmetricPadding2D(padding=pad)
            self.convval = Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  strides=(self.stride, self.stride),
                                  padding='valid',
                                  dilation_rate=self.dilations)
        else:
            self.convsam = Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  strides=(self.stride, self.stride),
                                  padding='same',
                                  dilation_rate=self.dilations)

    def call(self, x):
        if self.padding in ('reflect', 'symmetric'):
            if self.padding == 'reflect':
                x = self.padref(x)
            elif self.padding == 'symmetric':
                x = self.symref(x)
            return self.convval(x)
        else:  # same
            return self.convsam(x)
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters,
                  "kernel_size": self.kernel_size,
                  "stride": self.stride, 
                  "padding": self.padding, 
                  "dilations": self.dilations})
        
        return config
        
def residual_block(x, filters, conv_size=(3, 3), stride=1, dilations=1, relu_alpha=0.2, padding=None, norm=None, force_conv: bool = False):
    in_channels = int(x.shape[-1])
    x_in = x

    if stride > 1:
        x_in = AveragePooling2D(pool_size=(stride, stride))(x_in)
    if (filters != in_channels) or force_conv:
        x_in = Conv2D(filters=filters, kernel_size=(1, 1))(x_in)

    # first block of activation and 3x3 convolution (possibly strided, although we don't use this)
    x = LeakyReLU(relu_alpha)(x)
    x = Conv2DPadding(filters=filters, kernel_size=conv_size, stride=stride, dilations=dilations, padding=padding)(x)
    if norm == "batch":
        x = BatchNormalization()(x)
    elif norm == "layer":
        x = LayerNormalization()(x)
    elif norm is None or norm == "":
        pass
    else:
        raise ValueError(f"norm type {norm} not implemented")

    # second block of activation and 3x3 unstrided convolution
    x = LeakyReLU(relu_alpha)(x)
    x = Conv2DPadding(filters=filters, kernel_size=conv_size, stride=1, dilations=dilations, padding=padding)(x)
    if norm == "batch":
        x = BatchNormalization()(x)
    elif norm == "layer":
        x = LayerNormalization()(x)
    elif norm is None or norm == "":
        pass
    else:
        raise ValueError(f"norm type {norm} not implemented")
    
    # skip connection
    x = Add()([x, x_in])

    return x


def const_upscale_block(const_input, steps, filters):
    # Map (N x kH x kW x C) to (N x H x W x f), where k is downscaling factor
    const_output = const_input
    for step in steps:
        const_output = Conv2D(filters=filters, kernel_size=(step, step), strides=step, padding="valid", activation="relu")(const_output)
    return const_output

