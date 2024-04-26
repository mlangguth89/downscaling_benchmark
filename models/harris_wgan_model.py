# Harris et al 2022, WGAN model implementation
"""
Class for Harris et al 2022, conditional Wasserstein GAN model (CWGAN)
"""
import os
from typing import List, Tuple, Union
import inspect
import numpy as np
import h5py
from abstract_model_class import AbstractModelClass
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, concatenate, LeakyReLU, UpSampling2D, Layer, BatchNormalization, Conv2D, Add, AveragePooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model as k_plot_model
from custom_losses import get_custom_loss
from wgan_model import LearningRateSchedulerWGAN

list_or_tuple = Union[List, Tuple]

class GeneratorHarris(AbstractModelClass):
    # content based on original generator function from models.py (Harris repo)
    # structure based on Critic_Simple from wgan_model.py
    def __init__(self, shape_in: List, hparams: dict, varnames_tar: List):
        super().__init__(shape_in, hparams, varnames_tar, "", "")       # Pass empty savedir- and expname-arguments since this is not a stand-alone model
        
        # TODO: incorporate them accordingly
        old_input_kwargs = dict(
            downscaling_steps=5,
            input_channels=9,
            latent_variables=1,
            noise_channels=8,
            filters_gen=64,
            constant_fields=2,
            conv_size=(3, 3),
            padding=None,
            relu_alpha=0.2,
            norm=None,
        )
        # set submodels
        self.set_hparams(hparams)
        self.set_model()
        
    def set_model(self):
        # Network inputs
        # low resolution condition
        generator_input = Input(shape=(None, None, input_channels), name="lo_res_inputs")
        print(f"generator_input shape: {generator_input.shape}")
        # constant fields
        const_input = Input(shape=(None, None, constant_fields), name="hi_res_inputs")
        print(f"constants_input shape: {const_input.shape}")

        # Convolve constant fields down to match other input dimensions
        upscaled_const_input = const_upscale_block(
            const_input, steps=downscaling_steps, filters=filters_gen
        )
        print(f"upscaled constants shape: {upscaled_const_input.shape}")
        # noise
        noise_input = Input(shape=(None, None, noise_channels), name="noise_input")
        print(f"noise_input shape: {noise_input.shape}")
        # Concatenate all inputs together
        generator_output = concatenate(
            [generator_input, upscaled_const_input, noise_input]
        )
        print(f"Shape after first concatenate: {generator_output.shape}")

        # Pass through 3 residual blocks
        for ii in range(3):
            generator_output = residual_block(
                generator_output,
                filters=filters_gen,
                conv_size=conv_size,
                stride=1,
                relu_alpha=relu_alpha,
                norm=norm,
                padding=padding,
            )
        print("End of first residual block")
        print(f"Shape after first residual block: {generator_output.shape}")
        # Upsampling from low-res to high-res with alternating residual blocks
        # In the paper, this was [2*filters_gen, filters_gen] for steps of 5 and 2
        block_channels = [2 * filters_gen] * (len(downscaling_steps) - 1) + [filters_gen]
        for ii, step in enumerate(downscaling_steps):
            generator_output = UpSampling2D(size=(step, step), interpolation="bilinear")(
                generator_output
            )
            print(f"Shape after upsampling step {ii+1}: {generator_output.shape}")
            generator_output = residual_block(
                generator_output,
                filters=block_channels[ii],
                conv_size=conv_size,
                stride=1,
                relu_alpha=relu_alpha,
                norm=norm,
                padding=padding,
            )
            print(f"Shape after residual block: {generator_output.shape}")

        # Concatenate with original size constants field
        generator_output = concatenate([generator_output, const_input])
        print(f"Shape after second concatenate: {generator_output.shape}")

        # Pass through 3 residual blocks
        for ii in range(3):
            generator_output = residual_block(
                generator_output,
                filters=filters_gen,
                conv_size=conv_size,
                stride=1,
                relu_alpha=relu_alpha,
                norm=norm,
                padding=padding,
            )
        print(f"Shape after third residual block: {generator_output.shape}")

        # Output layer
        generator_output = Conv2D(
            filters=1, kernel_size=(1, 1), activation="softplus", name="output"
        )(generator_output)
        print(f"Output shape: {generator_output.shape}")

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
        self.hparams_default = {"num_conv": 4, "channels_start": 64, "activation": "swish",
                                "lbatch_norm": True, "kernel": (3, 3), "stride": (2, 2), "lr": 1.e-06,}


class DiscriminatorHarris(AbstractModelClass):
    # content based on original discriminator function from models.py (Harris repo)
    # structure based on Critic_Simple from wgan_model.py
    def __init__(self, shape_in: List, hparams: dict, varnames_tar: List):
        super().__init__(shape_in, hparams, varnames_tar, "", "")       # Pass empty savedir- and expname-arguments since this is not a stand-alone model
        
        # TODO: incorporate them accordingly
        old_input_kwargs = dict(
            downscaling_steps=5,
            input_channels=9,
            noise_channels=8,
            constant_fields=2,
            filters_disc=64,
            conv_size=(3, 3),
            padding=None,
            stride=1,
            relu_alpha=0.2,
            norm=None,
        )
        
        # set submodels
        self.set_hparams(hparams)
        self.set_model()
        
    def set_model(self):
        # Network inputs
        # low resolution condition
        generator_input = Input(shape=(None, None, input_channels), name="lo_res_inputs")
        print(f"generator_input shape: {generator_input.shape}")
        # constant fields
        const_input = Input(shape=(None, None, constant_fields), name="hi_res_inputs")
        print(f"constants_input shape: {const_input.shape}")
        # target image
        generator_output = Input(shape=(None, None, 1), name="output")
        print(f"generator_output shape: {generator_output.shape}")

        # convolve down constant fields to match ERA
        lo_res_const_input = const_upscale_block(
            const_input, steps=downscaling_steps, filters=filters_disc
        )
        print(f"upscaled constants shape: {lo_res_const_input.shape}")

        # concatenate constants to lo-res input
        lo_res_input = concatenate([generator_input, lo_res_const_input])
        print(f"Shape after lo-res concatenate: {lo_res_input.shape}")

        # concatenate constants to hi-res input
        hi_res_input = concatenate([generator_output, const_input])
        print(f"Shape after hi-res concatenate: {hi_res_input.shape}")

        # encode inputs using residual blocks
        # In the paper, this was [filters_disc, 2*filters_disc] for steps of 5 and 2
        block_channels = [filters_disc] * (len(downscaling_steps) - 1) + [2 * filters_disc]

        for ii, step in enumerate(downscaling_steps):
            lo_res_input = residual_block(
                lo_res_input,
                filters=block_channels[ii],
                conv_size=conv_size,
                stride=1,
                relu_alpha=relu_alpha,
                norm=norm,
                padding=padding,
            )
            print(f"Shape of lo-res input after residual block: {lo_res_input.shape}")
            hi_res_input = Conv2D(
                filters=block_channels[ii],
                kernel_size=(step, step),
                strides=step,
                padding="valid",
                activation="relu",
            )(hi_res_input)
            print(
                f"Shape of hi_res_input after upsampling step {ii+1}: {hi_res_input.shape}"
            )
            hi_res_input = residual_block(
                hi_res_input,
                filters=block_channels[ii],
                conv_size=conv_size,
                stride=1,
                relu_alpha=relu_alpha,
                norm=norm,
                padding=padding,
            )
            print(f"Shape of hi-res input after residual block: {hi_res_input.shape}")

        # concatenate hi- and lo-res inputs channel-wise before passing through discriminator
        disc_input = concatenate([lo_res_input, hi_res_input])
        print(
            f"Shape after concatenating lo-res input and hi-res input: {disc_input.shape}"
        )

        # encode in residual blocks
        disc_input = residual_block(
            disc_input,
            filters=filters_disc,
            conv_size=conv_size,
            stride=1,
            relu_alpha=relu_alpha,
            norm=norm,
            padding=padding,
        )
        print(f"Shape after residual block: {disc_input.shape}")
        print("End of second residual block")

        # discriminator output
        disc_output = GlobalAveragePooling2D()(disc_input)
        print(f"discriminator output shape after pooling: {disc_output.shape}")
        disc_output = Dense(64, activation="relu")(disc_output)
        print(f"discriminator output shape: {disc_output.shape}")
        disc_output = Dense(1, name="disc_output")(disc_output)
        print(f"discriminator output shape: {disc_output.shape}")

        self.model = Model(
            inputs=[generator_input, const_input, generator_output],
            outputs=disc_output,
            name="disc",
        )
                   
    def set_compile_options(self):
        raise RuntimeError(f"Critic model is supposed to be part of a composite model such as WGAN, but not as standalone model for training.")
        
    def set_fit_options(self):
        raise RuntimeError(f"Critic model is supposed to be part of a composite model such as WGAN, but not as standalone model for training.")
        
    def set_hparams_default(self):
        """
        Note: hyperparameter defaults of generator and critic model must be set in the respective model classes whose instances are just parsed here.
        """
        self.hparams_default = {"num_conv": 4, "channels_start": 64, "activation": "swish",
                                "lbatch_norm": True, "kernel": (3, 3), "stride": (2, 2), "lr": 1.e-06,}





    
# turn this into a set_model method for HarrisWGAN
def setup_model(*,
    # mode=None, == "GAN"
    # arch=None, == "forceconv"
    downscaling_steps=None, # steps: [5,]  # list of integers that multiply to the downscaling factor; the generator will use UpSampling2D layers of these sizes, alternating with residual blocks.
    input_channels=None, # number of predictors
    filters_gen=128,  # 128   # generator network width
    filters_disc=512,  # 512   # discriminator network width
    noise_channels=4,  # 4
    # latent_variables=None, # only in VAEGAN
    padding="reflect", # "reflect"  # convolution padding: 'same', 'reflect', or 'symmetric'
    # kl_weight=None, # kl_weight: 1e-8  # used for VAEGAN
    ensemble_size=None, # ensemble_size: 8  # size of ensemble for content loss; use null to turn off
    CLtype="ensmeanMSE",  # CL_type: "ensmeanMSE"  # type of content loss to use: 'CRPS', 'CRPS_phys', 'ensmeanMSE', 'ensmeanMSE_phys'
    content_loss_weight=1000,  # content_loss_weight: 1000.0  # we used 1000 for ensmeanMSE, and 100 for CRPS
    lr_disc=1e-5,  # learning_rate_disc: 1e-5  # if training blows up, decrease this
    lr_gen=1e-5
):   # learning_rate_gen: 1e-5  # if training blows up, decrease this

    gen = generator(
        downscaling_steps=downscaling_steps,
        input_channels=input_channels,
        noise_channels=noise_channels,
        filters_gen=filters_gen,
        padding=padding,
    )
    disc = discriminator(
       downscaling_steps=downscaling_steps,
       input_channels=input_channels,
       filters_disc=filters_disc,
       padding=padding
    )
    model = WGANGP(
        gen,
        disc,
        # mode,
        lr_disc=lr_disc,
        lr_gen=lr_gen,
        ensemble_size=ensemble_size,
        CLtype=CLtype,
        content_loss_weight=content_loss_weight
    )
    return model

model = setup_model(
    downscaling_steps=[5,],
    input_channels=input_channels,  # len(all_fcst_fields); num of predictors
    filters_gen=128,
    filters_disc=512,
    noise_channels=4,
    padding="reflect",
    lr_disc=1e-5,
    lr_gen=1e-5,
    # kl_weight=kl_weight,  # only VAEGAN
    ensemble_size=8,
    CLtype="ensmeanMSE",
    content_loss_weight=1000,
)



# to be deprecated
class Nontrainable(object):

    def __init__(self, models):
        if not isinstance(models, list):
            models = [models]
        self.models = models

    def __enter__(self):
        self.trainable_status = [m.trainable for m in self.models]
        for m in self.models:
            m.trainable = False
        return self.models

    def __exit__(self, type, value, traceback):
        for (m, t) in zip(self.models, self.trainable_status):
            m.trainable = t


def save_opt_weights(model, filepath):
    with h5py.File(filepath, 'w') as f:
        # Save optimizer weights.
        symbolic_weights = getattr(model.optimizer, 'weights')
        if symbolic_weights:
            optimizer_weights_group = f.create_group('optimizer_weights')
            weight_values = K.batch_get_value(symbolic_weights)
            weight_names = []
            for i, (w, val) in enumerate(zip(symbolic_weights,
                                             weight_values)):
                if hasattr(w, 'name') and w.name:
                    name = str(w.name)
                else:
                    name = 'param_' + str(i)
                weight_names.append(name.encode('utf8'))
            optimizer_weights_group.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_values):
                param_dset = optimizer_weights_group.create_dataset(
                    name,
                    val.shape,
                    dtype=val.dtype)
                if not val.shape:
                    # scalar
                    param_dset[()] = val
                else:
                    param_dset[:] = val


def load_opt_weights(model, filepath):
    with h5py.File(filepath, mode='r') as f:
        optimizer_weights_group = f['optimizer_weights']  # h5py group
        optimizer_weight_names = optimizer_weights_group.attrs['weight_names']
        for name in optimizer_weight_names:
            optimizer_weight_values = optimizer_weights_group[name]
        model.optimizer.set_weights(optimizer_weight_values)


def ensure_list(x):
    if type(x) != list:
        x = [x]
    return x


def input_shapes(model, prefix):
    shapes = [il.shape[1:] for il in
              model.inputs if il.name.startswith(prefix)]
    shapes = [tuple([d for d in dims]) for dims in shapes]
    return shapes


class WGANGP(object):

    def __init__(self, gen, disc, gradient_penalty_weight=10,
                 lr_disc=0.0001, lr_gen=0.0001, avg_seed=None,
                 kl_weight=None, ensemble_size=None, CLtype=None,
                 content_loss_weight=None):

        self.gen = gen
        self.disc = disc
        # self.mode = mode
        self.mode = "GAN"
        self.gradient_penalty_weight = gradient_penalty_weight
        self.learning_rate_disc = lr_disc
        self.learning_rate_gen = lr_gen
        self.kl_weight = kl_weight
        self.ensemble_size = ensemble_size
        self.CLtype = CLtype
        self.content_loss_weight = content_loss_weight
        self.build_wgan_gp()

    def filenames_from_root(self, root):
        fn = {
            "gen_weights": root+"-gen_weights.h5",
            "disc_weights": root+"-disc_weights.h5",
            "gen_opt_weights": root+"-gen_opt_weights.h5",
            "disc_opt_weights": root+"-disc_opt_weights.h5"
        }
        return fn

    def load(self, load_files):
        self.gen.load_weights(load_files["gen_weights"])
        self.disc.load_weights(load_files["disc_weights"])

        with Nontrainable(self.disc):
            self.gen_trainer.make_train_function()
            load_opt_weights(self.gen_trainer,
                             load_files["gen_opt_weights"])
        with Nontrainable(self.gen):
            self.disc_trainer.make_train_function()
            load_opt_weights(self.disc_trainer,
                             load_files["disc_opt_weights"])

    def save(self, save_fn_root):
        paths = self.filenames_from_root(save_fn_root)
        self.gen.save_weights(paths["gen_weights"], overwrite=True)
        self.disc.save_weights(paths["disc_weights"], overwrite=True)
        save_opt_weights(self.disc_trainer, paths["disc_opt_weights"])
        save_opt_weights(self.gen_trainer, paths["gen_opt_weights"])

    def build_wgan_gp(self):

        # find shapes for inputs
        if self.mode == 'GAN':
            cond_shapes = input_shapes(self.gen, "lo_res_inputs")
            const_shapes = input_shapes(self.gen, "hi_res_inputs")
            noise_shapes = input_shapes(self.gen, "noise_input")
        elif self.mode == 'VAEGAN':
            cond_shapes = input_shapes(self.gen.encoder, "lo_res_inputs")
            const_shapes = input_shapes(self.gen.encoder, "hi_res_inputs")
            noise_shapes = input_shapes(self.gen.decoder, "noise_input")
        sample_shapes = input_shapes(self.disc, "output")

        # Create generator training network
        with Nontrainable(self.disc):
            if self.mode == 'GAN':
                cond_in = [Input(shape=cond_shapes[0])]
                const_in = [Input(shape=const_shapes[0])]

                if self.ensemble_size is None:
                    noise_in = [Input(shape=noise_shapes[0])]
                else:
                    noise_in = [Input(shape=noise_shapes[0])
                                for ii in range(self.ensemble_size + 1)]
                gen_in = cond_in + const_in + noise_in

                gen_out = self.gen(gen_in[0:3])  # only use cond/const/noise
                gen_out = ensure_list(gen_out)
                disc_in_gen = cond_in + const_in + gen_out
                disc_out_gen = self.disc(disc_in_gen)
                full_gen_out = [disc_out_gen]
                if self.ensemble_size is not None:
                    # generate ensemble of predictions and add mean to gen_trainer output
                    preds = [self.gen([gen_in[0], gen_in[1], gen_in[3+ii]])
                             for ii in range(self.ensemble_size)]
                    preds = tf.stack(preds)
                    full_gen_out.append(preds)
                self.gen_trainer = Model(inputs=gen_in,
                                         outputs=full_gen_out,
                                         name='gen_trainer')
            elif self.mode == 'VAEGAN':
                self.gen_trainer = VAE_trainer(self.gen, self.disc,
                                               self.kl_weight,
                                               self.ensemble_size,
                                               self.CLtype,
                                               self.content_loss_weight)

        # Create discriminator training network
        with Nontrainable(self.gen):
            cond_in = [Input(shape=s, name='lo_res_inputs') for s in cond_shapes]
            const_in = [Input(shape=s, name='hi_res_inputs') for s in const_shapes]
            noise_in = [Input(shape=s, name='noise_input') for s in noise_shapes]
            sample_in = [Input(shape=s, name='output') for s in sample_shapes]
            gen_in = cond_in + const_in + noise_in
            disc_in_real = sample_in[0]
            if self.mode == 'GAN':
                disc_in_fake = self.gen(gen_in)
            elif self.mode == 'VAEGAN':
                encoder_in = cond_in + const_in
                encoder_mean, encoder_log_var = self.gen.encoder(encoder_in)
                decoder_in = [encoder_mean, encoder_log_var, noise_in, const_in]
                disc_in_fake = self.gen.decoder(decoder_in)
            disc_in_avg = RandomWeightedAverage()([disc_in_real, disc_in_fake])
            disc_out_real = self.disc(cond_in + const_in + [disc_in_real])
            disc_out_fake = self.disc(cond_in + const_in + [disc_in_fake])
            disc_out_avg = self.disc(cond_in + const_in + [disc_in_avg])
            disc_gp = GradientPenalty()([disc_out_avg, disc_in_avg])
            self.disc_trainer = Model(inputs=cond_in + const_in + noise_in + sample_in,
                                      outputs=[disc_out_real, disc_out_fake, disc_gp],
                                      name='disc_trainer')

        self.compile()

    def compile(self, opt_disc=None, opt_gen=None):
        # create optimizers
        if opt_disc is None:
            opt_disc = Adam(learning_rate=self.learning_rate_disc, beta_1=0.5, beta_2=0.9)
        self.opt_disc = opt_disc
        if opt_gen is None:
            opt_gen = Adam(learning_rate=self.learning_rate_gen, beta_1=0.5, beta_2=0.9)
        self.opt_gen = opt_gen

        with Nontrainable(self.disc):
            if self.mode == 'GAN':
                if self.ensemble_size is not None:
                    CLfn = CL_chooser(self.CLtype)
                    losses = [wasserstein_loss, CLfn]
                    loss_weights = [1.0, self.content_loss_weight]
                else:
                    losses = [wasserstein_loss]
                    loss_weights = [1.0]
                self.gen_trainer.compile(loss=losses,
                                         loss_weights=loss_weights,
                                         optimizer=self.opt_gen)
            elif self.mode == 'VAEGAN':
                self.gen_trainer.compile(optimizer=self.opt_gen)
        with Nontrainable(self.gen):
            self.disc_trainer.compile(
                loss=[wasserstein_loss, wasserstein_loss, 'mse'],
                loss_weights=[1.0, 1.0, self.gradient_penalty_weight],
                optimizer=self.opt_disc
            )
            self.disc_trainer.summary()
    
    # needs to be rewritten into the train_step method for the WGAN class
    def train(self, batch_gen, noise_gen, num_gen_batches=1,
              training_ratio=1, show_progress=True):

        disc_target_real = None
        for inputs, _ in batch_gen.take(1).as_numpy_iterator():
            tmp_batch = inputs["lo_res_inputs"]
            batch_size = tmp_batch.shape[0]
        del tmp_batch
        del inputs
        if show_progress:
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(num_gen_batches*batch_size)
        disc_target_real = np.ones((batch_size, 1), dtype=np.float32)
        disc_target_fake = -disc_target_real
        gen_target = disc_target_real
        target_gp = np.zeros((batch_size, 1), dtype=np.float32)
        disc_target = [disc_target_real, disc_target_fake, target_gp]

        batch_gen_iter = iter(batch_gen)

        if self.mode == 'VAEGAN':
            for tracker in self.gen_trainer.metrics:
                tracker.reset_states()

        for kk in range(num_gen_batches):

            # train discriminator
            disc_loss = None
            disc_loss_n = 0
            for rep in range(training_ratio):
                # generate some real samples
                inputs, outputs = batch_gen_iter.get_next()
                cond = inputs["lo_res_inputs"]
                const = inputs["hi_res_inputs"]
                sample = outputs["output"]

                with Nontrainable(self.gen):
                    dl = self.disc_trainer.train_on_batch(
                        [cond, const, noise_gen(), sample], disc_target)

                if disc_loss is None:
                    disc_loss = np.array(dl)
                else:
                    disc_loss += np.array(dl)
                disc_loss_n += 1

                del sample, cond, const

            disc_loss /= disc_loss_n

            with Nontrainable(self.disc):
                inputs, outputs = batch_gen_iter.get_next()
                cond = inputs["lo_res_inputs"]
                const = inputs["hi_res_inputs"]
                sample = outputs["output"]

                condconst = [cond, const]
                if self.ensemble_size is None:
                    gt_outputs = [gen_target]
                    noise_list = [noise_gen()]
                else:
                    noise_list = [noise_gen()
                                  for ii in range(self.ensemble_size + 1)]
                    gt_outputs = [gen_target, sample]
                gt_inputs = condconst + noise_list

                if self.mode == 'GAN':
                    gen_loss = self.gen_trainer.train_on_batch(
                        gt_inputs, gt_outputs)
                elif self.mode == 'VAEGAN':
                    gen_loss = self.gen_trainer.train_step(
                        [gt_inputs, gt_outputs])

                gen_loss = ensure_list(gen_loss)
                del sample, cond, const

            if show_progress:
                losses = []
                for ii, dl in enumerate(disc_loss):
                    losses.append((f"D{ii}", dl))
                for ii, gl in enumerate(gen_loss):
                    losses.append((f"G{ii}", gl))
                progbar.add(batch_size,
                            values=losses)

            loss_log = {}
            if self.mode == "det":
                raise RuntimeError("Doctor, what are you doing here? You're supposed to be on Gallifrey")
            elif self.mode == "GAN":
                loss_log["disc_loss"] = disc_loss[0]
                loss_log["disc_loss_real"] = disc_loss[1]
                loss_log["disc_loss_fake"] = disc_loss[2]
                loss_log["disc_loss_gp"] = disc_loss[3]
                loss_log["gen_loss_total"] = gen_loss[0]
                if self.ensemble_size is not None:
                    loss_log["gen_loss_disc"] = gen_loss[1]
                    loss_log["gen_loss_ct"] = gen_loss[2]
            elif self.mode == "VAEGAN":
                loss_log["disc_loss"] = disc_loss[0]
                loss_log["disc_loss_real"] = disc_loss[1]
                loss_log["disc_loss_fake"] = disc_loss[2]
                loss_log["disc_loss_gp"] = disc_loss[3]
                loss_log["gen_loss_total"] = gen_loss[0].numpy()
                loss_log["gen_loss_disc"] = gen_loss[1].numpy()
                loss_log["gen_loss_kl"] = gen_loss[2].numpy()
                if self.ensemble_size is not None:
                    loss_log["gen_loss_ct"] = gen_loss[3].numpy()
            gc.collect()

        return loss_log
    

    
    
    
    
    
    
    
    
    
from wgan_model import WGAN_Model

# TODO: rename cwgan to wgan_harris
class HarrisWGAN_Model(WGAN_Model):
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
        



class HarrisWGAN(AbstractModelClass):
    """tbd if this can instead inherit of WGAN"""
    
    def __init__(self, generator: AbstractModelClass, critic: AbstractModelClass, shape_in: List, hparams: dict, varnames_tar: List, savedir: str, expname: str):
        
        super().__init__(shape_in, hparams, varnames_tar, savedir, expname)

        self.modelname = "harriswgan"
        
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
        # correct number of dynamic predictorst
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
    
    

####################################################################################
####################################################################################
# some classes and function  that are used in the original harris gan implementation
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

class Conv2DPadding(Layer):
    def __init__(self, filters, kernel_size, stride, padding, dilations):
        super(Conv2DPadding, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilations
        if not isinstance(dilations, int):
            # padding calculation in build() would need to be adjusted to handle a tuple/list
            raise NotImplementedError("Only integer dilation is supported.")
        if padding is None:
            raise ValueError("padding should not be None")

    def build(self, x):
        if self.padding in ('reflect', 'symmetric'):
            pad = tuple((self.dilation*(s-1))//2 for s in self.kernel_size)  # only works if s is odd, or dilation is even
            if self.padding == 'reflect':
                self.padref = ReflectionPadding2D(padding=pad)
            elif self.padding == 'symmetric':
                self.symref = SymmetricPadding2D(padding=pad)
            self.convval = Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  strides=(self.stride, self.stride),
                                  padding='valid',
                                  dilation_rate=self.dilation)
        else:
            self.convsam = Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  strides=(self.stride, self.stride),
                                  padding='same',
                                  dilation_rate=self.dilation)

    def call(self, x):
        if self.padding in ('reflect', 'symmetric'):
            if self.padding == 'reflect':
                x = self.padref(x)
            elif self.padding == 'symmetric':
                x = self.symref(x)
            return self.convval(x)
        else:  # same
            return self.convsam(x)
        
def residual_block(x, filters, conv_size=(3, 3), stride=1, dilations=1, relu_alpha=0.2, norm=None, padding=None):
    in_channels = int(x.shape[-1])
    x_in = x

    if stride > 1:
        x_in = AveragePooling2D(pool_size=(stride, stride))(x_in)
    if (filters != in_channels):
        x_in = Conv2D(filters=filters, kernel_size=(1, 1))(x_in)

    # first block of activation and 3x3 convolution (possibly strided, although we don't use this)
    x = LeakyReLU(relu_alpha)(x)
    x = Conv2DPadding(filters=filters, kernel_size=conv_size, stride=stride, dilations=dilations, padding=padding)(x)
    if norm == "batch":
        x = BatchNormalization()(x)
    elif norm is None:
        pass
    else:
        print("norm type not implemented")

    # second block of activation and 3x3 unstrided convolution
    x = LeakyReLU(relu_alpha)(x)
    x = Conv2DPadding(filters=filters, kernel_size=conv_size, stride=1, dilations=dilations, padding=padding)(x)
    if norm == "batch":
        x = BatchNormalization()(x)
    elif norm is None:
        pass
    else:
        print("norm type not implemented")

    # skip connection
    x = Add()([x, x_in])

    return x


def const_upscale_block(const_input, steps, filters):
    # Map (N x kH x kW x C) to (N x H x W x f), where k is downscaling factor
    const_output = const_input
    for step in steps:
        const_output = Conv2D(filters=filters, kernel_size=(step, step), strides=step, padding="valid", activation="relu")(const_output)
    return const_output
