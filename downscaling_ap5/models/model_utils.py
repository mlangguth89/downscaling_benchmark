__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-05-26"
__update__ = "2022-05-31"

"""
Some auxiliary methods to create Keras models.
"""
# import modules
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPool2D)
from unet_model import UNET, unet_model
from wgan_model import WGAN, critic_model


class ModelEngine(object):
    """
    Class to get and instantiate known models.
    To add new models, please adapt known_models accoringly.
    General info:
    The key value of the implemented model should be based on keras.Model and should expect 'hparams', 'exp_name' and
    'model_savedir' as arguments for initialization. Further keyword arguments are possible.
    For composite models (e.g. GANs):
    The key value should be a tuple whose first element constitutes the composited model (based on keras.Model).
    The following arguments should then be model construction objects to define the components of the composite model.
    Example: WGAN is a composite model consisting of a generator and critic (see wgan_model.py).
             Thus, the derived class should be the first element of the tuple.
             The model constructions of the generator (e.g. a U-Net) and the critic must then constitute the second and
             third element of the tuple, i.e. {"wgan": (WGAN, unet_model, critic_model).
    """

    known_models = {"u-net": UNET,
                    "wgan": (WGAN, unet_model, critic_model)}

    def __init__(self, model_name: str):
        """
        Initialize the model if known.
        :param model_name: name of the model (must match any key of self.known_models)
        Hint: Pass help to get an overview of the available models.
        """
        self.modelname = model_name
        if self.modelname is None:
            self.model = None
        else:
            self.model = self.known_models[self.modelname]

    def __call__(self, hparams_dict, exp_name, save_dir, **kwargs):
        """
        Instantiate the model with some required arguments.
        """
        model_list = list(self.model)
        target_model = model_list[0]
        model_args = {"hparams": hparams_dict, "exp_name": exp_name, "model_savedir": save_dir, **kwargs}

        try:
            if len(model_list) == 1:
                model = target_model(**model_args)
            else:
                submodels = model_list[1:]
                model = target_model(*submodels, **model_args)
        except Exception as e:
            err_str = str(e)
            raise RuntimeError(f"Failed to instantiate the model. The following error occured: \n {err_str}")

        return model

    @property
    def modelname(self):
        return self._modelname

    @modelname.setter
    def modelname(self, model_name):

        help_str = self._get_help_str()
        model_name_local = model_name.lower()

        if model_name_local in self.known_models.keys():
            self._modelname = model_name_local
        elif model_name_local == "help":
            print(help_str)
            self._modelname = None
        else:
            raise ValueError(f"Model '{model_name}' is unknown. Please specify a known model. {help_str}")

    def _get_help_str(self):
        """
        Create help-string listing all known models.
        """
        return f"Known models are: {', '.join(list(self.known_models.keys()))}"


def get_model(model_name: str):
    """
    Get a Keras model given the provided model_name. Parse "help" to get an overview of existing models.
    :param model_name: name of known models.
    :return: model object or tuple of model objects for composite models. In case of help, None is returned!
    """
    known_models = {"u-net": UNET,
                    "wgan": (WGAN, unet_model, critic_model)}

    help_str = f"Known models are: {', '.join(list(known_models.keys()))}"
    model_name_local = model_name.lower()

    if model_name_local in known_models.keys():
        model = known_models[model_name_local]
    elif model_name_local == "help":
        print(help_str)
        model = None
    else:
        raise ValueError(f"Model '{model_name}' is unknown. Please specify a known model. {help_str}")

    return model


def handle_opt_utils(model: keras.Model, opt_funcname: str):
    """
    Retrieves dictionary of optional parameters from model when a corresponding class method is available.
    :param model: Keras model or derived Keras Model.
    :param opt_funcname: name of method providing dictionary of options.
    """
    func_opt = getattr(model, opt_funcname, None)
    if callable(func_opt):
        opt_dict = func_opt()
        assert isinstance(opt_dict, dict), f"Model method '{opt_funcname}' must provide a dictionary."
    elif func_opt is None:
        opt_dict = {}
    else:
        raise TypeError(f"Model method '{opt_funcname}' must be a callable providing dictionary of options.")

    return opt_dict


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
