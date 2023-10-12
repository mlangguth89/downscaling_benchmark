# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Some auxiliary methods to create Keras models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-05-26"
__update__ = "2023-10-12"

# import modules
from timeit import default_timer as timer
import xrarry as xr
import tensorflow.keras as keras
from unet_model import sha_unet, UNET
from wgan_model import WGAN, critic_model
from other_utils import to_list


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

    known_models = {"unet": (UNET, sha_unet),
                    "wgan": (WGAN, sha_unet, critic_model)}

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

    def __call__(self, shape_in, varnames_tar, hparams_dict, save_dir, exp_name, **kwargs):
        """
        Instantiate the model with some required arguments.
        """
        model_list = to_list(self.model)
        target_model = model_list[0]
        model_args = {"shape_in": shape_in, "varnames_tar": varnames_tar, "hparams": hparams_dict,
                      "exp_name": exp_name, "savedir": save_dir, **kwargs}

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


# define class for creating timer callback
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_times.append(timer() - self.epoch_time_start)


def get_loss_from_history(history: keras.callbacks.History, loss_name: str = "loss"):

    try:
        loss_req = history.history[loss_name]
    except KeyError:
        raise KeyError(f"Cannot find {loss_name} in Keras history callback." +
                       f"The following keys are available: {*history.history.keys(), }")

    return loss_req


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


def convert_to_xarray(mout_np, norm, varname, coords, dims, z_branch=False):
    """
    Converts numpy-array of model output to xarray.DataArray and performs denormalization.
    :param mout_np: numpy-array of model output
    :param norm: normalization object
    :param varname: name of variable
    :param coords: coordinates of target data
    :param dims: dimensions of target data
    :param z_branch: flag for z-branch
    :return: xarray.DataArray of model output with denormalized data
    """
    if z_branch:
        # slice data to get first channel only
        if isinstance(mout_np, list): mout_np = mout_np[0]
        mout_xr = xr.DataArray(mout_np[..., 0].squeeze(), coords=coords, dims=dims, name=varname)
    else:
        # no slicing required
        mout_xr = xr.DataArray(mout_np.squeeze(), coords=coords, dims=dims, name=varname)

    # perform denormalization
    mout_xr = norm.denormalize(mout_xr, varname=varname)

    return mout_xr


