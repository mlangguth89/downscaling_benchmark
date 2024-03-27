# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Model engine to get and instantiate known models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2023-12-15"
__update__ = "2024-03-21"

# import modules
from unet_model import Sha_UNet, DeepRU_UNet
from wgan_model import WGAN, Critic_Simple
from other_utils import to_list

class ModelEngine(object):
    """
    Class to get and instantiate known models.
    To add new models, please adapt known_models accordingly.
    General info:
    The key value of the implemented model should be based on keras.Model and should expect 'hparams', 'exp_name' and
    'model_savedir' as arguments for initialization. Further keyword arguments are possible.
    For composite models (e.g. GANs):
    The key value should be a tuple whose first element constitutes the composited model (based on keras.Model).
    The following arguments should then be model construction objects to define the components of the composite model.
    Example: WGAN is a composite model consisting of a generator and critic (see wgan_model.py).
             Thus, the derived class should be the first element of the tuple.
             The model constructions of the generator (e.g. a U-Net) and the critic must then constitute the second and
             third element of the tuple, i.e. {"wgan": (WGAN, Sha_UNet, Critic_Simple).
    """

    known_models = {"sha_unet": (Sha_UNet,),
                    "deepru": (DeepRU_UNet,),
                    "sha_wgan": (WGAN, Sha_UNet, Critic_Simple)}
    
    long_names = ["Sha U-Net", "DeepRU", "Sha WGAN"]
    
    assert len(known_models) == len(long_names), f"Conflicting number of known_models ({len(known_models)})" + \
                                                 f" and long_names ({len(long_names)})."

    def __init__(self, model_name: str):
        """
        Initialize the model if known.
        :param model_name: name of the model (must match any key of self.known_models)
        Hint: Pass help to get an overview of the available models.
        """
        self.modelname = model_name
        self.model = self.known_models[self.modelname]
        self.model_longname = self.long_names[list(self.known_models.keys()).index(self.modelname)]

    def __call__(self, shape_in, varnames_tar, hparams_dict, save_dir, expname, **kwargs):
        """
        Instantiate the model with some required arguments.
        """
        model_list = to_list(self.model)
        target_model = model_list[0]
        model_args = {"shape_in": shape_in, "varnames_tar": varnames_tar, "hparams": hparams_dict,
                      "savedir": save_dir, "expname": expname, **kwargs}

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
