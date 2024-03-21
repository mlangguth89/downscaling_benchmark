# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2023-12-11"
__update__ = "2024-03-20"

# import modules
import os
import inspect
from abc import ABC
from typing import Any, Dict

import tensorflow.keras as keras
import tensorflow as tf

from model_utils import TimeHistory, handle_opt_utils, make_keras_pickable
from other_utils import merge_dicts, remove_items, to_list
import json


class AbstractModelClass(ABC):
    """
    The AbstractModelClass provides a unified skeleton for any model provided to the machine learning workflow.

    The model can always be accessed by calling ModelClass.model or directly by an model method without parsing the
    model attribute name (e.g. ModelClass.model.compile -> ModelClass.compile). Beside the model, this class provides
    the corresponding loss function.
    Note that the __init__-method of the child class must always call the following to ensure a working model:
        # set customized hyperparameters
        self.set_hparams(hparams)
        # set model
        self.set_model()
        # set compile and fit options as well as custom objects
        self.set_compile_options()
        self.set_custom_objects(loss=self.compile_options['loss'])
        self.set_fit_options()
    """

    _requirements = []

    def __init__(self, input_shape, hparams, varnames_tar, savedir, expname) -> None:
        """Predefine internal attributes for model and loss."""
        make_keras_pickable()
        self.__model = None
        self.model_name = self.__class__.__name__
        self.__custom_objects = {}
        self.__fit_options = {}
        self.__allowed_compile_options = {'optimizer': None,
                                          'loss': None,
                                          'metrics': None,
                                          'loss_weights': None,
                                          'sample_weight_mode': None,
                                          'weighted_metrics': None,
                                          'target_tensors': None
                                          }
        self.__compile_options = self.__allowed_compile_options
        self.__compile_options_is_set = False
        self._input_shape = input_shape
        self.__hparams = hparams
        self._varnames_tar = to_list(varnames_tar)
        self._savedir = savedir
        self._expname = expname
        self._n_predictands = len(self._varnames_tar)

    def load_model(self, name: str, compile: bool = False) -> None:
        hist = self.model.history
        self.model.load_weights(name)
        self.model.history = hist
        if compile is True:
            self.model.compile(**self.compile_options)

    def __getattr__(self, name: str) -> Any:
        """
        Is called if __getattribute__ is not able to find requested attribute.

        Normally, the model class is saved into a variable like `model = ModelClass()`. To bypass a call like
        `model.model` to access the _model attribute, this method tries to search for the named attribute in the
        self.model namespace and returns this attribute if available. Therefore, following expression is true:
        `ModelClass().compile == ModelClass().model.compile` as long the called attribute/method is not part if the
        ModelClass itself.

        :param name: name of the attribute or method to call

        :return: attribute or method from self.model namespace
        """
        return self.model.__getattribute__(name)

    @property
    def model(self) -> keras.Model:
        """
        The model property containing a keras.Model instance.

        :return: the keras model
        """
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value

    @property
    def fit_options(self) -> Dict:
        """
        Additional fit options to be set for model.fit().
        """
        return self.__fit_options
    
    @fit_options.setter
    def fit_options(self, value) -> None:
        self.__fit_options = value 

    @property
    def custom_objects(self) -> Dict:
        """
        The custom objects property collects all non-keras utilities that are used in the model class.

        To load such a customised and already compiled model (e.g. from local disk), this information is required.

        :return: custom objects in a dictionary
        """
        return self.__custom_objects

    @custom_objects.setter
    def custom_objects(self, value) -> None:
        self.__custom_objects = value

    @property
    def compile_options(self) -> Dict:
        """
        The compile options property allows the user to use all keras.compile() arguments. They can ether be passed as
        dictionary (1), as attribute, without setting compile_options (2) or as mixture (partly defined as instance
        attributes and partly parsing a dictionary) of both of them (3).
        The method will raise an Error when the same parameter is set differently.

        Example (1) Recommended (includes check for valid keywords which are used as args in keras.compile)
        .. code-block:: python
            def set_compile_options(self):
                self.compile_options = {"optimizer": keras.optimizers.SGD(),
                                        "loss": keras.losses.mean_squared_error,
                                        "metrics": ["mse", "mae"]}

        Example (2)
        .. code-block:: python
            def set_compile_options(self):
                self.optimizer = keras.optimizers.SGD()
                self.loss = keras.losses.mean_squared_error
                self.metrics = ["mse", "mae"]

        Example (3)
        Correct:
        .. code-block:: python
            def set_compile_options(self):
                self.optimizer = keras.optimizers.SGD()
                self.loss = keras.losses.mean_squared_error
                self.compile_options = {"metrics": ["mse", "mae"]}

        Incorrect: (Will raise an error)
        .. code-block:: python
            def set_compile_options(self):
                self.optimizer = keras.optimizers.SGD()
                self.loss = keras.losses.mean_squared_error
                self.compile_options = {"optimizer": keras.optimizers.Adam(), "metrics": ["mse", "mae"]}

        Note:
        * As long as the attribute and the dict value have exactly the same values, the setter method will not raise
        an error
        * For example (2) there is no check implemented, if the attributes are valid compile options


        :return:
        """
        if self.__compile_options_is_set is False:
            self.compile_options = None
        return self.__compile_options

    @compile_options.setter
    def compile_options(self, value: Dict) -> None:
        
        if isinstance(value, dict):
            if not (set(value.keys()) <= set(self.__allowed_compile_options.keys())):
                raise ValueError(f"Got invalid key for compile_options. {value.keys()}")

        for allow_k in self.__allowed_compile_options.keys():
            if hasattr(self, allow_k):
                new_v_attr = getattr(self, allow_k)
                if new_v_attr == list():
                    new_v_attr = None
            else:
                new_v_attr = None
            if isinstance(value, dict):
                new_v_dic = value.pop(allow_k, None)
            elif value is None:
                new_v_dic = None
            else:
                raise TypeError(f"`compile_options' must be `dict' or `None', but is {type(value)}.")
            ## self.__compare_keras_optimizers() foremost disabled, because it does not work as expected
            #if (new_v_attr == new_v_dic or self.__compare_keras_optimizers(new_v_attr, new_v_dic)) or (
            #        (new_v_attr is None) ^ (new_v_dic is None)):
            if (new_v_attr == new_v_dic) or ((new_v_attr is None) ^ (new_v_dic is None)):
                if new_v_attr is not None:
                    self.__compile_options[allow_k] = new_v_attr
                else:
                    self.__compile_options[allow_k] = new_v_dic

            else:
                raise ValueError(
                    f"Got different values or arguments for same argument: self.{allow_k}={new_v_attr.__class__} and '{allow_k}': {new_v_dic.__class__}")
        self.__compile_options_is_set = True
        
    @property
    def hparams(self):
        return self.__hparams
    
    @hparams.setter
    def hparams(self, hparams):
        self.set_hparams_default()
        
        hparams_dict = merge_dicts(self.hparams_default, hparams, recursive=False)   # merge default and user-defined hyperparameters
        
        self.__hparams = hparams_dict      

    @property
    def savedir(self):
        return self._savedir
    
    @savedir.setter
    def savedir(self, savedir):
        
        if not os.path.isdir(savedir):
            os.makedirs(savedir, exist_ok=True)
        
        self._savedir = savedir   

    @property
    def expname(self):
        return self._expname
    
    @expname.setter
    def expname(self, expname):
        self._expname = expname

        
    def set_hparams(self, hparams):
        self.hparams = hparams

    @staticmethod
    def __extract_from_tuple(tup):
        """Return element of tuple if it contains only a single element."""
        return tup[0] if isinstance(tup, tuple) and len(tup) == 1 else tup

    @staticmethod
    def __compare_keras_optimizers(first, second):
        """
        Compares if optimiser and all settings of the optimisers are exactly equal.

        :return True if optimisers are interchangeable, or False if optimisers are distinguishable.
        """
        if isinstance(list, type(second)):
            res = False
        else:
            if first.__class__ == second.__class__ and '.'.join(
                    first.__module__.split('.')[0:4]) == 'tensorflow.python.keras.optimizer_v2':
                res = True
                init = tf.compat.v1.global_variables_initializer()
                with tf.compat.v1.Session() as sess:
                    sess.run(init)
                    for k, v in first.__dict__.items():
                        try:
                            res *= sess.run(v) == sess.run(second.__dict__[k])
                        except TypeError:
                            res *= v == second.__dict__[k]
            else:
                res = False
        return bool(res)

    def get_settings(self) -> Dict:
        """
        Get all class attributes that are not protected in the AbstractModelClass as dictionary.
        """
        return dict((k, v) for (k, v) in self.__dict__.items() if not k.startswith("_AbstractModelClass__"))

    def set_model(self):
        """Abstract method to set model."""
        raise NotImplementedError

    def set_compile_options(self):
        """
        This method only has to be defined in child class, when additional compile options should be used ()
        (other options than optimizer and loss)
        Has to be set as dictionary: {'optimizer': None,
                                      'loss': None,
                                      'metrics': None,
                                      'loss_weights': None,
                                      'sample_weight_mode': None,
                                      'weighted_metrics': None,
                                      'target_tensors': None
                                      }

        :return:
        """
        raise NotImplementedError
        
    def set_hparams_default(self):
        """
        Dictionary of default hyperparameters.
        """
        raise NotImplementedError
    
    def set_fit_options(self) -> None:
        """
        Add further fit options if get_fit_opts is defined in child class.
        Note that get_fit_opts has to provide a dictionary of fit options, which is then combined with the default TimeHistory callback.
        Example:
        To add further callbacks, define get_fit_opts in child class as follows:
        def get_fit_opts(self):
            return {'callbacks': [EarlyStopping(monitor='val_loss', patience=10)]}
        """
        fit_opts = {'callbacks': [TimeHistory()]}

        add_opts = handle_opt_utils(self, "get_fit_options")

        fit_opts.update({k: fit_opts.get(k, []) + v for k, v in add_opts.items()})

        self.fit_options = fit_opts

    def set_custom_objects(self, **kwargs) -> None:
        """
        Set custom objects that are not part of keras framework.

        These custom objects are needed if an already compiled model is loaded from disk. There is a special treatment
        for the Padding2D class, which is a base class for different padding types. For a correct behaviour, all
        supported subclasses are added as custom objects in addition to the given ones.

        :param kwargs: all custom objects, that should be saved
        """
        if "Padding2D" in kwargs.keys():
            kwargs.update(kwargs["Padding2D"].allowed_paddings)
        self.custom_objects = kwargs

    def save_hparams_to_json(self):
        """
        Save hyperparameters to json file und savedir.
        """
        with open(os.path.join(self.savedir, f"config_{self.model_name.lower()}.json"), "w") as f:
            json.dump(self.hparams, f)

    @classmethod
    def requirements(cls):
        """Return requirements and own arguments without duplicates."""
        return list(set(cls._requirements + cls.own_args()))

    @classmethod
    def own_args(cls, *args):
        """Return all arguments (including kwonlyargs)."""
        arg_spec = inspect.getfullargspec(cls)
        list_of_args = arg_spec.args + arg_spec.kwonlyargs + cls.super_args()
        return list(set(remove_items(list_of_args, ["self"] + list(args))))

    @classmethod
    def super_args(cls):
        args = []
        for super_cls in cls.__mro__:
            if super_cls == cls:
                continue
            if hasattr(super_cls, "own_args"):
                # args.extend(super_cls.own_args())
                args.extend(getattr(super_cls, "own_args")())
        return list(set(args))
