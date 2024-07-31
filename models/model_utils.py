# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Some auxiliary methods to create Keras models.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-05-26"
__update__ = "2024-03-07"

# import modules
import os
from timeit import default_timer as timer
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.models import Model


# define class for creating timer callback
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_times.append(timer() - self.epoch_time_start)

def check_horovod():
    """
    Check if job is run with Horovod based on environment variables

    :return: True if Horovod is detected, False otherwise
    """
    # Program is run with horovodrun
    with_horovod = "HOROVOD_RANK" in os.environ

    if not with_horovod:
        # Program is run with srun
        with_horovod = "SLURM_STEP_NUM_TASKS" in os.environ and int(os.environ["SLURM_STEP_NUM_TASKS"]) > 1

    return with_horovod


def set_gpu_memory_growth():
    """
    Set GPU memory growth to avoid allocating all memory at once.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


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


# Helpers from MLAir, see:
# https://gitlab.jsc.fz-juelich.de/esde/machine-learning/mlair/-/blob/master/mlair/helpers/helpers.py?ref_type=heads (MIT License)
def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_pickable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


