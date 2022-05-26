__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-05-26"
__update__ = "2022-05-26"

"""
Some auxiliary methods to create Keras models.
"""

import tensorflow.keras.layers as layers


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
