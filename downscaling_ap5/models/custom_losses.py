# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Some custmoized losses (e.g. on vector quantities)
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2023-06-16"
__update__ = "2023-06-16"

# import module
import inspect
import tensorflow as tf

def fix_channels(n_channels):
    """
    Decorator to fix number of channels in loss functions.
    """
    def decorator(func):
        def wrapper(y_true, y_pred, **func_kwargs):
            return func(y_true, y_pred, n_channels, **func_kwargs)
        return wrapper
    return decorator

def get_custom_loss(loss_name, **kwargs):
    """
    Loss factory including some customized losses and all available Keras losses
    :param loss_name: name of the loss function
    :return: the respective layer to deploy desired activation
    """
    known_losses = ["mse_channels", "mae_channels", "mae_vec", "mse_vec", "critic", "critic_generator"] + \
                   [loss_cls[0] for loss_cls in inspect.getmembers(tf.keras.losses, inspect.isclass)]

    loss_name = loss_name.lower()

    n_channels = kwargs.get("n_channels", None)

    if loss_name == "mse_channels":
        assert n_channels > 0, f"n_channels must be a number larger than zero, but is {n_channels}."
        loss_fn = fix_channels(**kwargs)(mse_channels)
    elif loss_name == "mae_channels":
        assert n_channels > 0, f"n_channels must be a number larger than zero, but is {n_channels}."
        loss_fn = fix_channels(**kwargs)(mae_channels)
    elif loss_name == "mae_vec":
        assert n_channels > 0, f"n_channels must be a number larger than zero, but is {n_channels}."
        loss_fn = fix_channels(**kwargs)(mae_vec)
    elif loss_name == "mse_vec":
        assert n_channels > 0, f"n_channels must be a number larger than zero, but is {n_channels}."
        loss_fn = fix_channels(**kwargs)(mse_vec)
    elif loss_name == "critic":
        loss_fn = critic_loss
    elif loss_name == "critic_generator":
        loss_fn = critic_gen_loss
    else:
        loss_fn = loss_name
        try:
            _ = tf.keras.losses.get(loss_name)
            loss_fn = loss_name
        except AttributeError:
            raise ValueError(f"{loss_name} is not a valid loss function. Choose one of the following: {known_losses}")

    return loss_fn


def mae_channels(x, x_hat, n_channels: int = None, channels_last: bool = True, avg_channels: bool = False):
    rloss = 0.
    if channels_last:
        # get MAE for all output heads
        for i in range(n_channels):
            rloss += tf.reduce_mean(tf.abs(tf.squeeze(x_hat[..., i]) - x[..., i]))
    else:
        for i in range(n_channels):
            rloss += tf.reduce_mean(tf.abs(tf.squeeze(x_hat[i, ...]) - x[i, ...]))
            
    if avg_channels:
        rloss /= n_channels
        
    return rloss
            
def mse_channels(x, x_hat, n_channels, channels_last: bool = True, avg_channels: bool = False):
    rloss = 0.
    if channels_last:
        # get MAE for all output heads
        for i in range(n_channels):
            rloss += tf.reduce_mean(tf.square(tf.squeeze(x_hat[..., i]) - x[..., i]))
    else:
        for i in range(n_channels):
            rloss += tf.reduce_mean(tf.square(tf.squeeze(x_hat[i, ...]) - x[i, ...]))
            
    if avg_channels:
        rloss /= n_channels
        
    return rloss

def mae_vec(x, x_hat, n_channels, channels_last: bool = True, avg_channels: bool = False, nd_vec:  int = None):
    
    if nd_vec is None:
        nd_vec = n_channels
        
    rloss = 0.
    if channels_last:
        vec_ind = -1
        diff = tf.squeeze(x_hat[..., 0:nd_vec]) - x[..., 0:nd_vec]
    else:
        vec_ind = 1
        diff = tf.squeeze(x_hat[:,0:nd_vec, ...]) - x[:,0:nd_vec, ...]
    
    rloss = tf.reduce_mean(tf.norm(diff, axis=vec_ind))
    #rloss = tf.reduce_mean(tf.math.reduce_euclidean_norm(diff, axis=vec_ind))

    if nd_vec > n_channels: 
        if channels_last:
            rloss += mae_channels(x[..., nd_vec::], x_hat[..., nd_vec::], True, avg_channels)
        else:
            rloss += mae_channels(x[:, nd_vec::, ...], x_hat[:, nd_vec::, ...], True, avg_channels)    
            
    return rloss

def mse_vec(x, x_hat, n_channels, channels_last: bool = True, avg_channels: bool = False, nd_vec:  int = None):
    
    if nd_vec is None:
        nd_vec = n_channels
        
    rloss = 0.
    
    if channels_last:
        vec_ind = -1
        diff = tf.squeeze(x_hat[..., 0:nd_vec]) - x[..., 0:nd_vec]
    else:
        vec_ind = 1
        diff = tf.squeeze(x_hat[:,0:nd_vec, ...]) - x[:,0:nd_vec, ...]
        
    rloss = tf.reduce_mean(tf.square(tf.norm(diff, axis=vec_ind)))
    
    if nd_vec > n_channels: 
        if channels_last:
            rloss += mse_channels(x[..., nd_vec::], x_hat[..., nd_vec::], True, avg_channels)
        else:
            rloss += mse_channels(x[:, nd_vec::, ...], x_hat[:, nd_vec::, ...], True, avg_channels)   
            
    return rloss

def critic_loss(critic_real, critic_gen):
    """
    The critic is optimized to maximize the difference between the generated and the real data max(real - gen).
    This is equivalent to minimizing the negative of this difference, i.e. min(gen - real) = max(real - gen)
    :param critic_real: critic on the real data
    :param critic_gen: critic on the generated data
    :return c_loss: loss to optize the critic
    """
    c_loss = tf.reduce_mean(critic_gen - critic_real)

    return c_loss


def critic_gen_loss(critic_gen):
    cg_loss = -tf.reduce_mean(critic_gen)

    return cg_loss
