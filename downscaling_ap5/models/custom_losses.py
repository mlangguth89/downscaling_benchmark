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
import tensorflow as tf

def mae(x, x_hat, n_channels, channels_last: bool = True, avg_channels: bool = False):
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
            
def mse(x, x_hat, n_channels, channels_last: bool = True, avg_channels: bool = False):
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
            rloss += mae(x[..., nd_vec::], x_hat[..., nd_vec::], True, avg_channels)
        else:
            rloss += mae(x[:, nd_vec::, ...], x_hat[:, nd_vec::, ...], True, avg_channels)    
            
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
            rloss += mse(x[..., nd_vec::], x_hat[..., nd_vec::], True, avg_channels)
        else:
            rloss += mse(x[:, nd_vec::, ...], x_hat[:, nd_vec::, ...], True, avg_channels)   
            
    return rloss