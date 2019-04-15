# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import numpy as np

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)
    
    
def conv1d(inputs, num_filters, kernel_size, name):
    return tf.layers.conv1d(inputs, num_filters, kernel_size, name=name)

def global_max_pooling(inputs, axis, name):
    return tf.reduce_max(inputs, axis=axis, name=name)
            
