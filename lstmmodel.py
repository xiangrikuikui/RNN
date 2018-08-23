# -*- coding: utf-8 -*-
from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 


import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.io as sio
import numpy as np
from tensorflow.python.ops.nn import relu,softmax
import tensorflow.contrib.layers as layers
lstm_size = 256
number_of_layers = 2
outputdim = 3
L1decay_1 = 0.0001
def lstm(model_input, sequence_length):
  """Creates a model which uses a stack of LSTMs to represent the video.

  Args:
    model_input: A 'batch_size' x 'num_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """
#  stacked_lstm = tf.contrib.rnn.MultiRNNCell(
#          [
#              tf.contrib.rnn.BasicLSTMCell(
#                  lstm_size, forget_bias=1.0, state_is_tuple=True)
#              for _ in range(number_of_layers)
#              ], state_is_tuple=True)
#  
#  basiccell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
#  cell = tf.contrib.rnn.DropoutWrapper(basiccell, output_keep_prob=0.8)
#  stacked_lstm = tf.contrib.rnn.MultiRNNCell([cell for _ in range(number_of_layers)])
#  with tf.variable_scope("RNN"):
#    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
#                                       sequence_length=num_frames,
#                                       dtype=tf.float32)
  
  def SingleCell():
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=0.8)
  
  stacked_lstm=tf.contrib.rnn.MultiRNNCell([SingleCell() for _ in range(number_of_layers)], state_is_tuple=True)
  
  with tf.variable_scope("LSTM"):
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=sequence_length,
                                       dtype=tf.float32)
    
    print("type(output): ", type(outputs))
    print("output: ", outputs)
    print("type(state): ", type(state))
    print("state: ", state)
    
  net = slim.fully_connected(outputs[:, -1, :], #state[-1][-1]
             outputdim, 
						 activation_fn=tf.nn.relu, 
						 weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
						 weights_regularizer=layers.l1_regularizer(L1decay_1),
						 biases_initializer=tf.zeros_initializer(),
						 trainable=True,
						 scope='fc')
  return net