# encoding:utf-8
# Authored by: chaofeng.gcf
# =============================================
"""The API detail
NOTE: customize model plugin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import numpy as np
from functools import reduce

from tensorflow.python.ops import variables
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.keras import initializers
from tensorflow.python.keras import activations
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.python.feature_column import feature_column as common_column
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras.layers import Dense


class CustomModel:
  def __init__(self,
               input_len=128,
               task_num=3,
               hidden_units=[64, 32, 1],
               use_bias=[True, True, False],
               activation=['relu', 'relu', 'sigmoid'],
               init_strategy='part',
               kernel_initializer='he_uniform',
               split_parts=10,
               filter_freq=0):
    '''
    Args:
      init_strategy: all, part.
      kernel_initializer: be used while init_strategy is not all.
    '''
    assert len(use_bias) == len(hidden_units) and \
      len(use_bias) == len(activation)

    self.input_len = input_len
    self.task_num = task_num
    self.hidden_units = hidden_units
    self.use_bias = list(map(lambda i:int(i), use_bias))
    self.activations = [activations.get(act) for act in activation]
    self.init_strategy = init_strategy
    self.kernel_initializer = kernel_initializer
    self.split_parts = split_parts
    self.filter_freq = filter_freq
    self.steps_to_live = None

    self.kernel_shape = list(zip(([self.input_len] + self.hidden_units[:-1]), self.hidden_units))
    self.kernel_params = list(map(lambda x: reduce(lambda i, j: i*j, x) ,self.kernel_shape))
    self.bias_params = list(np.array(self.hidden_units) * np.array(self.use_bias))
    self.layer_params = list(np.array(self.kernel_params) + np.array(self.bias_params))
    self.task_params = self.task_num * [sum(self.layer_params)]
    self.params = sum(self.task_params)

    self.name = str(11000)
    filter_options = variables.CounterFilter(filter_freq=self.filter_freq)
    ev_option = variables.EmbeddingVariableOption(filter_option=filter_options)
    sparse_id_column = feature_column.sparse_column_with_embedding(
                                                column_name = self.name,
                                                dtype = dtypes.int64,
                                                steps_to_live = self.steps_to_live,
                                                partition_num = self.split_parts,
                                                ev_option=ev_option)
    self.sparse_embedding_column = feature_column.embedding_column(
                                              sparse_id_column=sparse_id_column,
                                              dimension=self.params,
                                              combiner='sum',
                                              initializer = self.cm_initializer)

  def _initializer(self, shape, dtype=dtypes.float32, seed=None):
    def _compute_fans(shape):
      if len(shape) < 1:
        fan_in = fan_out = 1
      elif len(shape) == 1:
        fan_in = fan_out = shape[0]
      elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
      else:
        fan_in = shape[-2]
        fan_out = shape[-1]
      return fan_in, fan_out

    fan_in, fan_out = _compute_fans(shape)
    scale = 2
    scale /= max(1., fan_in)
    if self.kernel_initializer == 'he_uniform':
      limit = math.sqrt(3.0 * scale)
      return random_ops.random_uniform(
                 shape, -limit, limit, dtype, seed=seed)
    elif self.kernel_initializer == 'he_normal':
      stddev = math.sqrt(scale) / 0.87962566103423978
      return random_ops.truncated_normal(
                 shape, 0.0, stddev, dtype, seed=seed)

  def cm_initializer(self, shape, dtype=None, partition_info=None):
    def _wrapper_init(shape):
      elements = []
      for task_id in range(self.task_num):
        for kernel_shape, bias_param in zip(self.kernel_shape, self.bias_params):
          kernel = array_ops.reshape(self._initializer(shape[:1]+list(kernel_shape),
                                                       dtype=dtype), [shape[0], -1])
          if bias_param:
            bias = array_ops.zeros([shape[0], bias_param], dtype)
            elements.append(array_ops.concat([kernel, bias], axis=1))
          else:
            elements.append(kernel)
      params = array_ops.concat(elements, axis=1)
      return params
    
    if self.init_strategy == 'all':
      scale = 1.0 / math.sqrt(reduce(lambda i,j: i*j, self.hidden_units))
      return random_ops.truncated_normal(shape, 0, scale, dtypes.float32)
    elif self.init_strategy == 'part':
      return _wrapper_init(shape)

  def cmu(self, param, x):
    '''
    Inner computation graph which can be used to infer
    '''
    outputs = []
    i = 1
    for index, task in enumerate(array_ops.split(param, self.task_params, axis=1)):
      task_x = array_ops.reshape(gen_array_ops.stop_gradient(x[index]), [-1, 1, self.input_len])
      task_x = tf.Print(task_x, [task_x], message='x:\n', first_n=10, summarize=1000)
      j = 1
      task = tf.Print(task, [task], message=str(i)+'_task:\n', first_n=100, summarize=1000)
      for kernel_shape, bias_param, activation, layer in zip(self.kernel_shape,
                                                             self.bias_params,
                                                             self.activations,
                                                             array_ops.split(task,
                                                               self.layer_params,
                                                               axis=1)):
        kernel_param = reduce(lambda i,j: i*j, kernel_shape)
        layer = tf.Print(layer, [layer], message=str(i)+str(j)+'_layer:\n', first_n=1, summarize=1000)
        raw_kernel, bias = array_ops.split(layer, [kernel_param, bias_param], axis=1)
        kernel = array_ops.reshape(raw_kernel, [-1]+list(kernel_shape))
        kernel = tf.Print(kernel, [kernel], message=str(i)+str(j)+'_kernel:\n', first_n=1, summarize=1000)
        task_x = math_ops.matmul(task_x, kernel)
        task_x = tf.Print(task_x, [task_x], message=str(i)+str(j)+'_task_x:\n', first_n=1, summarize=1000)
        if bias_param:
          task_x += array_ops.reshape(bias, [-1, 1, bias_param])
        task_x = activation(task_x)
        j+=1
      i+=1
      task_x = array_ops.reshape(task_x, [-1, self.hidden_units[-1]])
      outputs.append(task_x)
    return outputs

  def forward(self, x, key):
    '''
    Process inputs before training.
    
    '''
    def _convert(tensor):
      split_string = string_ops.string_split(tensor, delimiter = ',')
      split_string_val = string_ops.string_to_number(split_string.values, out_type=dtypes.int64)
      tensor = sparse_tensor.SparseTensor(indices = split_string.indices,
                                          values = split_string_val,
                                          dense_shape = split_string.dense_shape)
      return tensor

    assert len(x) == self.task_num
    if not isinstance(key, sparse_tensor.SparseTensor):
      key = _convert(key)
    cm_param = common_column.input_layer({self.name : key},
                                         self.sparse_embedding_column)
    preds = self.cmu(cm_param, x)
    assert len(preds) == self.task_num
    return preds

  def get_config(self):
    return {'kernel_shape' : self.kernel_shape,
            'kernel_params' : self.kernel_params,
            'bias_params' : self.bias_params,
            'layer_params' : self.layer_params,
            'task_params' : self.task_params,
            'params' : self.params }


class NeuralNetwork:
  '''
  1. unify to process
  '''
  def __init__(self,
               scope_name,
               hidden_units=[1],
               activation=None,
               use_bias=None,
               initializer=None,
               **kwargs):
    assert len(hidden_units) > 0
    self.scope_name = scope_name
    self.hidden_units = hidden_units
    self.activation = activation if activation else ['relu'] * len(hidden_units)
    self.use_bias = use_bias if use_bias else [True] * len(hidden_units)
    self.initializer = initializer if initializer else 'he_uniform'
    self.dense_list = []
    self.outputs = []
    self.partitioner = None

    if 'partitioner' in kwargs:
      self.partitioner = kwargs['partitioner']
    for units, act, bias in zip(self.hidden_units, self.activation, self.use_bias):
      dense = Dense(units,
                    activation=act,
                    use_bias=bias,
                    kernel_initializer=self.initializer,
                    bias_initializer='zeros')
      self.dense_list.append(dense)

  def __call__(self, x):
    with variable_scope.variable_scope(self.scope_name,
                                       partitioner=self.partitioner,
                                       reuse=variable_scope.AUTO_REUSE):
      for dense in self.dense_list:
        x = dense(x)
        self.outputs.append(x)
    return self.outputs[-1]

  def get_layers(self):
    return self.dense_list

  def get_outputs(self):
    return self.outputs
  
  def get_config(self):
    return [layer.get_config() for layer in self.get_layers()]

def neural_network(x, config, name):
  return NeuralNetwork(config, name)(x)


if __name__ == '__main__':
  cm = CustomModel()
  print(cm.get_config())
  import tensorflow as tf
  key = tf.constant(['2314,3432', '3432'], dtype=tf.string)
  x1 = [random_ops.random_normal([2,128], 0, 1, tf.float32)]
  x2 = [random_ops.random_normal([2,128], 0, 1, tf.float32)]
  x3 = [random_ops.random_normal([2,128], 0, 1, tf.float32)]
  x= x1+x2+x3
  outs = cm.forward(x, key)
  print(tf.trainable_variables())
  #with tf.train.MonitoredTrainingSession() as sess:
  #  print(sess.run(outs))
  #net = NeuralNetwork('ctr_base', [128, 64, 1])
  #net(tf.constant([[2,1],[2,3]], dtype=tf.float32))
  #print(net.get_config())
  #print(net.get_layers()[0].trainable_variables)
  #net2 = NeuralNetwork('ctr_base', [128, 64, 1])
  #net2(tf.constant([[2,1],[2,3]], dtype=tf.float32))
  #print(net2.get_layers()[0].trainable_variables)
