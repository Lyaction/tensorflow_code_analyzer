# encoding:utf-8
# Authored by: chaofeng.gcf
# =============================================
"""The API detail
NOTE: customize model plugin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from functools import reduce

from tensorflow.python.ops import variables
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import initializers
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.layers import Dense


class CustomModel:
  def __init__(self,
               input_len=128
               task_num=3,
               hidden_units=[64, 1]
               use_bias=[True, False]
               kernel_initializer='he_uniform',
               split_parts=10,
               filter_freq=0):
    assert len(use_bias) == len(hidden_units)

    self.input_len = input_len
    self.task_num = task_num
    self.hidden_units = hidden_units
    self.use_bias = list(map(lambda i:int(i), use_bias))
    self.kernel_initializer = initializers.get(kernel_initializer)
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
                                              initializer = self._cm_initializer(hidden_units))

  def _compute_fans(self, shape):
    if len(shape) < 1:
      fan_in = fan_out = 1
    elif len(shape) == 1:
      fan_in = fan_out = shape[0]
    elif len(shape) == 2:
      fan_in = shape[0]
      fan_out = shape[1]
    else:
      raise ValueError('cm shape error.')
    return fan_in, fan_out

  def _cm_initializer(self, shape, dtype=None, partition_info=None):
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape
    fan_in, fan_out = _compute_fans(scale_shape)
    assert fan_in == fan_out and fan_in == self.params
    elements = []
    for kernel_shape, bias_param in zip(self.kernel_shape, self.bias_params):
      kernel = array_ops.reshape(self.kernel_initializer(kernel_shape, dtype=dtype))
      if bias_param:
        bias = array_ops.zeros(bias_param, dtype)
        elements.append(array_ops.concat([kernel, bias], axis=0))
      else:
        elements.append(kernel)
    params = array_ops.concat(elements, axis=0)
    return params

  def cmu(self, param, x):
    '''
    Inner computation graph which can be used to infer
    '''
    outputs = []
    x = array_ops.reshape(gen_array_ops.stop_gradient(x), [-1, 1, self.input_len])
    for task in array_ops.split(param, self.task_params, axis=1)
      task_x = x
      for kernel_shape, bias_param, layer in zip(self.kernel_shape,
                                                 self.bias_params,
                                                 array_ops.split(task, self.layer_params, axis=1))
        kernel_param = reduce(lambda i,j: i*j, kernel_shape)
        raw_kernel, bias = array_ops.split(layer, [kernel_param, bias_param], aixs=1)
        kernel = array_ops.reshape(raw_kernel, [-1, kernel_shape])
        task_x = math_ops.matmul(task_x, kernel)
        if use_bias:
          task_x += array_ops.reshape(bias, [-1, 1, bias_param])
      task_x = array_ops.reshape(task_x, [-1, self.hidden_units])
      outputs.append(task_x)
    return outputs

  def forward(self, x, key):
    '''
    Process inputs before training.
    '''
    cm_param = feature_column_ops.input_from_feature_columns(key, self.sparse_embedding_column)
    preds = self.cmu(cm_param, x)
    assert len(preds) = self.task_num
    return preds


class NeuralNetwork:
  '''
  1. 子网络统一构建、监控、拷贝
  '''
  def __init__(self, config, name, **kwargs):
    self.dense_list = []
    self.outputs = []
    self.partitioner = None

    if 'partitioner' in kwargs:
      self.partitioner = kwargs['partitioner']
    with variable_scope.variable_scope(name,
                                       partitioner = self.partitioner,
                                       reuse=variable_scope.AUTO_REUSE):
      for item in config:
        dense = Dense(item,
                      activation='relu',
                      use_bias=True,
                      kernel_initializer='glorot_uniform',
                      bias_initializer='zeros')
        self.dense_list.append(dense)

  def __call__(self, x):
    for dense in self.dense_list:
      x = dense(x)
      self.outputs.append(x)
    return self.outputs[-1]

  def get_layers(self):
    return self.dense_list

  def get_outputs(self):
    return self.outputs

def neural_network(x, config, name):
  return NeuralNetwork(config, name)(x)
