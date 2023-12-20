# encoding:utf-8
# Authered by: chaofeng.gcf
#=============================================
"""The API detail
NOTE: concrete dropout
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_


def _check_version(version):
  # v1:v2:10 v1:v1:0
  base, inc, inc_dims = version.split(":")
  inc_dims = int(inc_dims)
  if base == inc:
    assert 0 == inc_dims
    return base, inc, inc_dims
  assert 0 < inc_dims
  return base, inc, inc_dims


class TransferDense(Layer):
  def __init__(self,
               units,
               version,
               activation='relu',
               use_bias=True,
               **kwargs):
    super(TransferDense, self).__init__(**kwargs)
    self.units = units
    self.use_bias = use_bias
    self.activations = activations.get(activation)
    self.base, self.inc, self.inc_dims = _check_version(version)
    self.init_op = constant_op.constant([], name='placehold')
    self.can_not_be_restored = []
    print(locals())

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs should be defined. Found `None`.')
    self.dims = tensor_shape.dimension_value(input_shape[-1])
    self.weight = self.add_weight(self.inc,
                                  shape=[self.dims, self.units],
                                  initializer=init_ops.glorot_uniform_initializer(),
                                  trainable=True)
    if self.inc_dims:
      base_dims = self.dims-self.inc_dims
      self.base_weight = self.add_weight(self.base,
                                         shape=[base_dims, self.units],
                                         initializer=init_ops.glorot_uniform_initializer(),
                                         trainable=True)
      self.init_op = variables.variables_initializer([self.weight])
      self.assign_op = state_ops.assign(self.weight[:base_dims], self.base_weight)
      self.can_not_be_restored = [self.weight]

    if self.use_bias:
      self.bias = self.add_weight('bias',
                                  shape=[self.units],
                                  initializer=init_ops.zeros_initializer(),
                                  trainable=True)

    self.built = True

  def scaffold(self, global_variables):
    ready_op = constant_op.constant([], name='ready_op')
    ready_for_local_init_op = constant_op.constant([], name='ready_for_local_init_op')
    scaffold = None
    if self.inc_dims:
      saver = saver_.Saver([i for i in global_variables if i not in self.can_not_be_restored])
      scaffold = monitored_session.Scaffold(saver=saver,
                                            ready_op=ready_op,
                                            ready_for_local_init_op=ready_for_local_init_op)
    return scaffold

  def after_session(self, sess):
    if self.inc_dims:
      sess.run(self.init_op)
      sess.run(self.assign_op)

  def call(self, inputs):
    out_puts = gen_math_ops.mat_mul(inputs, self.weight)
    if self.use_bias:
      out_puts += self.bias
    out_puts = self.activations(out_puts)
    return out_puts

def transfer_dense(inputs, units, version, activation='relu', use_bias=True):
  return TransferDense(units, use_bias=use_bias, version=version, activation=activation)(inputs)
