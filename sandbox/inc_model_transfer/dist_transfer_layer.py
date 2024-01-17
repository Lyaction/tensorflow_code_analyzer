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
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import resources
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.training import monitored_session
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.python.training import saver as saver_


TRANSFER_FLAG = "special_transfer"
TRANSFER_FUNCTION = "special_transfer_function"

def _check_version(version):
  # v1:v2:10 v1:v1:0
  base, inc, inc_dims = version.split(":")
  inc_dims = int(inc_dims)
  if base == inc:
    assert 0 == inc_dims
    return base, inc, inc_dims
  assert 0 < inc_dims
  return base, inc, inc_dims

def _check_exit(flag, ckpt):
  vars_map = checkpoint_utils.load_checkpoint(ckpt).get_variable_to_shape_map()
  for var in vars_map:
    if flag in var:
      return True
  return False

class TransferDense(Layer):
  def __init__(self,
               units,
               version,
               unique_name='',
               activation='relu',
               use_bias=True,
               ckpt=None,
               **kwargs):
    super(TransferDense, self).__init__(**kwargs)
    self.units = units
    self.unique_name = unique_name
    self.use_bias = use_bias
    self.activations = activations.get(activation)
    self.base, self.inc, self.inc_dims = _check_version(version)
    self.init_op = constant_op.constant([], name='placehold')
    self.ckpt = ckpt

    self.inc_kernel_name = self.unique_name + TRANSFER_FLAG + self.inc
    self.base_kernel_name = self.unique_name + TRANSFER_FLAG + self.base
    print(locals())

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs should be defined. Found `None`.')
    self.dims = tensor_shape.dimension_value(input_shape[-1])
    self.weight = self.add_weight(self.inc_kernel_name,
                                  shape=[self.dims, self.units],
                                  initializer=init_ops.glorot_uniform_initializer(),
                                  trainable=True)
    # if not inc mode
    if self.inc_dims:
      if _check_exit(self.inc_kernel_name, self.ckpt):
        self.inc_dims = False
    ops.add_to_collection(TRANSFER_FUNCTION, self.after_session)

    if self.inc_dims:
      base_dims = self.dims-self.inc_dims
      self.base_weight = self.add_weight(self.base_kernel_name,
                                         shape=[base_dims, self.units],
                                         initializer=init_ops.glorot_uniform_initializer(),
                                         trainable=True)
      self.init_op = variables.variables_initializer([self.weight])
      self.assign_op = state_ops.assign(self.weight[:base_dims], self.base_weight)
      ops.add_to_collection(TRANSFER_FLAG, self.inc_kernel_name)

    if self.use_bias:
      self.bias = self.add_weight('bias',
                                  shape=[self.units],
                                  initializer=init_ops.zeros_initializer(),
                                  trainable=True)

    self.built = True

  def after_session(self, sess, is_chief):
    if is_chief:
      ops.get_default_graph()._unsafe_unfinalize()
      if self.inc_dims:
        sess.run(self.init_op)
        sess.run(self.assign_op)
        # if Adagrad opt
        inc_adagrad = [i for i in variables.global_variables() if self.inc_kernel_name in i.name and "Adagrad" in i.name]
        print(inc_adagrad)
        assert 1 >= len(inc_adagrad)
        if 1 == len(inc_adagrad):
          self.adagrad_assign = state_ops.assign(inc_adagrad[0], array_ops.ones_like(self.weight)*0.05)
          sess.run(self.adagrad_assign)
      ops.get_default_graph().finalize()

  def is_inc(self):
    return self.inc_dims

  def call(self, inputs):
    out_puts = gen_math_ops.mat_mul(inputs, self.weight)
    if self.use_bias:
      out_puts += self.bias
    out_puts = self.activations(out_puts)
    return out_puts

def transfer_dense(inputs, units, version, activation='relu', use_bias=True):
  return TransferDense(units, use_bias=use_bias, version=version, activation=activation)(inputs)

def wait_for_session(ready, is_chief):
    ready = array_ops.concat([variables.report_uninitialized_variables(),
                              resources.report_uninitialized_resources()], 0)
    init_op = control_flow_ops.group(variables.global_variables_initializer(),
                                     resources.initialize_resources(resources.shared_resources()))
    ready_for_local = array_ops.concat([variables.report_uninitialized_variables(variables.global_variables()),
                                        resources.report_uninitialized_resources(resources.shared_resources())
                                        ], 0)
    init_op_for_local = control_flow_ops.group(variables.local_variables_initializer(),
                                               lookup_ops.tables_initializer(),
                                               resources.initialize_resources(resources.local_resources()))

def transfer_variables():
  transfer_vars = []
  for var in variables.global_variables():
    is_trans = True
    for key in ops.get_collection(TRANSFER_FLAG):
      if key in var.name: is_trans = False
    if is_trans:
      transfer_vars.append(var)
    else:
      print(var)
  return transfer_vars
