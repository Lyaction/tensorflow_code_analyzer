# encoding:utf-8
"""The API detail
NOTE:
author by chaofeng.gcf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.summary import summary as summary_lib
from flag_define import *

class Summary(object):

  monitor={'scalar': [], 'histogram': []}

  @classmethod
  def add_tensor_histogram(cls, name, tensor):
    if FLAGS.summary_trainabel_var:
      summary_lib.histogram(name, tensor)

  @classmethod
  def add_scalar(cls, name, scalar):
    if FLAGS.summary_scalar:
      summary_lib.scalar(name, scalar)

  @classmethod
  def add_layer_histogram(cls, name, layer):
    if FLAGS.summary_trainabel_var:
      if layer.kernel:
        summary_lib.histogram(name+"_kernel", layer.kernel)
      if layer.bias:
        summary_lib.histogram(name+"_bias", layer.bias)

  @classmethod
  def add_summary(cls, name, tensor, mtype='scalar', grads=False):
    if mtype in monitor:
      monitor[mtype].append([name, tensor, grads])

  @classmethod
  def final(cls, loss):
    for item in monitor['scalar']:
      summary_lib.scalar(item[0], item[1])


