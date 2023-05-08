# encoding:utf-8
# Last modified at 2022.08.23
# Authored by chaofeng.gcf
# ========================================================================

"""Mask api for tensorflow models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_parsing_ops

class GenMask:
  """Tool class for MCT model
  """

  def __init__(self, tower_conf_path, max_batch_size=512, max_conv=3, use_weight=False):
    """Init function 
    Arguments:
      tower_conf_path: Path of tower configuration
      max_batch_size: max batch size during training or testing
      max_conv: max convs in conv_info
    Raises:
      ValueError: If convert type is not unique
    """
    if not os.path.exists(tower_conf_path):
      raise ValueError('tower conf file not detected.')
  
    # read configuration
    tower_name_set = set()
    tower_dict = {}
    with open(tower_conf_path, 'rt') as f:
      for line in f:
        line = line.strip()
        if line.startswith('#') or line == '':
          continue
        convert_type, tower_name = line.split(':')
        if convert_type in tower_dict:
            raise ValueError('convert type repeated: [%s]' % convert_type)
        tower_dict[convert_type] = tower_name.split(',')
        for i in tower_name.split(','):
          tower_name_set.add(i)

    # generate hashtable
    one_hot_matrix = np.eye(len(tower_name_set), dtype=int)
    default_val = "|".join(["0"] * len(tower_name_set))
    tower_name_list = sorted(list(tower_name_set), key=lambda x: int(x[1:]))
    tower_index = dict(zip(tower_name_list, range(len(tower_name_list))))
    keys, values = [], []
    for key, value in tower_dict.items():
      keys.append(key)
      convert_mask = np.sum(one_hot_matrix[np.array([tower_index[tower] for tower in value])], 0)
      values.append("|".join(map(str, list(convert_mask))))
    keys_tensor = constant_op.constant(keys)
    values_tensor = constant_op.constant(values, dtypes.string)
    self.mask_table = lookup_ops.HashTable(lookup_ops.KeyValueTensorInitializer(keys, values), default_val)

    # global configuration
    self.max_batch = max_batch_size
    self.max_conv = max_conv
    self.tower_num = len(tower_name_set)
    self.use_weight = use_weight

  def build(self, sess=None):
    """ build inner kv dict
    """
    self.mask_table.init.run(session=sess)

  def eval_mask_label(self, conv_info):
    """ Returns trituple: mask for train towers, mask for labels, weights for towers
    Arguments:
      conv_info: string, conv_info 
    """
    mask_convert_type = gen_string_ops.regex_replace(conv_info, "\|[0-9\.]*", "")
    label_convert_type = gen_string_ops.regex_replace(conv_info,
                                                      ",[0-9\.]*\|0\|[0-9\.]*|[0-9\.]*\|0\|[0-9\.]*,|[0-9\.]*\|0\|[0-9\.]*",
                                                      "")
    label_convert_type = gen_string_ops.regex_replace(label_convert_type, "\|[0-9\.]*", "")
    if self.use_weight:
      weight_convert_type = gen_string_ops.regex_replace(conv_info, "[0-9\.]*\|", "")
      return (self._interal_mask(mask_convert_type),
              self._interal_mask(label_convert_type),
              self._interal_weight(mask_convert_type, weight_convert_type))
    else:
      return (self._interal_mask(mask_convert_type), self._interal_mask(label_convert_type), [])

  def eval_mask_delay_weight(self, delay_weight):
    """ Return each tower delay weight, according to each tower mask convert type
    Arguments:
      delay_wight: string;
      example:
        1|0.1,1000|0.5: convert type:1 -> delay wight:0.1; convert type:1000 -> delay wight:0.5
    """
    mask_convert_type = gen_string_ops.regex_replace(delay_weight, "\|[0-9\.]*", "")
    weight_convert_type = gen_string_ops.regex_replace(delay_weight, "[0-9\.]*\|", "")
    return self._interal_weight(mask_convert_type, weight_convert_type)

  def _interal_mask(self, convert_type):
    """ Returns mask from inner kv dict
    Arguments:
      convert_type: array-like,  shape `(n_samples, n_convs)`
    """
    multi_convert_type = string_ops.string_split(convert_type, ",")
    sparse_mask = self.mask_table.lookup(multi_convert_type)
    mask_shape = clip_ops.clip_by_value(sparse_mask.dense_shape,
                                        [1, self.max_conv],
                                        [self.max_batch, self.max_conv])
    mask_padding = array_ops.reshape(sparse_ops.sparse_to_dense(sparse_mask.indices,
                                                                mask_shape,
                                                                sparse_mask.values,
                                                                "|".join(self.tower_num*["0"])), [-1])
    mask_tower_split=string_ops.string_split(mask_padding, "|")
    mask = math_ops.reduce_sum(array_ops.reshape(gen_parsing_ops.string_to_number(mask_tower_split.values,
                                                                                  out_type=dtypes.float32),
                                                 [-1, self.max_conv, self.tower_num]), axis=1)
    return math_ops.cast(math_ops.cast(mask, dtypes.bool), dtypes.float32)

  def _interal_weight(self, convert_type, weights):
    """ returns weight for towers
    arguments:
      weights: array-like,  shape `(n_samples, n_convs)`
    """

    def each_conv_list(sub_convert_info):
      """ preprocess
      arguments:
        sub_convert_info
      """
      multi_info = string_ops.string_split(sub_convert_info, ",")
      multi_info_shape = clip_ops.clip_by_value(multi_info.dense_shape,
                                          [1, self.max_conv],
                                          [self.max_batch, self.max_conv])
      return array_ops.transpose(sparse_ops.sparse_to_dense(multi_info.indices,
                                                            multi_info_shape,
                                                            multi_info.values,
                                                            "0"))

    convert_type_list, weight_list = each_conv_list(convert_type), each_conv_list(weights)
    return math_ops.reduce_sum([self._interal_mask(convert_type_list[i]) *
                                gen_parsing_ops.string_to_number(array_ops.reshape(weight_list[i], [-1, 1]),
                                                                 dtypes.float32)
                                for i in range(self.max_conv)], axis=0)

