# encoding:utf-8
# Authered by: chaofeng.gcf
#=============================================
"""The API detail
NOTE: outer_table
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import tensorflow as tf
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.feature_column import feature_column
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops

from tensorflow.python import keras


def gen_outer_table(outer_conf, ps_num,
                    sparse_features_num,
                    max_feature_num=10000,
                    embedding_dim=14,
                    mode='train'):
  if not os.path.exists(outer_conf):
    raise ValueError('Outer table conf not exist.')
  fp = open(outer_conf, 'r')
  content = fp.readlines()
  fp.close()
  real_all_slots_name, real_train_slots_index = [], []
  outer_dict = {}
  for line in content:
    if line.startswith('#') or line.strip()=='':
      continue
    fields = line.strip().split(' ')
    if len(fields) == 4:
      key_name = fields[0].strip()
      value_slots = fields[1].strip().split(',')
      seq_len = int(fields[2].strip())
      emb_len = int(fields[3].strip())
      if (key_name == "") or (len(value_slots) == 0) or (seq_len == 0) or (emb_len == 0):
        raise ValueError('Outer table conf wrong line detected: {}'.format(line.strip()))
      outer_dict[key_name] = (len(value_slots), seq_len, value_slots, emb_len)
    elif len(fields) == 2:
      real_all_slots_name = fields[1].split(',')
    else:
      raise ValueError('Outer table conf wrong line detected: {}'.format(line.strip()))
  for key, info in sorted(outer_dict.items(), key=lambda x: x[0]):
    for slot in info[2]:
      slot_name = key + '_slot_' + slot
      if slot_name not in real_all_slots_name:
        raise ValueError('outer table slot conf error: {} not in {}.'.format(slot, real_all_slots_name))
      real_train_slots_index.append(real_all_slots_name.index(slot_name))
  trainable = (mode == 'train')
  return _OuterTable(outer_dict=outer_dict,
                     sparse_features_num=sparse_features_num,
                     max_feature_num=max_feature_num,
                     embedding_dim=embedding_dim,
                     ps_num=ps_num,
                     real_all_slots_name=real_all_slots_name,
                     real_train_slots_index=real_train_slots_index,
                     trainable=trainable)
  
  
class _OuterTable(
    collections.namedtuple(
    '_OuterTable',
    ('outer_dict', 'sparse_features_num', 'max_feature_num', 'embedding_dim', 'ps_num', 'real_all_slots_name', 'real_train_slots_index', 'trainable'))):
  '''
  outer_dict:
  slot_num:
  trainable:
  '''

  @property
  def slot_num(self):
    return sum(item[0] for item in self.outer_dict.values())

  @property
  def real_slots_name(self):
    return self.real_all_slots_name

  @property
  def start_index(self):
    if self.sparse_features_num < self.slot_num:
      raise ValueError('sparse feas num: {} lower than outer table slot num: {} '.format(self.sparse_features_num, self.slot_num))
    return self.sparse_features_num-self.slot_num

  @property
  def slot_index(self):
    return self.real_train_slots_index

  @property
  def input_dim(self):
    return self.slot_num * self.embedding_dim

  @property
  def output_dim(self):
    return sum([item[3] for item in self.outer_dict.values()])

  @property
  def outer_conf(self):
    '''
    dump conf, will be changed in future version
    '''
    return 'user_emb:' + str(self.output_dim) + ':2' + "\n"

  def _gen_mask(self, sp_input):
    '''Computes the mask of sparse tensor
    '''
    ones_indice = tf.SparseTensor(indices=sp_input.indices,
                                  values=tf.ones(tf.shape(sp_input.values), dtype=tf.int64),
                                  dense_shape =sp_input.dense_shape)
    return tf.sparse.reduce_sum(ones_indice, axis=1, keepdims=False)

  def parse_outer_ins(self, inputs):
    '''
    Args:
      inputs: list of tensor
      start_index: column name index
    '''
    self.slots_mask = []
    self.outer_sparse_ins = {}
    if len(inputs) != self.slot_num:
      raise ValueError('Outer table input slot num {} not equal conf slot num {}'.format(len(inputs), self.slot_num))
    column_names = [str(self.max_feature_num+i) for i in range(self.start_index, self.start_index+self.slot_num)]
    seq_len = []
    for key, info in sorted(self.outer_dict.items(), key=lambda x: x[0]):
      seq_len += [info[1]] * info[0]
    for column_name, item_len, slot_seq in zip(column_names, seq_len, inputs):
      print("outer input info: ", column_name, item_len)
      slot_fea = tf.string_split(slot_seq, "|", False)
      slot_shape = tf.clip_by_value(slot_fea.dense_shape, [1, item_len], [512, item_len])
      total_string = tf.reshape(tf.sparse_to_dense(slot_fea.indices, slot_shape, slot_fea.values, ""), [-1])
      sparse_string = tf.string_split(total_string, ",")
      sparse_value = tf.string_to_number(sparse_string.values, out_type=tf.int64)
      
      self.slots_mask.append(self._gen_mask(slot_fea))
      self.outer_sparse_ins[column_name] = tf.SparseTensor(sparse_string.indices, sparse_value, sparse_string.dense_shape)

  def split_columns(self, params):
    if 'sparse_columns' not in params:
      raise ValueError('params has not sparse_columns !')
    sparse_columns = params['sparse_columns']
    self.outer_masks = {}
    self.outer_sparse_columns = {}
    if len(sparse_columns) < self.slot_num:
      raise ValueError('total columns {} lower than outer slots num {} !'.format(len(sparse_columns), self.slot_num))
    params['sparse_columns'], outer_columns = sparse_columns[:0-self.slot_num], sparse_columns[0-self.slot_num:]
    start = 0
    for key, info in sorted(self.outer_dict.items(), key=lambda x: x[0]):
      end = start + info[0]
      self.outer_masks[key]=self.slots_mask[start]
      self.outer_sparse_columns[key] = feature_column.InputLayer(outer_columns[start:end], name=key)
      start = end

  def build_outer_table_layer(self, outer_input):
    if not isinstance(outer_input, dict):
      raise ValueError('outer input should be dict type ! Given: {}'.format(self.outer_sparse_columns))
    with tf.variable_scope('outer_layer',
                           partitioner = partitioned_variables.min_max_variable_partitioner(
                               max_partitions=self.ps_num, min_slice_size=64*1024),
                           initializer = tf.keras.initializers.he_uniform()):
      # struct 1
      output_list = [tf.reduce_sum(action_seq, 1) for key, action_seq in sorted(outer_input.items(), key=lambda x: x[0])]
      outer_table_input=tf.concat(output_list, 1)
      mid = tf.layers.dense(outer_table_input, units=128, use_bias=False, activation=tf.nn.relu,
                            kernel_initializer=tf.glorot_uniform_initializer())
      outer_table_output = tf.layers.dense(mid, units=64, use_bias=False, activation=tf.nn.relu,
                                           kernel_initializer=tf.glorot_uniform_initializer())
    return outer_table_output

      # struct 2
      #for (key, action_seq), dim in zip(list(sorted(outer_input.items(), key=lambda x: x[0])), [32, 16, 32]):
      #  outputs, _ = dynamic_rnn(cell=GRUCell(num_units=dim, name='gru_'+key),
      #                           sequence_length=self.outer_masks[key],
      #                           inputs=action_seq,
      #                           dtype=tf.float32,
      #                           parallel_iterations=None)
      #  outer_outputs.append(tf.reduce_sum(outputs, 1))

      # struct 3
      #outer_outputs, seq_len, emb_len = [], [], []
      #for key, info in sorted(self.outer_dict.items(), key=lambda x: x[0]):
      #  seq_len += [info[1]]
      #  emb_len +=[info[3]]
      #for (key, action_seq), max_len, dim in zip(list(sorted(outer_input.items(), key=lambda x: x[0])), seq_len, emb_len):
      #  att_ins = core_layers.dense(action_seq,
      #                              units=dim,
      #                              use_bias=False,
      #                              kernel_initializer=tf.glorot_uniform_initializer(),
      #                              name='reduce_'+key)
      #  mid1 = core_layers.dense(att_ins,
      #                           units=dim/2,
      #                           activation=None,
      #                           use_bias=False,
      #                           kernel_initializer=tf.glorot_uniform_initializer(),
      #                           name='attention_'+key+'_1')
      #  mid2 = core_layers.dense(mid1,
      #                           units=1,
      #                           activation=tf.nn.sigmoid,
      #                           use_bias=False,
      #                           kernel_initializer=tf.glorot_uniform_initializer(),
      #                           name='attention_'+key+'_2')
      #  mask = array_ops.reshape(array_ops.sequence_mask(self.outer_masks[key],
      #                                                   max_len,
      #                                                   dtype=dtypes.float32),
      #                           [-1, max_len, 1])
      #  outer_outputs.append(math_ops.reduce_sum(mid2 * mask * att_ins, 1))

      # struct 4
      #outer_outputs, seq_len = [], []
      #for key, info in sorted(self.outer_dict.items(), key=lambda x: x[0]):
      #  seq_len += [info[1]]
      #  emb_len +=[info[3]]
      #for (key, action_seq), max_len, dim in zip(list(sorted(outer_input.items(), key=lambda x: x[0])), seq_len, emb_len):
      #  gru_out = keras.layers.GRU(units=dim, return_sequences=True, unroll=True, name='gru_'+key)(action_seq)
      #  mask = array_ops.reshape(array_ops.sequence_mask(self.outer_masks[key],
      #                                                   max_len,
      #                                                   dtype=dtypes.float32),
      #                           [-1, max_len, 1])
      #  outer_outputs.append(math_ops.reduce_sum(mask * gru_out, 1))
    #return array_ops.concat(outer_outputs, 1)

  def outer_table_fn(self):
    '''
    have to initialize outer_sparse_ins and outer_sparse_columns before call
    '''
    if not isinstance(self.outer_sparse_columns, dict):
      raise ValueError('outer sparse columns should be dict type ! Given: {}'.format(self.outer_sparse_columns))
    if not isinstance(self.outer_sparse_ins, dict):
      raise ValueError('outer sparse ins should be dict type ! Given: {}'.format(self.outer_sparse_ins))
    if any(not isinstance(tensor, tf.SparseTensor) for tensor in self.outer_sparse_ins.values()):
      raise ValueError('outer sparse ins dict value should be SparseTensor !')
    #for key, columns in self.outer_sparse_columns.items():
    #  if any(not isinstance(column, tf.feature_column._EmbeddingColumn) for column in columns):
    #    raise ValueError('Items of outer feature columns must be a _DenseColumn !')
    
    outer_input={}
    with tf.variable_scope("outer_input", reuse=tf.AUTO_REUSE):
      for key, columns in sorted(self.outer_sparse_columns.items(), key=lambda x: x[0]):
        feas_num, seq_len = self.outer_dict[key][:2]
        # input_dense = tf.feature_column.input_layer(self.outer_sparse_ins, columns, trainable=self.trainable)
        input_dense = columns(self.outer_sparse_ins)
        outer_input[key] = tf.reshape(input_dense, [-1, seq_len, self.embedding_dim*feas_num])
    return self.build_outer_table_layer(outer_input)

  def gen_outer_emb(self, input_string, params):
    '''
    get user outer embedding
    '''
    record_defaults = [['']] + [['']] * self.slot_num
    records = tf.decode_csv(input_string, use_quote_delim=False, record_defaults = record_defaults, field_delim = ';')
    dmpid_sign = records[0]
    self.parse_outer_ins(records[1:])
    self.split_columns(params)
    with tf.variable_scope('input',
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                           reuse=tf.AUTO_REUSE):
      outer_emb = self.outer_table_fn()
    return {'run_ops': [dmpid_sign, outer_emb]}

  def _print(self, debug_tensors):
    with tf.train.MonitoredTrainingSession() as sess:
        print('debug_result:', sess.run(debug_tensors))
        import sys
        sys.exit()
