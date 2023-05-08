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
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.base_layer import Layer


class FeatureSelection(Layer):
  """Feature selection for sparse inputs.
  """

  def __init__(self,
               embedding,
               kernel_regularizer=None,
               init_min=0.1,
               init_max=0.9,
               **kwargs):
    super(FeatureSelection, self).__init__(**kwargs)
    self.embedding = int(embedding)
    self.initializer = initializers.RandomUniform(minval=0.1, maxval=0.9)
    self.regularizer = regularizers.get(kernel_regularizer)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs should be defined. Found `None`.')
    self.dims = int(tensor_shape.dimension_value(input_shape[-1]) / self.embedding)
    self.weight = self.add_weight('weight',
                                  shape=[self.dims],
                                  initializer=self.initializer,
                                  regularizer=self.regularizer,
                                  trainable=True)
    self._retain_prob = math_ops.sigmoid(self.weight)
    self.scale_prob = array_ops.reshape(gen_array_ops.tile(
                                            array_ops.reshape(self._retain_prob, [-1,1]),
                                            [1,self.embedding]),
                                        [-1])
    self.built = True

  def _dropped_inputs(self, inputs):
    epsion = 10**-6
    scale = 0.1

    noise = random_ops.random_uniform(array_ops.shape(self.weight), minval=0, maxval=1)
    drop_prob = gen_math_ops.log(self._retain_prob + epsion) \
                - gen_math_ops.log(1 - self._retain_prob + epsion) \
                + gen_math_ops.log(noise + epsion) \
                - gen_math_ops.log(1 - noise + epsion)
    self._drop_prob = math_ops.sigmoid(drop_prob / scale)

    random_drop = array_ops.reshape(gen_array_ops.tile(
                                        array_ops.reshape(self._drop_prob, [-1,1]),
                                        [1,self.embedding]),
                                    [-1])
    outputs = inputs * random_drop / self.scale_prob
    return outputs

  def call(self, inputs, training=True):
    outputs = tf_utils.smart_cond(training,
                        lambda: self._dropped_inputs(inputs),
                        lambda: array_ops.identity(inputs))
    return outputs

  @property
  def retain_prob(self):
    return self._retain_prob

  @property
  def drop_prob(self):
    return self._drop_prob if hasattr(self, '_drop_prob') else []


def feature_selection(embedding_dims, inputs, training=True):
  return FeatureSelection(embedding_dims)(inputs, training=training)


if __name__ == '__main__':
    fs = FeatureSelection(2)
    x = random_ops.random_uniform([4, 10])
    y = fs(x, training=False)
    from tensorflow.python.training import monitored_session
    with monitored_session.MonitoredTrainingSession() as sess:
        outs = sess.run([x, fs.trainable_weights, fs.retain_prob, fs.drop_prob, y])
        print("inputs: ", outs[0])
        print("trainable_weights: ", outs[1])
        print("retain_prob: ", outs[2])
        print("drop_prob: ", outs[3])
        print("outputs: ", outs[4])
