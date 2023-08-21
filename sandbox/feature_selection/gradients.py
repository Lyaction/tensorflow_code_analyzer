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
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import gradients_impl
from tensorflow.python.training import monitored_session
import tensorflow as tf
import numpy as np


epsion = 10**-7
#retain_prob = random_ops.random_uniform([10], minval=0.8, maxval=0.8) 
#noise = random_ops.random_uniform(retain_prob.shape, minval=0, maxval=1)
retain_prob = tf.constant(np.arange(0.01 ,1 ,0.01), tf.float32)
noise = random_ops.random_uniform(retain_prob.shape, minval=0.7, maxval=0.7)
drop_prob = gen_math_ops.log(retain_prob + epsion) \
            - gen_math_ops.log(1 - retain_prob + epsion) \
            + gen_math_ops.log(noise + epsion) \
            - gen_math_ops.log(1 - noise + epsion)
drop_prob = math_ops.sigmoid(drop_prob / 0.1)
y = drop_prob / retain_prob

g = gradients_impl.gradients(drop_prob, [retain_prob])
g2 = gradients_impl.gradients(y, [retain_prob])
g3 = gradients_impl.gradients(drop_prob / tf.stop_gradient(retain_prob), [retain_prob])

with monitored_session.MonitoredTrainingSession() as sess:
    outs = sess.run([retain_prob, noise, drop_prob, g, g2, g3])
    print("retain_prob: ", outs[0])
    print("noise: ", outs[1])
    print("drop_prob: ", outs[2])
    print("g: ", outs[3])
    print("g2: ", outs[4])
    print("g3: ", outs[5])
