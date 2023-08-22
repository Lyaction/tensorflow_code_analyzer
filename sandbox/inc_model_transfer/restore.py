import tensorflow as tf
import numpy as np

# 定义新的模型
global_step = tf.train.get_or_create_global_step()
x = tf.placeholder(tf.float32, [None, 12], name='input')
fixed_kernel = tf.get_variable("kernel", [10, 8])
add_kernel = tf.get_variable("add_kernel", [2, 8])
x = tf.matmul(x, tf.concat([fixed_kernel, add_kernel], 0))
x = tf.layers.dense(x, 8, activation=None, name='output1')
x = tf.layers.dense(x, 4, activation=None, name='output2')
x = tf.layers.dense(x, 1, activation=None, name='output3')
x = tf.layers.dense(x, 1, activation=None, name='output4')

var_list = [v for v in tf.trainable_variables() if not v.name.startswith('output4') and not v.name.startswith('add_kernel')]
var_list += [v for v in tf.global_variables() if v.name.startswith('global_step')]
print(var_list)
init_list = [v for v in tf.trainable_variables() if v.name.startswith('output4') or v.name.startswith('add_kernel')]
print(init_list)
saver = tf.train.Saver(var_list=var_list)
init_op = tf.variables_initializer(init_list)
init_feed_dict = {'output4/kernel:0':np.array([[2.1]], np.float32), 'output4/bias:0':np.array([4.1], np.float32)}
#ready_for_local_init_op = tf.report_uninitialized_variables(init_list)
scaffold = tf.train.Scaffold(saver=saver, init_op=init_op)
with tf.train.MonitoredTrainingSession(scaffold=scaffold, checkpoint_dir='model/') as sess:
    
    print(sess.run(tf.trainable_variables()))
