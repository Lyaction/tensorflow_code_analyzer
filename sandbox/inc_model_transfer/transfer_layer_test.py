# 使用前更改 tensorflow/python/training/session_manager.py


import tensorflow as tf
import sys
from transfer_layer import transfer_dense, TransferDense


# global funcs
def get_session(sess):
  session = sess
  while type(session).__name__ != 'Session':
    # pylint: disable=W0212
    session = session._sess
  return session

# 配置
mode = sys.argv[1]
if mode == 'weekly':
    # 周级模型
    ins=2
    version = 'v1:v1:0'
    base_dir = 'weekly'
    sub_dir = 'weekly'
if mode == 'inc':
    # 周级模型基础上加增量特征
    ins=3
    version = 'v1:v2:1'
    base_dir = 'weekly'
    sub_dir = 'inc'
if mode == 'inc2':
    # 增量模型上加增量特征
    ins=4
    version = 'v2:v3:1'
    base_dir = 'inc'
    sub_dir = 'inc2'
if mode == 'inc3':
    # 增量模型上加增量特征
    ins=4
    version = 'v3:v3:0'
    base_dir = 'inc2'
    sub_dir = 'inc3'

# 定义新模型
global_step = tf.train.get_or_create_global_step()
x = tf.placeholder(tf.float32, [None, ins], name='input')
t_dense = TransferDense(4, version)
with tf.variable_scope('part1'):
    y = t_dense(x)
with tf.variable_scope('part2'):
    y = tf.layers.dense(y, 1, activation=None, name='output3')

# 定义 Saver 对象并保存模型参数
checkpoint_dir = 'model/'+base_dir+'/'
#variables_can_be_restored = [i for i in tf.global_variables() if i not in t_dense.no_ckpt_var()]
#print("can be restored", variables_can_be_restored)
#saver = tf.train.Saver(variables_can_be_restored)
saver = tf.train.Saver()

with tf.train.MonitoredTrainingSession(scaffold=t_dense.scaffold(tf.global_variables()), checkpoint_dir=checkpoint_dir) as sess:
    t_dense.after_session(sess)
    # ...训练模型...
    print(tf.trainable_variables())
    print(sess.run(tf.trainable_variables()))
    print(sess.run(global_step))
    saver.save(get_session(sess), 'model/'+sub_dir+'/model1.ckpt')
