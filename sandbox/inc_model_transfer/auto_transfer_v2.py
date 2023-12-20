# 使用前更改 tensorflow/python/training/session_manager.py


import tensorflow as tf
import sys


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
    raw_version = '0'
    version = '1'
    raw_fixed_ins = 8
    raw_add_ins = 2
    new_add_ins = 2
    is_inc = False
    base_dir = 'weekly'
    sub_dir = 'weekly'
if mode == 'inc':
    # 周级模型基础上加增量特征
    raw_version = '1'
    version = '2'
    raw_fixed_ins = 10
    raw_add_ins = 2
    new_add_ins = 2
    is_inc = True
    base_dir = 'weekly'
    sub_dir = 'inc'
if mode == 'inc2':
    # 增量模型上加增量特征
    raw_version = '2'
    version = '3'
    raw_fixed_ins = 12
    raw_add_ins = 2
    new_add_ins = 2
    is_inc = True
    base_dir = 'inc'
    sub_dir = 'inc2'

#迁移封装
class layer_transfer():
    def __init__(self):
        new_fixed_ins = raw_fixed_ins + raw_add_ins
        self.raw_fixed_kernel = tf.get_variable("fixed_kernel"+raw_version, [raw_fixed_ins, 8])
        self.raw_add_kernel = tf.get_variable("add_kernel_"+raw_version, [raw_add_ins, 8])
        self.new_fixed_kernel = tf.get_variable("fixed_kernel"+version, [new_fixed_ins, 8])
        self.new_add_kernel = tf.get_variable("add_kernel_"+version, [new_add_ins, 8])
        self.kernel = tf.concat([self.new_fixed_kernel, self.new_add_kernel], 0)
        self.init_op = tf.assign(self.new_fixed_kernel, tf.concat([self.raw_fixed_kernel, self.raw_add_kernel], 0))

    def __call__(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def after_session(self, session):
        sess.run(self.extra_init_op)
        sess.run(self.init_op)

    def scaffold(self):
        if not is_inc: return None
        var_list = [v for v in tf.trainable_variables() if v not in [self.new_fixed_kernel, self.new_add_kernel]]
        var_list += [v for v in tf.global_variables() if v.name.startswith('global_step')]
        init_list = [self.new_fixed_kernel, self.new_add_kernel]
        print(var_list)
        print(init_list)
        saver = tf.train.Saver(var_list=var_list)
        self.extra_init_op = tf.variables_initializer(init_list)
        ready_op = tf.constant([], name='ready_op')
        ready_for_local_init_op = tf.constant([], name='ready_for_local_init_op')
        return tf.train.Scaffold(saver=saver, ready_op=ready_op, ready_for_local_init_op=ready_for_local_init_op)


# 定义新模型
global_step = tf.train.get_or_create_global_step()
x = tf.placeholder(tf.float32, [None, raw_fixed_ins+raw_add_ins+new_add_ins], name='input')
with tf.variable_scope('part1'):
    transfer = layer_transfer()
    x = transfer(x)
with tf.variable_scope('part2'):
    x = tf.layers.dense(x, 8, activation=None, name='output1')
    x = tf.layers.dense(x, 4, activation=None, name='output2')
    x = tf.layers.dense(x, 1, activation=None, name='output3')

# 定义 Saver 对象并保存模型参数
saver1 = tf.train.Saver()

with tf.train.MonitoredTrainingSession(scaffold=transfer.scaffold(), checkpoint_dir='model/'+base_dir+'/') as sess:
    transfer.after_session(sess)
    # ...训练模型...
    print(tf.trainable_variables())
    print(sess.run(tf.trainable_variables()))
    print(sess.run(global_step))
    saver1.save(get_session(sess), 'model/'+sub_dir+'/model1.ckpt')
