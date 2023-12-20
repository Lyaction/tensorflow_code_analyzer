import tensorflow as tf

def get_session(sess):
  session = sess
  while type(session).__name__ != 'Session':
    # pylint: disable=W0212
    session = session._sess
  return session

# 定义原模型
global_step = tf.train.get_or_create_global_step()
x = tf.placeholder(tf.float32, [None, 10], name='input')
fixed_kernel = tf.get_variable("kernel", [10, 8])
x = tf.matmul(x, fixed_kernel)
x = tf.layers.dense(x, 8, activation=None, name='output1')
x = tf.layers.dense(x, 4, activation=None, name='output2')
x = tf.layers.dense(x, 1, activation=None, name='output3')

# 定义 Saver 对象并保存模型参数
saver1 = tf.train.Saver()
with tf.train.MonitoredTrainingSession() as sess:
    # ...训练模型...
    print(tf.trainable_variables())
    print(sess.run(tf.trainable_variables()))
    print(sess.run(global_step))
    saver1.save(get_session(sess), 'model/summary/model.ckpt')
