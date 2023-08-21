import tensorflow as tf

x = tf.constant([1,2])

sess = tf.Session()

x.eval(session=sess)

