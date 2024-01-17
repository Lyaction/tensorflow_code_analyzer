import tensorflow as tf

# 定义一个自定义的初始化器
def custom_initializer(shape, dtype=None, partition_info=None):
    # 根据shape创建一个自定义的初始值
    print(partition_info)
    init_value = tf.constant(0.1, shape=shape, dtype=dtype)
    return init_value

# 创建一个变量，并使用自定义初始化器进行初始化
my_variable = tf.get_variable("my_variable", shape=[2, 3], initializer=custom_initializer)
initilizer = tf.initializers.he_normal()
#out = initilizer()

# 在会话中运行并打印变量的值
with tf.train.MonitoredTrainingSession() as sess:
    print(sess.run(my_variable))
