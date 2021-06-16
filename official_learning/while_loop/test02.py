import tensorflow as tf

tf.enable_eager_execution()

i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: (tf.add(i, 1), )
r = tf.while_loop(c, b, [i])
print(r)