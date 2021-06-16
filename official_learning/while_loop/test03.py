import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)
c = tf.constant(3)


def cond(a, b, c):
    return a < 6


def body(a, b, c):
    a += 1
    b += 1
    c += 1
    return a, b, c  # same with [a, b, c]


a, b, c = tf.while_loop(cond, body, [a, b, c])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([a, b, c]))
    print(sess.run([a]))
