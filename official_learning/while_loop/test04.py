import tensorflow as tf

list = []

step = tf.constant(1)

n = tf.constant(2)

def cond(list, step, n):
    return step < 5



def body(step, n):
    step += 1
    n += 1
    return list, step, n

list, step, n = tf.while_loop(cond, body, [list,step, n])

with tf.Session() as sess:
    print(sess.run([step, n]))
