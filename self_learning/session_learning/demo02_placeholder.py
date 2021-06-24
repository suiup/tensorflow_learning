import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = tf.add(a, b)

print("a", a)
print("b", b)
print("c", c)

with tf.Session() as sess:
    sum = sess.run(c, feed_dict={a: 3, b: 4})
    print("sum: ", sum)