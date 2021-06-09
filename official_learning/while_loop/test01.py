import tensorflow as tf

n = 10000
x = tf.constant(list(range(n)))
c = lambda i, x: i < n
b = lambda i, x: (tf.compat.v1.Print(i + 1, [i]), tf.compat.v1.Print(x + 1,
[i], "x:"))
i, out = tf.while_loop(c, b, (0, x))
with tf.compat.v1.Session() as sess:
    print(sess.run(i))  # prints [0] ... [9999]

    # The following line may increment the counter and x in parallel.
    # The counter thread may get ahead of the other thread, but not the
    # other way around. So you may see things like
    # [9996] x:[9987]
    # meaning that the counter thread is on iteration 9996,
    # while the other thread is on iteration 9987
    print(sess.run(out).shape)