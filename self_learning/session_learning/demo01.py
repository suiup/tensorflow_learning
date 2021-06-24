import tensorflow as tf


a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
c = tf.add(a,b, name="c")
print("a", a)
print("b", b)
print("c", c)


# 加上config 可以看见每一步执行在哪个设备上
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    abc = sess.run([a,b,c])
    a,b,c = sess.run([a,b,c])
    print("abc", abc)
    print("abc", a,b,c)
    tf.summary.FileWriter("logs/", sess.graph)
    # tensorboard --logdir logs

