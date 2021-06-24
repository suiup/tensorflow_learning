import tensorflow as tf


a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
c = a + b
c_c = tf.add(a,b, name="c_c")
print("a", a)
print("b", b)

print("c", c)
print("c_c", c_c)

# 加上config 可以看见每一步执行在哪个设备上
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    print(sess.run(c))
    print(sess.run(c_c))
    print(c_c.eval())
    tf.summary.FileWriter("logs/", sess.graph)
    # tensorboard --logdir logs

