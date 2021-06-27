import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data


"""
用全连接手写数据集识别案例
    Mnist 数据集可以去官网下载 
    http://yann.lecun.com/exdb/mnist




"""

# 准备数据
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)


x = tf.placeholder(dtype=tf.float32, shape=[None,784])
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# 构建模型
Weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]))
bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))
y_predict = tf.matmul(x, Weights) + bias

#构造损失函数
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

# 优化损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    image, label = mnist.train.next_batch(100)
    print("before training error %f " % sess.run(error, feed_dict={x: image, y_true:label}))
    # 开始训练
    for i in range(100):
        _, loss = sess.run([optimizer, error], feed_dict={x:image, y_true: label})
        print("第%d次的训练损失为%f" % (i + 1, loss))

# print(mnist.train.next_batch(1))

# print(mnist.train.images[1])
# print(mnist.train.labels[0])