import tensorflow as tf

"""
线性回归

构建模型： y = w1x1 + w2x2 + ... + wnxn + b
构造损失函数 loss function: 均方误差
优化损失： 梯度下降

当梯度下降最小的时候，使得损失函数比较小的时候，得到的权重和偏秩就是模型的参数

"""

"""
准备真实数据

    100 个样本

    x 特征值 (100, 1)
    y_true 目标值 (100, 1)
    y_true = 0.8x + 0.7

假定 x 和 y 之间的关系，满足
    y = kx + b    
    k ~ 0.8 
    b ~ 0.7

    流程分析：
    （100, 1）*（１, 1） = (100, 1)
    y_predict = x * weight(1,1) + bias(1,1)

    １. 构建模型：
    y_predict = tf.malmul(x, weights) + bias

    2. 构造损失函数
    error = tf.reduce_mean(tf.square(y_predict - y_true))

    reduce_mean　均值
    square 平方

    3. 优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate =0.01).minimize(error)



"""

with tf.variable_scope("prepare_data"):
    # 1)　准备数据
    X = tf.random_normal(shape=[100, 1], name="feature")
    y_true = tf.matmul(X, [[0.8]]) + 0.7

with tf.variable_scope("create_model"):
    weights = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), name="Weights")
    bias = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), name="Bias")
    y_predict = tf.matmul(X, weights) + bias

with tf.variable_scope("loss_function"):
    error = tf.reduce_mean(tf.square(y_predict - y_true))

with tf.variable_scope("optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

# 2.1 收集标量
tf.summary.scalar("error", error)
# 2.2 收集高维变量
tf.summary.histogram("weights",weights)
tf.summary.histogram("bias",bias)

# 3. 合并变量
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    #　1.0 创建事件文件
    file_writer = tf.summary.FileWriter("linear/", graph=sess.graph)

    # 　查看初始化模型参数之后的值
    print("parameters before:  weights: %f, bias: %f, loss: %f" % (weights.eval(), bias.eval(), error.eval()))

    # 　开始训练
    for i in range(1000):
        sess.run(optimizer)
        if(i % 50 == 0):
            print("parameters after:  weights: %f, bias: %f, loss: %f, index: %f" % (
            weights.eval(), bias.eval(), error.eval(), i))
        # 运行合并变量操作
        summary = sess.run(merged)
        # 将每次迭代后的变量写入事件文件
        file_writer.add_summary(summary, i)























