import tensorflow as tf


"""

模型的保存和加载

tf.train.Saver(var_list=None, max_to_keep=5)
    保存和加载模型，保存文件格式　checkpoint 文件
    var_list 指定将要保存和还原的变量，可以作为一个　dict　或一个列表传递
    max_to_keep: 指示要保留的最近检查点文件的最大数量。创建新文件时，会删除旧的文件，如果没有或０，　则保留所有检查点文件。　默认为５
    
    
１）　实例化　Saver
2）　保存
    saver.save(sess, path)    
3) 加载
    saver.restore(sess, path)
        
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


#　创建　Saver对象
saver = tf.train.Saver()

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

        #　保存模型
        if( i % 100 == 0):
            saver.save(sess, "./temp/model/my_linear.ckpt")

