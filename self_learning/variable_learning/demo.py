import tensorflow as tf

"""
创建变量


一般大写的就都是 操作
"""

"""

可以使用tf.variable_scope()修改变量的命名空间

代码模块化，比较清晰

"""
with tf.variable_scope("scope"):
    a = tf.Variable(initial_value=50)
    b = tf.Variable(initial_value=40)
with tf.variable_scope("c_scope"):
    c = tf.add(a, b)

print("a: ", a)
print("b: ", b)
print("c: ", c)

"""
变量需要初始化
tf.global_variables_initializer()
"""
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 运行初始化
    sess.run(init)
    a_value, b_value, c_value = sess.run([a,b,c])
    print("a_value: ", a_value)
    print("b_value: ", b_value)
    print("c_value: ", c_value)


