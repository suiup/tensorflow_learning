import tensorflow as tf


"""
 张量（tensor）     n维数组             n阶张量
 标量     一个数字            0阶张量
 向量     一维数组 [1,2,3]    1阶张量
 矩阵     二维数组 [[1,2,3]
                  [4,5,6]]  2阶张量

"""

tensor1 = tf.constant(4.0) # 标量
tensor2 = tf.constant([1,2,3,4]) # 一维数组
linear_squares = tf.constant([[4], # 二维数组
                              [5],
                              [6],
                              [7]], dtype=tf.int32) # 4行1列
print("tensor1: ", tensor1)
print("tensor2: ", tensor2)
print("linear_squares: ", linear_squares)
# mean = 均值  stddev = 标准差
normal = tf.random_normal(shape=[2,3], mean=1.5, stddev=0.2) # 正态分布

l_cast = tf.cast(linear_squares, dtype=tf.float32)
print("l_cast: ", l_cast)

"""
改变静态形状

只有在形状没有完全固定下来的情况下，才可以通过set_shape(shape) 改变形状
"""
a_p = tf.placeholder(dtype=tf.float32, shape=[None, None])
b_p = tf.placeholder(dtype=tf.float32, shape=[None,10])
c_p = tf.placeholder(dtype=tf.float32, shape=[3,2])
print("a_p: ", a_p) #打印出来后，shape(? ,?) 这样是可以改变形状的
print("b_p: ", b_p)
print("c_p: ", c_p)

# a_p.set_shape([2,3])
# b_p.set_shape([2,10])

# print("a_p: ", a_p)
# print("b_p: ", b_p)
"""
修改动态形状  产生一个新的  tensor

前后修改后的元素数量不能改变

tf.reshape(tensor, shape)
"""
a_p_reshape = tf.reshape(a_p, shape=[2,3,1])
print("a_p_reshape: ", a_p_reshape)


"""
数学运算查看 api
"""



with tf.Session() as sess:
    print(linear_squares.eval())
    print(normal.eval()) # 直接获取值