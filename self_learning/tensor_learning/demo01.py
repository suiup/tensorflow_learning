import tensorflow as tf


"""
 张量（tensor）     n维数组             n阶张量
 标量     一个数字            0阶张量
 向量     一维数组 [1,2,3]    1阶张量
 矩阵     二维数组 [[1,2,3]
                  [4,5,6]]  2阶张量

"""

tensor1 = tf.constant(4.0)
tensor2 = tf.constant([1,2,3,4])
linear_squares = tf.constant([[4],
                              [5],
                              [6],
                              [7]], dtype=tf.int32) # 4行1列
print("tensor1: ", tensor1)
print("tensor2: ", tensor2)
print("linear_squares: ", linear_squares)
# mean = 均值  stddev = 标准差
normal = tf.random_normal(shape=[2,3], mean=1.5, stddev=0.2) # 正态分布


with tf.Session() as sess:
    print(linear_squares.eval())
    print(normal.eval())