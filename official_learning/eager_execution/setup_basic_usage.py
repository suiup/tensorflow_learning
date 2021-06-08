import os
import tensorflow as tf
import cProfile

# enable eager execution
tf.enable_eager_execution()

#In Tensorflow 2.0, eager execution is enabled by default.
tf.executing_eagerly()

# Now you can run TensorFlow operations and the results will return immediately:
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)

# Broadcasting support
b = tf.add(a, 1)
print(b)

# Operator overloading is supported
print(a * b)

# Use NumPy values
import numpy as np

c = np.multiply(a, b)
print(c)
