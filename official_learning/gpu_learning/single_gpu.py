import tensorflow as tf

tf.debugging.set_log_device_placement(True)
tf.config.set_soft_device_placement(True)

# 如果指定的设备不存在，则会引发 RuntimeError 错误：.../device:GPU:2 unknown device。
# 当指定的设备不存在时，如果希望 TensorFlow 自动选择存在且支持的设备来执行运算，可以调用
# tf.config.set_soft_device_placement(True)。

# Creates some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)

# try:
#   # Specify an invalid GPU device
#   with tf.device('/device:GPU:2'):
#     a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#     b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#     c = tf.matmul(a, b)
# except RuntimeError as e:
#   print(e)