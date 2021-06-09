import tensorflow as tf
# 手动设备放置
# 如果您希望在自己选择的设备上执行特定运算，而不是在自动选择的设备上执行，
# 则可以使用 with tf.device 创建设备上下文。创建完成后，该上下文中的所有运算都会在同一指定设备上运行。

tf.debugging.set_log_device_placement(True)

# Place tensors on the CPU
with tf.device('/CPU:0'):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

c = tf.matmul(a, b)
print(c)