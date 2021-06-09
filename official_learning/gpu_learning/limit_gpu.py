import tensorflow as tf
# 默认情况下，TensorFlow 会映射进程可见的所有 GPU（取决于 CUDA_VISIBLE_DEVICES）的几乎全部内存。
# 这是为了减少内存碎片，更有效地利用设备上相对宝贵的 GPU 内存资源。
# 为了将 TensorFlow 限制为使用一组特定的 GPU，我们使用 tf.config.experimental.set_visible_devices 方法。


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)