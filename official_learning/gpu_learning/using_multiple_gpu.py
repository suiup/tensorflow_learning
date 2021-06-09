import tensorflow as tf

# 针对多个 GPU 进行开发可让模型使用额外的资源进行扩展。
# 如果在具有单个 GPU 的系统上进行开发，可以使用虚拟设备模拟多个 GPU。
# 这样，无需额外的资源就可以轻松对多 GPU 设置进行测试。

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)