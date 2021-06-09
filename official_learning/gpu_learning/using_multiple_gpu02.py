import tensorflow as tf

#建立可供运行时使用的多个逻辑 GPU 后，我们可以通过 tf.distribute.Strategy 或手动放置来利用多个 GPU。

# 使用 tf.distribute.Strategy
# 使用多个 GPU 的最佳做法是使用 tf.distribute.Strategy。下面是一个简单示例

tf.debugging.set_log_device_placement(True)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  inputs = tf.keras.layers.Input(shape=(1,))
  predictions = tf.keras.layers.Dense(1)(inputs)
  model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
  model.compile(loss='mse',
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))