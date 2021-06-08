import tensorflow as tf

# tf.keras.metrics are stored as objects. Update a metric by passing the new data to the callable,
# and retrieve the result using the tf.keras.metrics.result method, for example:


m = tf.keras.metrics.Mean("loss")
m(0)
m(5)
m.result()  # => 2.5   (0+5)/2 = 2.5
m([8, 9])
m.result()  # => 5.5  (0+5+8+9)/4 = 5.5
