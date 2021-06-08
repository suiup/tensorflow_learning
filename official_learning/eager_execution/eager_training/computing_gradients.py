import tensorflow as tf

tf.enable_eager_execution()

w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w  # w^2 --> loss' = 2w --> grad = 2 * 1 = 2

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)