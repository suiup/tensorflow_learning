import  tensorflow as tf

tf.enable_eager_execution()


logdir = "./tb/"
# writer = tf.summary.FileWriter(logdir)
# writer = tf.contrib.summary(logdir)
writer = tf.summary.create_file_writer(logdir)

steps = 1000
with writer.as_default():  # or call writer.set_as_default() before the loop.
  for i in range(steps):
    step = i + 1
    # Calculate loss with your real train function.
    loss = 1 - 0.001 * step
    if step % 100 == 0:
      tf.summary.scalar('loss', loss, step=step)