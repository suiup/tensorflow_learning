# Computing gradients
# 计算梯度
自动微分对实现机器学习算法（例如用于训练神经网络的反向传播）十分有用。在 Eager Execution 期间，请使用 tf.GradientTape 跟踪运算以便稍后计算梯度。

您可以在 Eager Execution 中使用 tf.GradientTape 来训练和/或计算梯度。这对复杂的训练循环特别有用。

由于在每次调用期间都可能进行不同运算，所有前向传递的运算都会记录到“条带”中。要计算梯度，请反向播放条带，然后丢弃。特定 tf.GradientTape 只能计算一个梯度；后续调用会引发运行时错误。

Automatic differentiation is useful for implementing machine learning algorithms such as backpropagation for training neural networks. During eager execution, use tf.GradientTape to trace operations for computing gradients later.

You can use tf.GradientTape to train and/or compute gradients in eager. It is especially useful for complicated training loops.

Since different operations can occur during each call, all forward-pass operations get recorded to a "tape". To compute the gradient, play the tape backwards and then discard. A particular tf.GradientTape can only compute one gradient; subsequent calls throw a runtime error.

# Train a model
The example creates a multi-layer model that classifies the standard MNIST handwritten digits. It demonstrates the optimizer and layer APIs to build trainable graphs in an eager execution environment.

# Variables and optimizers
tf.Variable objects store mutable tf.Tensor-like values accessed during training to make automatic differentiation easier.

The collections of variables can be encapsulated into layers or models, along with methods that operate on them. See Custom Keras layers and models for details. The main difference between layers and models is that models add methods like Model.fit, Model.evaluate, and Model.save.

For example, the automatic differentiation example above can be rewritten:


# Summaries and TensorBoard

TensorBoard 是一种可视化工具，用于了解、调试和优化模型训练过程。它使用在执行程序时编写的摘要事件。

您可以在 Eager Execution 中使用 tf.summary 记录变量摘要。例如，要每 100 个训练步骤记录一次 loss 的摘要


TensorBoard is a visualization tool for understanding, debugging and optimizing the model training process. It uses summary events that are written while executing the program.

You can use tf.summary to record summaries of variable in eager execution. For example, to record summaries of loss once every 100 training steps:
