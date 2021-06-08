# Eager Execution

TensorFlow 的 Eager Execution 是一种命令式编程环境，可立即评估运算，无需构建计算图：运算会返回具体的值，而非构建供稍后运行的计算图。这样能使您轻松入门 TensorFlow 并调试模型，同时也减少了样板代码。要跟随本指南进行学习，请在交互式 python 解释器中运行以下代码示例。

Eager Execution 是用于研究和实验的灵活机器学习平台，具备以下特性：

直观的界面 - 自然地组织代码结构并使用 Python 数据结构。快速迭代小模型和小数据。
更方便的调试功能 - 直接调用运算以检查正在运行的模型并测试更改。使用标准 Python 调试工具立即报告错误。
自然的控制流 - 使用 Python 而非计算图控制流，简化了动态模型的规范。
Eager Execution 支持大部分 TensorFlow 运算和 GPU 加速。

注：启用 Eager Execution 后可能会增加某些模型的开销。我们正在持续改进其性能，如果您遇到问题，请提交错误报告并分享您的基准。


TensorFlow's eager execution is an imperative programming environment that evaluates operations immediately, without building graphs: operations return concrete values instead of constructing a computational graph to run later. This makes it easy to get started with TensorFlow and debug models, and it reduces boilerplate as well. To follow along with this guide, run the code samples below in an interactive python interpreter.

Eager execution is a flexible machine learning platform for research and experimentation, providing:

An intuitive interface—Structure your code naturally and use Python data structures. Quickly iterate on small models and small data.
Easier debugging—Call ops directly to inspect running models and test changes. Use standard Python debugging tools for immediate error reporting.
Natural control flow—Use Python control flow instead of graph control flow, simplifying the specification of dynamic models.
Eager execution supports most TensorFlow operations and GPU acceleration.

