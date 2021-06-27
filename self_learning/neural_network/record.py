import tensorflow as tf


"""
神经网络：
    输入层　隐藏层　输出层
    
    softmax 回归
        logits加上softmax 映射，----多分类问题
            
    损失函数
        交叉熵损失
        总损失
        最小二乘法 - 线性回归的损失 - 均方误差
        
    优化损失函数
        
    tf.nn.softmax_cross_entropy_with_logits(labels=None, logits=None, name=None)
    
        计算logits和labels之间的交叉熵损失
        labels: 标签值（真实值）
        logits: 样本加权之后的值
        return: 返回损失值列表
    
    tf.reduce_mean(input_tensor)
        计算张量的尺寸的元素平均值    
"""








