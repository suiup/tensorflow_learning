import tensorflow as tf
import os


"""
文件读取流程
    多线程　+　多列
        １、构造文件名队列
            file_queue = tf.train.string_input_producer(string_tensor,shuffle=True)
        ２、读取与解码
            读取：
                tf.TextLineReader:
                    阅读文本文件逗号分割值（CSV）格式，默认按行读取
                    return: 读取器实例
                tf.WholeFileReader: 
                    用于读取图片文件
                    return: 读取器实例    
                tf.FixedLengthRecordReader(record_bytes):
                    二进制文件
                    return: 读取器实例    
                tf.TFRecordReader:
                    读取TFRecords文件
                    return: 读取器实例        
                
                key,value = 读取器.read(file_queue)
                key: 文件名
                value: 一个样本
                
                
                
                由于默认只会读取一个样本，所以如果想要进行批处理，需要使用
                tf.train.batch 或　tf.train.shuffle_batch 进行批处理操作，便于以后指定每批次多个样本的训练
        
        
            解码：
                tf.decode_csv:
                    解码文本文件内容
                
                tf.image.decode_jpeg(contents)
                    将JPEG编码的图像解码为unit8张量
                    return: unit8张量，　３－D形状[height,width,channels]    
        
                tf.image.decode_png(contents)
                    将PNG编码的图像解码为unit8张量
                    return: 张量类型，３-Dx形状[height, width, channels]
                
                tf.decode_raw: 
                    解码二进制文件内容
                    与 tf.FixedLengthRecordReader 搭配使用，　二进制读取为unit8类型        
                
                解码阶段，默认所有的内容都加码成tf.unit8类型，如果之后需要转换成指定类型，则可以使用tf.cast()进行转换
            
            
        ３、批处理队列
            解码之后，可以直接获取默认的一个样本内容了，但是如果想要获取多个样本，需要加入到新的队列进行批处理
            
            tf.train.batch(tensor,batch_size,num_threads=1, capacity=32,name=None)
            
                读取指定大小（个数）的张量
                tensors: 可以是包含张量的列表，批处理的内容放到列表中
                batch_size：　从列表中读取的批处理的大小
                num_threads: 进入队列的线程数
                capacity: 整数，队列中元素的最大数量
                return: tensors 
            
            tf.train.shuffle_batch
        
        ４、手动开启线程
        
            以上用到的队列都是　tf.train.QueueRunner对象
            
            每个QueueRunner都负责一个阶段，tf.train.start_queue_runners　函数会要求
            图中的每个QueueRunner启动它的运行队列操作的线程。　这些操作需要在会话中开启
            
            tf.train.start_queue_runners(sess=None,coord=None)
                收集图中所有队列线程，默认同时开启线程
                sess: 所在的会话
                coord: 线程协调器
                return: 返回所有线程
            
            tf.train.Coordinator()
                线程协调器
                request_stop()　请求停止
                should_stop() 询问是否结束
                join(threads=None,stop_grace_period_secs=120): 回收线程
                return 线程协调员实例    
    

有三种获取数据到　Tensorflow 程序的方法
    １、QueueRunner: 基于队列的输入管道，从Tensorflow图形开头的文件中读取数据
    ２、Feeding: 运行每一步时，　Python代码提供数据
    ３、预加载数据：　Tensorflow图中的张量包含所有数据
    



    
    

"""















