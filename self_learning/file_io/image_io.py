import tensorflow as tf
import os


"""
图片数据
    图像基本知识
        特征抽取
            文本　－　数值（二维数组　shape(n_samples, m_features)）
            字典　－　数值　（二维数组　shape(n_samples, m_features)）
            图片　－　数值　（三维数组　shape(图片长度，图片宽度，图片通道数)）
                组成图片的最基本单位是像素
            
            1、图片三要素
                灰度图　[长，宽，１]
                    每个像素点　[0, 255]
                
                彩色图 [长，宽，３]
                    每个像素点用三个　[0, 255]  
            ２、张量形状
                Tensor(指令名称，shape, dtype)
                一张图片：　shape = （height,width,channels）
                多张图片：　shape = (batch, height, width, channels)
    
    图片特征值处理
        样本和样本的形状不统一，没办法进行批量操作和运算
        
        需要将图片缩放到统一的大小
        
        tf.image.resize_images(images, size)
            缩放图片大小
            images: 4-D形状[batch, height, width, channels]　或　3-D　形状的张量 [height, width, channels]    
            size: 1-D　int32　张量，　new_height,new_width　图像的尺寸
        return: ４-D格式或者　3-D　格式的图片
        
    
    数据格式
        存储： uint8 节约空间
        矩阵计算：　float32 提高精度    
    
"""


"""
    案例，图片处理
        １、构造文件名队列
        ２、读取与解码
            使样本的形状和类型进行统一
        ３、批处理    
"""


def picture_read(file_list):
    file_queue = tf.train.string_input_producer(file_list)
    # 读取阶段
    reader = tf.WholeFileReader()
    # key　文件名　value:一张图片的原始编码形式

    key, value = reader.read(file_queue)
    print("key: ",key)
    print("value: ",value)
    # 解码阶段
    image = tf.image.decode_png(value)
    print("image", image)



    #　图像的形状，　类型修改
    image_resize = tf.image.resize_images(image, [200, 200])
    print("image_resize: ", image_resize)

    # 　静态形状修改
    image_resize.set_shape(shape=[200, 200, 3])

    #　批处理
    image_batch = tf.train.batch([image_resize], batch_size=100, num_threads=1, capacity=20)
    print("image_batch", image_batch)



    with tf.Session() as sess:
        #　开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        key_new, value_new, image_new, image_resize_new = sess.run([key, value, image, image_resize])
        print("key_new: ", key_new)
        print("value_new: ", value_new)
        print("image_new: ", image_new)
        print("image_resize: ", image_resize_new)
        coord.request_stop()
        coord.join(threads)

    return None


if __name__ == "__main__":
    filename = os.listdir("./dog")
    file_list = [os.path.join("./dog", file) for file in filename]
    print(file_list)
    picture_read(file_list)






















