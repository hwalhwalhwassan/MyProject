import tensorflow as tf
import sys
import numpy as np
import pickle
import cv2
from six.moves import xrange # range보다 빠르고 메모리가 효율적으로 동
LABELS_FILENAME = 'labels.txt'

  
def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
  
def image_to_tfexample(image_data, image_format, class_id): # data(이미지)를 encoding, 형식을 정의하여 encoding하는 것
    return tf.train.Example(features=tf.train.Features(feature={'image/encoded': bytes_feature(image_data),
                                                                'image/format ': bytes_feature(image_format),
                                                                'image/class/label': int64_feature(class_id)}))
  
### main
convert_dir = '/home/lemin/Desktop/EmotiW_aux'
dataset_dir = '/home/lemin/Desktop/EmotiW_aux/cifar10'

dataset_type = 'png'

training_filename = '%s/cifar10_train.tfrecord' % (convert_dir)
test_filename     = '%s/cifar10_test.tfrecord'  % (convert_dir)

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_png(image_placeholder) # encoding된 이미지
        dataset_len = 0
        with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer: # 텐서플로우 형식으로 writing하는 파일을 tranining_filename에 저장
            for batch in range(1,6):
                with open('%s/data_batch_%d'%(dataset_dir, batch), 'rb') as fo: # 옵션 'r'은 유니코드, 옵션 'rb'은 바이너리
                    img_queue = pickle.load(fo, encoding='bytes') # binary data를 encoding
                
                for n in xrange(img_queue[b'data'].shape[0]): # 여기서부터 tfrecord_writer 파일에 pickle에 있는 데이터를 write
                    image = img_queue[b'data'][n] # pickle로 읽어오면 dictionary형태이기 때문에 b'data'를 key값으로 해서 데이터에 접근
                    # image.shape를 보면 (10000, 3072)인데 5개의 batch가 있으므로 cifar10의 5만개의 데이터를 나타내고 3072는 32x32x3의 이미지 사이즈
                    image = np.transpose(image.reshape(3,32,32),(1,2,0))
                    label = img_queue[b'labels'][n]
                    image_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: image})
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(label))
                    tfrecord_writer.write(example.SerializeToString())
                    sys.stdout.write('\r>> Reading train_dataset images %d/%d' 
                                     % (n+dataset_len+1 , 5*img_queue[b'data'].shape[0]))
                dataset_len += img_queue[b'data'].shape[0]
        sys.stdout.write('\r\n') 
        # with tf.python_io.TFRecordWriter(test_filename) as tfrecord_writer:
        #     with open('%s/test_batch'%dataset_dir, 'rb') as fo:
        #         img_queue = pickle.load(fo, encoding='bytes')
        #
        #     for n in xrange(img_queue[b'data'].shape[0]):
        #         image = img_queue[b'data'][n]
        #         image = np.transpose(image.reshape(3,32,32),(1,2,0))
        #         label = img_queue[b'labels'][n]
        #         image_string = sess.run(encoded_image,
        #                           feed_dict={image_placeholder: image})
        #         example = image_to_tfexample(image_string, str.encode(dataset_type), int(label))
        #         tfrecord_writer.write(example.SerializeToString())
        #         sys.stdout.write('\r>> Reading test_dataset images %d/%d'
        #                          % (n+1 , img_queue[b'data'].shape[0]))
                    
# tfRecord 장점 : encoding되서 데이터를 빠르게 불러올 수 있음
# tfRecord 단점 : 수정x, 새로 encoding해야 함