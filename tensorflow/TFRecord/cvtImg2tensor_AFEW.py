import tensorflow as tf
import sys
import numpy as np
import cv2
from six.moves import xrange  # range보다 빠르고 메모리가 효율적으로 동
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage.io import imshow

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, class_id):  # data(이미지)를 encoding, 형식을 정의하여 encoding하는 것
    return tf.train.Example(features=tf.train.Features(feature={'image/encoded': bytes_feature(image_data),
                                                                'image/format ': bytes_feature(image_format),
                                                                'image/class/label': int64_feature(class_id)}))

def list_all_files(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            joined = os.path.join(root, filename)
            if extensions is None or ext.lower() in extensions:
                yield joined

def list_box(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        boxes = []
        boxes.append(dirnames)
        for box in boxes:
            return box

def box_paths(directory, box):
    joined = []
    for i in xrange(len(box)):
        joined.append(os.path.join(directory, str(box[i])))
    return joined

### main
DATA_ROOT = '/home/lemin/nas/DMSL/AFEW/Train_only'
SAVE_ROOT = '/home/lemin/nas/DMSL/AFEW/Tensor/Train_only'
X_DATA = SAVE_ROOT + '/x_train_color.tfrecord'
Y_DATA = SAVE_ROOT + '/y_train_color.tfrecord'

dataset_type = 'jpg'
LABEL = {
    'angry': '/Angry',
    'disgust': '/Disgust',
    'fear': '/Fear',
    'happy': '/Happy',
    'neutral': '/Neutral',
    'sad': '/Sad',
    'surprise': '/Surprise'
}
paths_dict = defaultdict(lambda : [])
label_dict = defaultdict(lambda : [])

num = 0
for key, value in LABEL.items():
    label_path = DATA_ROOT + value
    box = list(list_box(label_path, ['.jpg', '.png']))
    paths = box_paths(label_path, box)
    temp_num = 0
    for path in paths:
        temp = list(list_all_files(path, ['.jpg', '.png']))
        paths_dict[key] += temp
        num += len(temp)
        temp_num += len(temp)
    print("Loaded", temp_num, key + ' file lists')
print('Total # of paths : {}'.format(num))

for key in list(paths_dict.keys()):
    if key == 'angry':
        label = 0
    elif key == 'disgust':
        label = 1
    elif key == 'fear':
        label = 2
    elif key == 'happy':
        label = 3
    elif key == 'neutral':
        label = 4
    elif key == 'sad':
        label = 5
    elif key == 'surprise':
        label = 6

    for path in paths_dict[key]:
        label_dict[key] += [(path, label)]

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_png(image_placeholder)  # encoding된 이미지
        dataset_len = 0
        with tf.python_io.TFRecordWriter(X_DATA) as tfrecord_writer:  # 텐서플로우 형식으로 writing하는 파일을 tranining_filename에 저장
            for emot in LABEL.keys():
                lists = label_dict[emot]
                per_emot_num = 0
                for path, label in lists:
                    image = cv2.imread(path)
                    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                    # image_string = sess.run(encoded_image, feed_dict={image_placeholder : image})
                    image_string = image.tostring()
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(label))
                    tfrecord_writer.write(example.SerializeToString())
                    per_emot_num += 1
                dataset_len += per_emot_num
                print(emot + ' : ', per_emot_num)
            print('Total dataset : ', dataset_len)

# tfRecord 장점 : encoding되서 데이터를 빠르게 불러올 수 있음
# tfRecord 단점 : 수정x, 새로 encoding해야 함