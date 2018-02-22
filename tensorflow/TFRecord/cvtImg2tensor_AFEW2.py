import tensorflow as tf
import numpy as np
import cv2
import sys
from collections import defaultdict
import os
from six.moves import xrange

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

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.astype(np.float32)
    return img


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, class_id):
    features = {'image/encoded': bytes_feature(image_data),
                'image/format' : bytes_feature(image_format),
                'image/class/label': int64_feature(class_id)}
    return tf.train.Example(features=tf.train.Features(feature=features))

if __name__ == "__main__":
    # TODO : confirm here!
    DATA_ROOT = '/home/lemin/nas/DMSL/AFEW/afew-mtcnn/REARRANGE/Train_only'
    SAVE_ROOT = '/home/lemin/nas/DMSL/AFEW/Tensor/Train_only/train_raw.tfrecord'

    dataset_type = 'raw'

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

    train_addrs = []
    train_labels = []
    for _, item in list(label_dict.items()):
        addrs, labels = zip(*item)
        train_addrs += list(addrs)
        train_labels += list(labels)

    slim = tf.contrib.slim
    with tf.device('/cpu:0'):
        sess = tf.Session()
        # image_placeholder = tf.placeholder(dtype=tf.uint8)
        # encoded_image = tf.image.encode_jpeg(image_placeholder)
        with tf.python_io.TFRecordWriter(SAVE_ROOT) as tfrecord_writer:
            for i in xrange(len(train_addrs)):
                if not i % 1000:
                    print('Train data: {}/{}'.format(i, len(train_addrs)))
                image = load_image(train_addrs[i])
                label = train_labels[i]

                # image_string = sess.run(encoded_image, feed_dict={image_placeholder: image})
                image_string = image.tostring()
                example = image_to_tfexample(image_string, str.encode(dataset_type), int(label))
                tfrecord_writer.write(example.SerializeToString())