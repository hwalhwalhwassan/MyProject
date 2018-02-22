
import matplotlib
# matplotlib.use('qt5agg')
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

data_path = '/home/lemin/nas/DMSL/AFEW/Tensor/Train_only/train_jpeg.tfrecord'

keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                    'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))}

filename_queue = tf.train.string_input_producer([data_path])

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(serialized_example, features=keys_to_features)

# image = tf.decode_raw(features['image/encoded'], tf.uint8)
image = tf.image.decode_jpeg(features['image/encoded'], channels=3) # if consist of jpeg, using decode_jpeg instead of decode_raw
image = tf.reshape(image, [224,224,3])
label = tf.cast(features['image/class/label'], tf.int64)

images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for batch_index in range(5):
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)

        for j in range(6):
            plt.subplot(2, 3, j + 1)
            plt.imshow(img[j, ...])
            print(lbl[j])
        plt.show()

    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()