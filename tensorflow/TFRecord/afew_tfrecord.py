
import numpy as np
import os
import tensorflow as tf
import keras
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras import layers
from keras.callbacks import Callback
import matplotlib.pyplot as plt

if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend, '
                       'because it requires TFRecords, which '
                       'are not supported on other platforms.')


class EvaluateInputTensor(Callback):
    """ Validate a model which does not expect external numpy data during training.

    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
    Instead, create a second model for validation which is also configured
    with input tensors and add it to the `EvaluateInputTensor` callback
    to perform validation.

    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.

    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, model, steps, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)


def cnn_layers(x_train_input):
    x = layers.Conv2D(32, (3, 3),
                      activation='relu', padding='valid')(x_train_input)
    x = layers.MaxPooling2D(pool_size=(5, 5))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(5, 5))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x_train_out = layers.Dense(num_classes,
                               activation='softmax',
                               name='x_train_out')(x)
    return x_train_out

def read_and_decode(tfrecord_path, batch_shape):
    keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))}

    filename_queue = tf.train.string_input_producer([tfrecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=keys_to_features)

    # image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.reshape(image, batch_shape[1:])
    label = tf.cast(features['image/class/label'], tf.int64)

    return image, label

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list = "0"
sess = K.get_session()
# sess = tf.Session(config=config)

batch_size = 10
batch_shape = (batch_size, 224, 224, 3)
epochs = 5
num_classes = 7
num_example = 11062

capacity = 100
min_after_dequeue = 30
enqueue_many = True

# TODO : read & decode function
data_path = '/home/lemin/nas/DMSL/AFEW/Tensor/Train_only/train_jpeg.tfrecord'
images, labels = read_and_decode(data_path, batch_shape=batch_shape)

x_train_batch, y_train_batch = tf.train.shuffle_batch(
    tensors=[images, labels],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue,
    #enqueue_many=enqueue_many,
    num_threads=4)

x_train_batch = tf.cast(x_train_batch, tf.float32)
x_train_batch = tf.reshape(x_train_batch, shape=batch_shape)

y_train_batch = tf.cast(y_train_batch, tf.int32)
y_train_batch = tf.one_hot(y_train_batch, num_classes)

x_batch_shape = x_train_batch.get_shape().as_list()
y_batch_shape = y_train_batch.get_shape().as_list()

model_input = layers.Input(tensor=x_train_batch)
model_output = cnn_layers(model_input)
train_model = keras.models.Model(inputs=model_input, outputs=model_output)

# Pass the target tensor `y_train_batch` to `compile`
# via the `target_tensors` keyword argument:
train_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[y_train_batch])
train_model.summary()

# x_test_batch, y_test_batch = tf.train.batch(
#     tensors=[data.test.images, data.test.labels.astype(np.int32)],
#     batch_size=batch_size,
#     capacity=capacity,
#     enqueue_many=enqueue_many,
#     num_threads=8)
#
# # Create a separate test model
# # to perform validation during training
# x_test_batch = tf.cast(x_test_batch, tf.float32)
# x_test_batch = tf.reshape(x_test_batch, shape=batch_shape)
#
# y_test_batch = tf.cast(y_test_batch, tf.int32)
# y_test_batch = tf.one_hot(y_test_batch, num_classes)
#
# x_test_batch_shape = x_test_batch.get_shape().as_list()
# y_test_batch_shape = y_test_batch.get_shape().as_list()
#
# test_model_input = layers.Input(tensor=x_test_batch)
# test_model_output = cnn_layers(test_model_input)
# test_model = keras.models.Model(inputs=test_model_input, outputs=test_model_output)
#
# # Pass the target tensor `y_test_batch` to `compile`
# # via the `target_tensors` keyword argument:
# test_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
#                    loss='categorical_crossentropy',
#                    metrics=['accuracy'],
#                    target_tensors=[y_test_batch])

# Fit the model using data from the TFRecord data tensors.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)

train_model.fit(epochs=epochs,
                steps_per_epoch=int(np.ceil(num_example / float(batch_size))))

# Save the model weights.
train_model.save_weights('saved_wt.h5')

# Clean up the TF session.
coord.request_stop()
coord.join(threads)
K.clear_session()
# # Second Session to test loading trained model without tensors
# x_test = np.reshape(data.test.images, (data.test.images.shape[0], 28, 28, 1))
# y_test = data.test.labels
# x_test_inp = layers.Input(shape=(x_test.shape[1:]))
# test_out = cnn_layers(x_test_inp)
# test_model = keras.models.Model(inputs=x_test_inp, outputs=test_out)
#
# test_model.load_weights('saved_wt.h5')
# test_model.compile(optimizer='rmsprop',
#                    loss='categorical_crossentropy',
#                    metrics=['accuracy'])
# test_model.summary()
#
# loss, acc = test_model.evaluate(x_test,
#                                 keras.utils.to_categorical(y_test),
#                                 batch_size=batch_size)
# print('\nTest accuracy: {0}'.format(acc))
#