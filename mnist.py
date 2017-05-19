# -*- coding:utf-8 -*-


import os
import gzip
import binascii
import struct
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from six.moves.urllib.request import urlretrieve


SOURCE_UR ='http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY="/tmp/mnist-data"
IMAGE_SIZE=28
PIXEL_DEPTH=255
NUM_LABELS=10
NUM_CHANNELS=1
SEED=42
BATCH_SIZE = 60


def maybe_download(filename):
    """A helper to download the data files if not present."""
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Already downloaded', filename)
    return filepath


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
  
    For MNIST data, the number of channels is always 1.

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and dimensions; we know these values.
        bytestream.read(16)

        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and count; we know these values.
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    # Convert to dense 1-hot representation.
    return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)


def processing():
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # data
    train_data = extract_data(train_data_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)

    # labels
    train_labels = extract_labels(train_labels_filename, 60000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # segmenting data into training, test and validation
    VALIDATION_SIZE = 5000

    validation_data = train_data[:VALIDATION_SIZE, :, :, :]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, :, :, :]
    train_labels = train_labels[VALIDATION_SIZE:]

    train_size = train_labels.shape[0]
    print('Validation shape', validation_data.shape)
    print('Train size', train_size)

    return train_data, train_labels, \
      test_data, test_labels, \
      validation_data, validation_labels


def weights_biases():
    # Variable -> 変数を作る
    # truncated_normal -> Tensorを正規分布かつ標準偏差の２倍までのランダムな値で初期化する
    # conv1
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32], # 5x5 filter, depth 32.
            stddev=0.1,
            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))

    # conv2
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
            stddev=0.1,
            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    # fc1
    fc1_weights = tf.Variable(
        tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
            stddev=0.1,
            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

    # fc2
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
            stddev=0.1,
            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    return conv1_weights, conv1_biases, \
        conv2_weights, conv2_biases, \
        fc1_weights, fc1_biases, \
        fc2_weights, fc2_biases


def model(data, train=False, **parameters):
    conv = tf.nn.conv2d(data,
        parameter.conv1_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(conv, parameters.conv1_biases)
    pool = tf.nn.max_pool(relu,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    conv = tf.nn.conv2d(pool,
        parameters.conv2_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, parameters.conv2_biases))
    pool = tf.nn.max_pool(relu,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]
    # fully connetect layer
    hidden = tf.nn.relu(tf.matmul(reshape, parameters.fc1_weights) + parameters.fc1_biases)

    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

    return tf.matmul(hidden, parameters.fc2_weights) + parameters.fc2_biases


def settings_model(**argument):
    conv1_weights, conv1_biases, \
        conv2_weights, conv2_biases, \
        fc1_weights, fc1_biases, \
        fc2_weights, fc2_biases = weights_biases()
    logits = model(argument.train_data_node, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=argument.train_labels_node, logits=logits))

    # L2 regularization
    regularizers = (tf.nn.l2_loss(argument.fc2_weights) + tf.nn.l2_loss(argument.fc1_biases) \
                    + tf.nn.l2_loss(argument.fc2_weigths) + tf.nn.l2_loss(argument.fc2_biases))
    loss += 5e-4 * regularizers

    # Optimizer
    batch = tr.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01,
        batch * BATCH_SIZE,
        train_size,
        0.95,
        staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9) \
        .minimize(loss, global_step=batch)

    train_prediction = tf.nn.softmax(logits)
    validation_prediction = tf.nn.softmax(model(argument.validation_data_node))
    test_prediction = tf.nn.softmax(model(argument.test_data_node))

    return train_prediction, validation_prediction, test_prediction


def error_rate(predictions, labels):
    """Return the error rate and confusions."""
    correct = numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1))
    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    confusions = numpy.zeros([10, 10], numpy.float32)
    bundled = zip(numpy.argmax(predictions, 1), numpy.argmax(labels, 1))
    for predicted, actual in bundled:
        confusions[predicted, actual] += 1
    
    return error, confusions


def training():
    # Train over the first 1/4th of our training set.
    steps = train_size // BATCH_SIZE
    for step in range(steps):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        # Run the graph and fetch some of the nodes.
        _, l, lr, predictions = s.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
        
        # Print out the loss periodically.
        if step % 100 == 0:
            error, _ = error_rate(predictions, batch_labels)
            print('Step %d of %d' % (step, steps))
            print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))
            print('Validation error: %.1f%%' % error_rate(
                  validation_prediction.eval(), validation_labels)[0])


def main():
    train_data, train_labels,\
    test_data, test_labels, \
    validation_data, validation_labels = processing()

    # placeholder -> データが格納される予定地のようなもの
    # 型みたいなイメージで認識
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, NUM_CHANNELS))

    validation_data_node = tf.constant(validation_data)
    test_data_node = tf.constant(test_data)

    # model + loss
    settings_model({'train_data_node': train_data_node, 'validation_data_node': validation_data_node,
        'test_data_node': test_data_node, 'fc1_weights': fc1_weights, ''})
    

    s = tf.InteractiveSession()
    s.as_default()
    tf.global_variables_initializer().run()

    # Grab the first BATCH_SIZE examples and labels.
    batch_data = train_data[:BATCH_SIZE, :, :, :]
    batch_labels = train_labels[:BATCH_SIZE]

    # This dictionary maps the batch data (as a numpy array) to the
    # node in the graph it should be fed to.
    feed_dict = {train_data_node: batch_data,
                 train_labels_node: batch_labels}

    # Run the graph and fetch some of the nodes.
    _, l, lr, predictions = s.run(
      [optimizer, loss, learning_rate, train_prediction],
      feed_dict=feed_dict)

    test_error, confusions = error_rate(test_prediction.eval(), test_labels)
    print('Test error: %.1f%%' % test_error)

if __name__ == '__main__':
    main()

