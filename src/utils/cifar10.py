import tensorflow as tf
import numpy as np
import os
import sys
from .paths import CIFAR10_PATH

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_FILES = 5
NUM_TRAIN = 10000 * NUM_DATA_FILES
NUM_TEST = 10000

NUM_PARALLEL_CALLS = 24


def record_dataset(filenames):
  label_bytes = 1
  image_bytes = DEPTH * HEIGHT * WIDTH
  record_bytes = label_bytes + image_bytes
  return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def get_filenames(training):
  data_dir = os.path.join(CIFAR10_PATH, 'cifar-10-batches-bin')
  if training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record):
  """Parse a CIFAR-10 record from value."""
  # Every record consists of a label followed by the image, with a fixed number
  # of bytes for each.
  label_offset = 0
  label_bytes = 1
  image_bytes = DEPTH * HEIGHT * WIDTH
  record_bytes = label_bytes + image_bytes

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32.
  label = tf.cast(record_vector[label_offset], tf.int32)
  label = tf.one_hot(label, NUM_CLASSES)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      record_vector[label_offset + label_bytes:record_bytes],
      [DEPTH, HEIGHT, WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
  return image, label


def preprocess_image(image, training):
  """Preprocess a single image of layout [height, width, depth]."""
  if training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)
    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)

  # transpose image back to depth major
  image = tf.transpose(image, [2, 1, 0])

  return image


def input_fn(training, batch_size):
  dataset = record_dataset(get_filenames(training))

  if training:
    dataset = dataset.shuffle(buffer_size=NUM_TRAIN)

  dataset = dataset.map(parse_record, num_parallel_calls=NUM_PARALLEL_CALLS)
  dataset = dataset.map(lambda image, label:
                        (preprocess_image(image, training), label),
                        num_parallel_calls=NUM_PARALLEL_CALLS)

  dataset = dataset.prefetch(2 * batch_size)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels
